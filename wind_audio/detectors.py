"""Training-free L1 configuration and event-window extraction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .events import event_features
from .features import FeatureConfig
from .l1 import spectral_change
from .windowing import merge_window, track_windows

DEFAULT_POSTPROCESS = {"min_duration": 0.06, "merge_gap": 0.35, "low_ratio": 0.6, "padding": 0.05,
                       "reference_s": 0.6, "release_db": 3.0, "quiet_s": 0.25,
                       "settle_s": 0.6, "stability_db": 1.5, "min_prominence": 0.3,
                       "relative_floor_db": 15.0}


class SpectralChangeDetector:
    detector_type = "spectral_change"
    score_type = "spectral_change_db"
    version = "spectral_event_v2"

    def __init__(self, feature_config=None, threshold=6.0, postprocess=None):
        self.feature_config = feature_config or FeatureConfig()
        self.threshold = float(threshold)
        unknown = set(postprocess or {}) - set(DEFAULT_POSTPROCESS)
        if unknown:
            raise ValueError(f"不支持的后处理参数: {sorted(unknown)}")
        self.postprocess = {**DEFAULT_POSTPROCESS, **(postprocess or {})}
        self.validate(self.threshold, self.postprocess)

    @staticmethod
    def validate(threshold, pp):
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("频谱变化阈值必须为有限正数")
        if not all(np.isfinite(v) for v in pp.values()):
            raise ValueError("后处理参数必须有限")
        if not 0 < pp["low_ratio"] < 1 or any(pp[k] < 0 for k in ("min_duration", "merge_gap", "padding")):
            raise ValueError("low_ratio 需在 (0,1) 内；持续时间、合并间隔、边界补充不能为负")
        if any(pp[k] <= 0 for k in ("reference_s", "release_db", "quiet_s", "settle_s",
                                    "stability_db", "relative_floor_db")) or not 0 < pp["min_prominence"] < 1:
            raise ValueError("事件跟踪时间/阈值须为正数；min_prominence 须在 (0,1) 内")

    def score(self, bundle):
        self.feature_config.assert_compatible(bundle.config)
        return spectral_change(bundle)

    def to_dict(self):
        return {"schema_version": 1, "detector": self.detector_type, "version": self.version,
                "threshold": self.threshold, "feature_config": self.feature_config.to_dict(),
                "postprocess": self.postprocess}

    def predict_events(self, bundle, threshold=None, postprocess=None):
        return self.analyze(bundle, threshold, postprocess)[0]

    def analyze(self, bundle, threshold=None, postprocess=None):
        threshold = self.threshold if threshold is None else float(threshold)
        if set(postprocess or {}) - set(DEFAULT_POSTPROCESS):
            raise ValueError("不支持的事件参数")
        pp = {**self.postprocess, **(postprocess or {})}
        self.validate(threshold, pp)
        score = self.score(bundle)
        raw, traces = track_windows(bundle, threshold, pp, score)
        # Padding belongs to the detection window, not the separate listening context.
        # Merge overlapping padded windows so L2 never receives duplicate candidates.
        windows = []
        for event in raw:
            item = dict(event)
            item["start_s"] = max(0.0, event["start_s"] - pp["padding"])
            item["end_s"] = min(bundle.duration_s, event["end_s"] + pp["padding"])
            if windows and item["start_s"] <= windows[-1]["end_s"] + 1e-9:
                merge_window(windows[-1], item)
            else:
                windows.append(item)
        output = []
        for i, event in enumerate(windows, 1):
            a, b = event["i0"], event["i1"]
            segment = score[a:b + 1]
            event.update(duration_s=event["end_s"] - event["start_s"],
                         peak_score=float(segment.max()), mean_score=float(segment.mean()),
                         peak_time_s=float(bundle.times[a + int(np.argmax(segment))]),
                         at_edge=event["start_s"] <= 0 or event["end_s"] >= bundle.duration_s)
            details = event_features(bundle, event, score, self.score_type)
            output.append({**details, **event, "event_id": f"{bundle.audio_id}_ev{i:03d}",
                           "score_type": self.score_type, "threshold": threshold})
        traces["active"] = np.zeros(len(bundle.times), dtype=bool)
        for event in output:
            traces["active"] |= (bundle.times >= event["start_s"]) & (bundle.times < event["end_s"])
        return output, traces


def load_detector(config_path=None):
    if config_path is None:
        return SpectralChangeDetector()
    path = Path(config_path)
    if not path.is_file():
        raise ValueError("请传入 L1 JSON 配置文件，不再加载旧训练模型目录")
    obj = json.loads(path.read_text(encoding="utf-8"))
    allowed = {"schema_version", "detector", "version", "threshold", "feature_config", "postprocess"}
    if (set(obj) - allowed or obj.get("schema_version") != 1 or
            obj.get("detector") != "spectral_change" or obj.get("version") != SpectralChangeDetector.version):
        raise ValueError("请使用 spectral_event_v2 配置；旧边缘检测配置不可静默复用")
    return SpectralChangeDetector(FeatureConfig.from_dict(obj["feature_config"]),
                                  obj["threshold"], obj["postprocess"])
