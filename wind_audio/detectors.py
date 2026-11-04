"""L1 detectors: the legacy spectral-change detector and the stability detector.

Two training-free algorithms share one interface (``score`` / ``analyze`` /
``predict_events`` / ``to_dict``) so the detection and test entry points stay
identical:

* ``SpectralChangeDetector`` -- v2 legacy, kept for reproducing old reports and
  loading ``config/l1.json`` unchanged.
* ``StabilityDetector`` -- v3, multi-timescale stability with a local noise
  tolerance and an explicit event state machine (``config/l1_v3.json``).

Configurations never change meaning silently: ``schema_version`` selects the
detector, and an unknown version is an error.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .events import event_features
from .features import FeatureConfig
from .l1 import spectral_change
from .stability import StabilityConfig, analyze as analyze_stability
from .state import EventConfig, complement_events, track_events, track_stable_segments
from .windowing import merge_window, track_windows

DEFAULT_POSTPROCESS = {"min_duration": 0.06, "merge_gap": 0.35, "low_ratio": 0.6, "padding": 0.05,
                       "reference_s": 0.6, "release_db": 3.0, "quiet_s": 0.25,
                       "settle_s": 0.6, "stability_db": 1.5, "min_prominence": 0.3,
                       "relative_floor_db": 15.0}


class SpectralChangeDetector:
    """Legacy spectral-change detector (configuration schema_version 1)."""

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
        traces["curves"] = [
            {"key": "change_db", "label": "频谱变化", "unit": "dB", "values": traces["change_db"],
             "threshold": threshold, "title": "L1 变化分数（dB，不是概率）",
             "note": "触发阈值为虚线；分数回落不代表事件结束。"},
            {"key": "activity_db", "label": "持续活动", "unit": "dB", "values": traces["activity_db"],
             "threshold": pp["release_db"], "title": "相对事件前背景的持续活动（dB）",
             "note": "释放阈值为点线；需持续稳定才结束。"},
        ]
        return output, traces


class StabilityDetector:
    """Multi-timescale stability detector (configuration schema_version 2)."""

    detector_type = "spectral_stability"
    score_type = "change_z"
    version = "spectral_stability_v3"

    def __init__(self, feature_config=None, stability=None, events=None):
        self.feature_config = feature_config or FeatureConfig()
        self.stability = stability or StabilityConfig()
        self.events = events or EventConfig()

    def score(self, bundle):
        self.feature_config.assert_compatible(bundle.config)
        return analyze_stability(bundle, self.stability).change_z

    def to_dict(self):
        return {"schema_version": 2, "detector": self.detector_type, "version": self.version,
                "feature_config": self.feature_config.to_dict(),
                "stability": self.stability.to_dict(), "event": self.events.to_dict()}

    def _settings(self, postprocess):
        cfg = self.events
        if postprocess:
            unknown = set(postprocess) - set(EventConfig().to_dict())
            if unknown:
                raise ValueError(f"不支持的事件参数: {sorted(unknown)}")
            cfg = replace(cfg, **postprocess)
        return cfg

    def predict_events(self, bundle, threshold=None, postprocess=None):
        return self.analyze(bundle, threshold, postprocess)[0]

    def analyze(self, bundle, threshold=None, postprocess=None):
        cfg = self._settings(postprocess)
        if threshold is not None:
            cfg = replace(cfg, trigger_change_z=float(threshold))
        frames = analyze_stability(bundle, self.stability)
        raw, traces = track_events(frames, cfg, bundle.duration_s)
        if cfg.output_mode == "stable":
            # Primary output: the stable segments.  Anomalies are the complement
            # of the whole recording minus those segments.
            segments = track_stable_segments(frames, cfg, bundle.duration_s)
            spans = complement_events(segments, bundle.duration_s, cfg)
            derived = []
            for span in spans:
                i0 = int(np.searchsorted(bundle.times, span["start_s"], side="left"))
                i1 = int(np.searchsorted(bundle.times, span["end_s"], side="right")) - 1
                i0 = int(np.clip(i0, 0, len(bundle.times) - 1))
                i1 = int(np.clip(i1, i0, len(bundle.times) - 1))
                derived.append({**span, "i0": i0, "i1": i1,
                                "boundary_reason": "complement_of_stable",
                                "right_censored": False,
                                "at_edge": span["start_s"] <= 0 or span["end_s"] >= bundle.duration_s,
                                "peak_change_z": float(frames.change_z[i0:i1 + 1].max()),
                                "mean_change_z": float(frames.change_z[i0:i1 + 1].mean()),
                                "peak_exceed": float(frames.exceed[i0:i1 + 1].max()),
                                "peak_slow_ratio": float(frames.slow_ratio[i0:i1 + 1].max()),
                                "peak_time_s": float(bundle.times[i0 + int(np.argmax(frames.change_z[i0:i1 + 1]))]),
                                "triggers": [], "trigger_start_s": span["start_s"],
                                "trigger_end_s": span["end_s"], "stable_confirmed_s": 0.0})
            traces["nonstable_windows"] = raw
            traces["stable_segments"] = segments
            raw = derived
        output = []
        for i, event in enumerate(raw, 1):
            a, b = event["i0"], event["i1"]
            segment = frames.change_z[a:b + 1]
            event.update(duration_s=event["end_s"] - event["start_s"],
                         peak_score=float(segment.max()), mean_score=float(segment.mean()),
                         peak_time_s=float(bundle.times[a + int(np.argmax(segment))]))
            details = event_features(bundle, event, frames.change_z, self.score_type)
            output.append({**details, **event, "event_id": f"{bundle.audio_id}_ev{i:03d}",
                           "score_type": self.score_type,
                           "trigger_change_z": cfg.trigger_change_z,
                           "trigger_exceed": cfg.trigger_exceed})
        stable_frac = np.minimum(frames.fluct_frac, frames.drift_frac)
        traces.update({
            "change_z": frames.change_z,
            "exceed": frames.exceed,
            "slow_ratio": frames.slow_ratio,
            "fluct_frac": frames.fluct_frac,
            "drift_frac": frames.drift_frac,
            "stable": frames.stable,
            "level_dbfs": frames.level_db,
            "sigma_db": frames.sigma_db,
            "state": traces["state"],
            "stable_frac": stable_frac,
            "curves": [
                {"key": "change_z", "label": "变化强度", "unit": "noise",
                 "values": frames.change_z, "threshold": cfg.trigger_change_z,
                 "title": "变化强度（单位：局部噪声 σ，不是 dB 也不是概率）",
                 "note": "虚线为触发阈值；需与超容差比例同时满足，并且连续 3 帧或单帧明显超阈才算触发。"},
                {"key": "exceed", "label": "超容差频带比例", "unit": "ratio",
                 "values": frames.exceed, "threshold": cfg.trigger_exceed,
                 "title": "超过噪声容差的频带比例（0–1）",
                 "note": "虚线为触发比例；宽带稳定噪声不会因此报警，需与变化强度同时满足。"},
                {"key": "stable_frac", "label": "稳定频带比例", "unit": "ratio",
                 "values": stable_frac, "threshold": self.stability.stable_band_frac,
                 "title": "稳定频带比例（0–1，越高越稳定）",
                 "note": "虚线只是稳定判据之一：还需漂移在容差内、且观察窗内无快速变化；"
                         "持续满足确认时长才结束事件。"},
            ],
        })
        return output, traces


def load_detector(config_path=None):
    """Load a detector from JSON, dispatching on ``schema_version``."""
    if config_path is None:
        return StabilityDetector()
    path = Path(config_path)
    if not path.is_file():
        raise ValueError("请传入 L1 JSON 配置文件，不再加载旧训练模型目录")
    obj = json.loads(path.read_text(encoding="utf-8"))
    schema = obj.get("schema_version")
    if schema == 3:
        from .normal import NormalModelDetector
        return NormalModelDetector.from_dict(obj)
    allowed = {"schema_version", "detector", "version", "threshold",
               "feature_config", "postprocess", "stability", "event",
               "fit"}          # "fit" only records how the thresholds were obtained
    if set(obj) - allowed:
        raise ValueError(f"配置包含未知字段: {sorted(set(obj) - allowed)}")
    missing = {"schema_version", "detector", "version", "feature_config"} - set(obj)
    if missing:
        raise ValueError(f"配置缺少字段: {sorted(missing)}")
    if schema == 1:
        if obj.get("detector") != "spectral_change" or obj.get("version") != SpectralChangeDetector.version:
            raise ValueError("schema_version=1 只支持 spectral_event_v2 配置")
        return SpectralChangeDetector(FeatureConfig.from_dict(obj["feature_config"]),
                                      obj["threshold"], obj["postprocess"])
    if schema == 2:
        if obj.get("detector") != "spectral_stability" or obj.get("version") != StabilityDetector.version:
            raise ValueError("schema_version=2 只支持 spectral_stability_v3 配置")
        return StabilityDetector(
            FeatureConfig.from_dict(obj["feature_config"]),
            StabilityConfig.from_dict(obj.get("stability", {})),
            EventConfig.from_dict(obj.get("event", {})),
        )
    raise ValueError(f"不支持的配置 schema_version={schema!r}；"
                     "旧配置按 spectral_event_v2 解释，新配置需要 schema_version=2")
