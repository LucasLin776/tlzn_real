"""Normal-only learning of temporal spectral variability (offline detector).

This is a fitted one-class statistical envelope, not a timbre classifier.
Absolute pitch, spectrum and level are deliberately not model coordinates:
a new steady operating state must become normal again. Parameters are frozen
at inference, so persistent anomalies cannot enlarge their own noise tolerance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from .events import event_features, extract_events
from .features import FeatureConfig

FEATURE_NAMES = ("fast_change_db", "short_fluct_db", "drift_db", "long_fluct_db")
# Physical lower bounds prevent almost-silent training from amplifying tiny noise.
SCALE_FLOORS = np.array([0.75, 0.60, 0.75, 0.75])


@dataclass(frozen=True)
class NormalConfig:
    smooth_s: float = 0.06
    fast_s: float = 0.10
    short_s: float = 0.60
    drift_s: float = 0.40
    long_s: float = 1.20
    active_range_db: float = 30.0
    enter_s: float = 0.16
    release_s: float = 0.60
    min_duration: float = 0.20
    merge_gap: float = 0.35
    low_ratio: float = 0.80
    strong_ratio: float = 2.50
    padding: float = 0.05

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("正常模型参数必须为有限数值")
        if not 0 < self.smooth_s <= self.fast_s <= self.short_s <= self.long_s <= 10:
            raise ValueError("要求 0 < smooth_s <= fast_s <= short_s <= long_s <= 10")
        if not 0 < self.drift_s <= self.long_s or not 20 <= self.active_range_db <= 80:
            raise ValueError("无效漂移窗口或有效频带范围")
        if any(getattr(self, k) <= 0 for k in ("enter_s", "release_s", "min_duration")):
            raise ValueError("进入/退出/最短持续时间必须为正")
        if self.merge_gap < 0 or self.padding < 0 or not 0 < self.low_ratio < 1 or self.strong_ratio <= 1:
            raise ValueError("无效正常模型事件参数")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if set(data) - set(asdict(cls())):
            raise ValueError("正常模型包含未知参数")
        return cls(**data)


def context_margin(feature_config, cfg):
    """Conservative full dependency radius; used to exclude label boundaries."""
    return (max(cfg.long_s / 2, cfg.drift_s, cfg.fast_s) + cfg.smooth_s / 2
            + feature_config.n_fft / feature_config.sr / 2
            + feature_config.hop / feature_config.sr + feature_config.target_step)


def _mean(x, width):
    width = max(1, int(width))
    left = width // 2
    padded = np.pad(x, ((left, width - left - 1), (0, 0)), mode="edge")
    prefix = np.vstack([np.zeros(x.shape[1]), np.cumsum(padded, axis=0)])
    return (prefix[width:] - prefix[:-width]) / width


def normal_features(bundle, cfg):
    """Four dB-valued variability features, independent of absolute spectrum.

    Centred windows are intentional: this pipeline processes complete files,
    not a zero-lookahead live stream. Boundary padding never enters training.
    """
    step = bundle.config.target_step
    idx = np.clip(np.searchsorted(bundle.mel_times, bundle.times), 0, len(bundle.mel_times) - 1)
    # A centred STFT pads at file edges. Use the nearest complete FFT frame
    # there, so padding artefacts cannot masquerade as a sudden sound change.
    half_fft = bundle.config.n_fft / bundle.config.sr / 2
    complete = np.flatnonzero((bundle.mel_times >= half_fft)
                             & (bundle.mel_times <= bundle.duration_s - half_fft))
    if not len(complete):
        raise ValueError("音频太短，缺少完整频谱帧")
    idx = np.clip(idx, complete[0], complete[-1])
    power = np.maximum(bundle.mel_power[idx].astype(np.float64), 1e-12)
    power = _mean(power, round(cfg.smooth_s / step))
    level = 10 * np.log10(power)
    level = np.maximum(level, level.max(axis=1, keepdims=True) - cfg.active_range_db)
    weights = np.sqrt(power)
    weights[power < power.max(axis=1, keepdims=True) * 10 ** (-cfg.active_range_db / 10)] = 0
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)

    def rms(values):
        return np.sqrt(np.maximum((weights * values ** 2).sum(axis=1), 0))

    def spread(seconds):
        width = max(1, round(seconds / step))
        # Subtract a constant first to avoid cancellation near digital silence.
        centred = level - level[0]
        variance = np.maximum(_mean(centred ** 2, width) - _mean(centred, width) ** 2, 0)
        return np.sqrt((weights * variance).sum(axis=1))

    fast = max(1, round(cfg.fast_s / step))
    previous = level[np.maximum(np.arange(len(level)) - fast, 0)]
    drift = max(1, round(cfg.drift_s / step))
    mean = _mean(level, drift)
    before = mean[np.maximum(np.arange(len(mean)) - drift // 2, 0)]
    after = mean[np.minimum(np.arange(len(mean)) + (drift + 1) // 2, len(mean) - 1)]
    return np.column_stack([rms(level - previous), spread(cfg.short_s),
                            rms(after - before), spread(cfg.long_s)])


def _samples(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES) or len(values) < 100:
        raise ValueError("正常片段不足：至少需要 100 个有效帧，且每帧含四个特征")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("正常特征必须是有限非负数")
    return values


class NormalModelDetector:
    detector_type = "normal_envelope"
    score_type = "normal_deviation_ratio"
    version = "normal_envelope_v1"
    requires_training = True

    def __init__(self, scales, threshold, feature_config=None, normal=None, training=None):
        self.feature_config = feature_config or FeatureConfig()
        self.normal = normal or NormalConfig()
        self.scales = np.asarray(scales, dtype=np.float64).copy()
        self.threshold = float(threshold)
        self.training = dict(training or {})
        if self.scales.shape != (len(FEATURE_NAMES),) or not np.isfinite(self.scales).all() or (self.scales <= 0).any():
            raise ValueError("正常模型尺度必须是四个有限正数")
        if not np.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("正常模型阈值必须是有限正数")

    @classmethod
    def fit(cls, train_values, calibration_values, *, feature_config=None, normal=None, alpha=0.01):
        if not np.isfinite(alpha) or not 0 < alpha < 0.5:
            raise ValueError("alpha 需在 (0, 0.5) 内")
        train, calibration = _samples(train_values), _samples(calibration_values)
        scales = np.maximum(np.quantile(train, 0.99, axis=0), SCALE_FLOORS)
        scores = (calibration / scales).max(axis=1)
        # Only upper deviations are abnormal: quieter/steadier than training is normal.
        threshold = max(1.0, float(np.quantile(scores, 1 - alpha, method="higher")))
        return cls(scales, threshold, feature_config, normal,
                   {"method": "normal_only_upper_envelope", "alpha": alpha,
                    "train_frames": len(train), "calibration_frames": len(calibration),
                    "calibration_frame_exceedance": float(np.mean(scores > threshold)),
                    "note": "alpha 是正常校准帧的经验尾部比例，不是事件误报率保证"})

    def to_dict(self):
        return {"schema_version": 3, "detector": self.detector_type, "version": self.version,
                "feature_config": self.feature_config.to_dict(), "normal": self.normal.to_dict(),
                "feature_names": list(FEATURE_NAMES), "scales": self.scales.tolist(),
                "threshold": self.threshold, "training": self.training}

    @classmethod
    def from_dict(cls, obj):
        required = {"schema_version", "detector", "version", "feature_config", "normal",
                    "feature_names", "scales", "threshold", "training"}
        if set(obj) != required or obj["schema_version"] != 3 or obj["detector"] != cls.detector_type:
            raise ValueError("无效正常模型字段或版本")
        if obj["version"] != cls.version or obj["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("正常模型特征/版本不匹配，请重新训练")
        return cls(obj["scales"], obj["threshold"], FeatureConfig.from_dict(obj["feature_config"]),
                   NormalConfig.from_dict(obj["normal"]), obj["training"])

    def score(self, bundle):
        self.feature_config.assert_compatible(bundle.config)
        return (normal_features(bundle, self.normal) / self.scales).max(axis=1)

    def predict_events(self, bundle, threshold=None, postprocess=None):
        """Compatibility API for anomaly consumers (including L2)."""
        return self.analyze(bundle, threshold, postprocess)[0]

    def predict_normal_regions(self, bundle, threshold=None, postprocess=None):
        """Primary normal-region output, without any labels as input."""
        return self.analyze(bundle, threshold, postprocess)[1]["normal_regions"]

    def analyze(self, bundle, threshold=None, postprocess=None):
        self.feature_config.assert_compatible(bundle.config)
        allowed = {"enter_s", "release_s", "min_duration", "merge_gap", "padding", "low_ratio", "strong_ratio"}
        if set(postprocess or {}) - allowed:
            raise ValueError("正常模型只允许覆盖事件过滤参数，特征窗口改变后须重新训练")
        cfg = replace(self.normal, **(postprocess or {}))
        threshold = self.threshold if threshold is None else float(threshold)
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("正常偏离阈值必须为有限正数")
        if bundle.duration_s < 2 * context_margin(self.feature_config, cfg):
            raise ValueError("音频太短，正常模型无法获得完整观察窗（至少约 1.4 秒）")
        values = normal_features(bundle, cfg)
        components = values / self.scales
        score = components.max(axis=1)
        events, traces = normal_windows(score, bundle.times, bundle.duration_s, threshold, cfg,
                                         bundle.config.target_step)
        output = []
        for i, event in enumerate(events, 1):
            details = event_features(bundle, event, score, self.score_type)
            output.append({**details, **event, "event_id": f"{bundle.audio_id}_ev{i:03d}",
                           "score_type": self.score_type, "threshold": threshold})
        active = np.zeros(len(score), bool)
        for event in output:
            active |= (bundle.times >= event["start_s"]) & (bundle.times < event["end_s"])
        normal_regions = normal_complement(output, bundle.duration_s)
        traces.update(active=active, stable=~active, normal_score=score, components=components,
                      stable_segments=normal_regions, normal_regions=normal_regions,
                      primary_output="normal_regions",
                      output_mode="normal_model", anomaly_source="learned_normal_deviation",
                      curves=[{"key": "normal_deviation", "values": score, "threshold": threshold,
                               "label": "偏离正常的程度", "unit": "deviation_ratio",
                               "title": "相对正常样本的偏离（不是概率）",
                               "note": "超过虚线后还需通过持续时间/强度过滤；阈值只来自正常样本。"}])
        return output, traces


def normal_windows(score, times, duration, threshold, cfg, step):
    """Hysteresis with sustained entry, backdated release, then burst merging.

    Reject brief weak candidates BEFORE merging: repeated irrelevant clicks
    must not combine into a sufficiently long false alarm. Strong short events
    remain eligible; noise rejection is not simply dropping all short sounds.
    """
    raw = extract_events(score, times, threshold, threshold * cfg.low_ratio,
                         0, 0, duration, step)
    accepted, rejected = [], []
    for event in raw:
        segment = score[event["i0"]:event["i1"] + 1]
        runs = np.diff(np.r_[False, segment >= threshold, False].astype(int))
        lengths = np.flatnonzero(runs == -1) - np.flatnonzero(runs == 1)
        sustained = len(lengths) and lengths.max() * step >= cfg.enter_s - 1e-9
        strong = event["peak_score"] >= threshold * cfg.strong_ratio
        if not strong and (not sustained or event["duration_s"] < cfg.min_duration - 1e-9):
            rejected.append({"start_s": event["start_s"], "end_s": event["end_s"],
                             "reason": "brief_weak_deviation", "peak_score": event["peak_score"]})
            continue
        accepted.append(dict(event))
    merged = []
    for event in accepted:
        if merged and event["start_s"] - merged[-1]["end_s"] <= max(cfg.release_s, cfg.merge_gap) + 1e-9:
            merged[-1].update(end_s=event["end_s"], i1=event["i1"])
        else:
            merged.append(event)
    output = []
    for event in merged:
        event["start_s"] = max(0, event["start_s"] - cfg.padding)
        event["end_s"] = min(duration, event["end_s"] + cfg.padding)
        if output and event["start_s"] <= output[-1]["end_s"]:
            output[-1].update(end_s=event["end_s"], i1=event["i1"])
        else:
            output.append(event)
    for event in output:
        a, b = event["i0"], event["i1"]
        segment = score[a:b + 1]
        censored = event["i1"] == len(score) - 1
        event.update(duration_s=event["end_s"] - event["start_s"], peak_score=float(segment.max()),
                     mean_score=float(segment.mean()), peak_time_s=float(times[a + np.argmax(segment)]),
                     boundary_reason="end_of_file" if censored else "return_to_learned_normal",
                     right_censored=censored,
                     at_edge=event["start_s"] == 0 or event["end_s"] >= duration)
    return output, {"rejected_triggers": rejected}


def normal_complement(events, duration):
    spans, cursor = [], 0.0
    for event in events:
        if event["start_s"] > cursor:
            spans.append((cursor, event["start_s"]))
        cursor = max(cursor, event["end_s"])
    if cursor < duration:
        spans.append((cursor, duration))
    return [{"start_s": a, "end_s": b, "duration_s": b - a, "coverage": (b - a) / duration}
            for a, b in spans]
