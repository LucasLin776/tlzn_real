"""Normal-only learning of temporal spectral variability (offline detector).

This is a fitted one-class statistical envelope, not a timbre classifier.
Absolute pitch, spectrum and level are deliberately not model coordinates:
a new steady operating state must become normal again. Parameters are frozen
at inference, so persistent anomalies cannot enlarge their own noise tolerance.

Scoring has two modes:

* ``max``   -- the legacy ``max_j(values_j / scale_j)``: any single dimension
  above its training tail fires.
* ``joint`` -- one-sided whitened distance ``sqrt(u' R^-1 u)`` with
  ``u = max(z - median, 0)``, rescaled so its training median matches the
  legacy score.  It amplifies unusual *combinations* of the four features.
  Measured caveat: a device that is uniformly more active, and a broadband
  anomaly, both move along the dominant correlation axis, so whitening
  discounts both; simulated cross-device false alarms improved only marginally
  while broadband misses got worse.  It is therefore opt-in, not the default.

Post-processing additionally enforces a minimum anomaly duration and a minimum
normal gap (``min_anomaly_s`` / ``min_normal_s``) so that a few frames of
boundary jitter cannot fragment the delivered regions.  A separate impulse
channel reports short high-contrast peaks for manual review; it never changes
``events``, the normal regions, or the duration metrics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from .events import event_features, extract_events
from .features import FeatureConfig

FEATURE_NAMES = ("fast_change_db", "short_fluct_db", "drift_db", "long_fluct_db")
# Physical lower bounds prevent almost-silent training from amplifying tiny noise.
SCALE_FLOORS = np.array([0.75, 0.60, 0.75, 0.75])
SCORE_MODES = ("joint", "max")
CORRELATION_SHRINKAGE = 0.02


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
    # Segment-level constraints: a short fragment is a boundary artefact, not a
    # second event, and a short normal island must not split one disturbance.
    min_anomaly_s: float = 0.30
    min_normal_s: float = 0.50
    # Impulse channel: presence hints only, excluded from regions and metrics.
    short_channel: bool = True
    short_peak_ratio: float = 2.0
    short_window_s: float = 0.10

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
        if self.min_anomaly_s < 0 or self.min_normal_s < 0 or self.short_window_s <= 0:
            raise ValueError("最短异常段/最短正常间隙/短事件窗口不能为负")
        if self.short_peak_ratio <= 1:
            raise ValueError("短事件通道的峰值倍数必须大于 1")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if set(data) - set(asdict(cls())):
            raise ValueError("正常模型包含未知参数")
        return cls(**data)


@dataclass(frozen=True)
class ScoreConfig:
    """How per-frame feature values become one deviation score."""

    mode: str = "max"
    center: tuple = (0.0, 0.0, 0.0, 0.0)
    precision: tuple = ()
    normalizer: float = 1.0

    def __post_init__(self):
        if self.mode not in SCORE_MODES:
            raise ValueError(f"打分方式只支持 {' / '.join(SCORE_MODES)}")
        if self.mode == "joint":
            if np.shape(self.precision) != (len(FEATURE_NAMES), len(FEATURE_NAMES)):
                raise ValueError("联合打分需要 4x4 精度矩阵")
            if not np.isfinite(np.asarray(self.precision)).all():
                raise ValueError("精度矩阵必须为有限数值")
            if not np.isfinite(self.normalizer) or self.normalizer <= 0:
                raise ValueError("联合打分归一化系数必须为正")

    def to_dict(self):
        return {"mode": self.mode, "center": list(self.center),
                "precision": [list(row) for row in self.precision],
                "normalizer": float(self.normalizer)}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("score 字段必须是对象")
        allowed = {"mode", "center", "precision", "normalizer"}
        if set(data) - allowed:
            raise ValueError("score 字段包含未知键")
        return cls(mode=data.get("mode", "max"),
                   center=tuple(data.get("center", (0.0,) * len(FEATURE_NAMES))),
                   precision=tuple(tuple(row) for row in data.get("precision", ())),
                   normalizer=float(data.get("normalizer", 1.0)))


def frame_score(values, scales, score: ScoreConfig):
    """Turn raw dB features into the deviation score used by every threshold."""
    z = np.asarray(values, dtype=np.float64) / np.asarray(scales, dtype=np.float64)
    if score.mode == "max":
        return z.max(axis=1)
    center = np.asarray(score.center, dtype=np.float64)
    precision = np.asarray(score.precision, dtype=np.float64)
    u = np.maximum(z - center, 0.0)
    quadratic = np.einsum("ij,ij->i", u @ precision, u)
    return np.sqrt(np.maximum(quadratic, 0.0)) * score.normalizer


def _fit_score(train, scales, mode):
    """Fit the joint score on training frames (centres, correlation, rescale).

    Degenerate training data (a dimension without any variation) carries no
    correlation information, so the joint form falls back to ``max`` instead of
    dividing by an almost zero median.
    """
    if mode == "max":
        return ScoreConfig(mode="max")
    z = np.asarray(train, dtype=np.float64) / np.asarray(scales, dtype=np.float64)
    if np.any(np.std(z, axis=0) < 1e-9):
        return ScoreConfig(mode="max")
    center = np.median(z, axis=0)
    correlation = np.corrcoef(z, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(FEATURE_NAMES)
    correlation = (1 - CORRELATION_SHRINKAGE) * correlation + CORRELATION_SHRINKAGE * np.eye(n)
    precision = np.linalg.pinv(correlation)
    unit = ScoreConfig("joint", tuple(center), tuple(map(tuple, precision)), 1.0)
    raw = frame_score(train, scales, unit)
    legacy = z.max(axis=1)
    median_raw = float(np.median(raw))
    if not np.isfinite(median_raw) or median_raw < 1e-6:
        return ScoreConfig(mode="max")
    normalizer = float(np.clip(np.median(legacy) / median_raw, 1e-3, 1e3))
    return ScoreConfig("joint", tuple(center), tuple(map(tuple, precision)), normalizer)


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

    def __init__(self, scales, threshold, feature_config=None, normal=None, training=None,
                 score=None):
        self.feature_config = feature_config or FeatureConfig()
        self.normal = normal or NormalConfig()
        self.scales = np.asarray(scales, dtype=np.float64).copy()
        self.threshold = float(threshold)
        self.training = dict(training or {})
        self.score_config = score or ScoreConfig(mode="max")
        if self.scales.shape != (len(FEATURE_NAMES),) or not np.isfinite(self.scales).all() or (self.scales <= 0).any():
            raise ValueError("正常模型尺度必须是四个有限正数")
        if not np.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("正常模型阈值必须是有限正数")

    @classmethod
    def fit(cls, train_values, calibration_values, *, feature_config=None, normal=None, alpha=0.01,
            score_mode="max"):
        if not np.isfinite(alpha) or not 0 < alpha < 0.5:
            raise ValueError("alpha 需在 (0, 0.5) 内")
        if score_mode not in SCORE_MODES:
            raise ValueError(f"打分方式只支持 {' / '.join(SCORE_MODES)}")
        train, calibration = _samples(train_values), _samples(calibration_values)
        scales = np.maximum(np.quantile(train, 0.99, axis=0), SCALE_FLOORS)
        score = _fit_score(train, scales, score_mode)
        scores = frame_score(calibration, scales, score)
        # Only upper deviations are abnormal: quieter/steadier than training is normal.
        threshold = max(1.0, float(np.quantile(scores, 1 - alpha, method="higher")))
        detector = cls(scales, threshold, feature_config, normal,
                       {"method": "normal_only_upper_envelope", "alpha": alpha,
                        "train_frames": len(train), "calibration_frames": len(calibration),
                        "calibration_frame_exceedance": float(np.mean(scores > threshold)),
                        "score_mode": score_mode,
                        "note": "alpha 是正常校准帧的经验尾部比例，不是事件误报率保证"}, score)
        return detector

    def with_threshold(self, threshold):
        """Same model, different decision threshold (event-level calibration)."""
        threshold = float(threshold)
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("阈值必须为有限正数")
        return NormalModelDetector(self.scales, threshold, self.feature_config, self.normal,
                                   self.training, self.score_config)

    def to_dict(self):
        return {"schema_version": 3, "detector": self.detector_type, "version": self.version,
                "feature_config": self.feature_config.to_dict(), "normal": self.normal.to_dict(),
                "feature_names": list(FEATURE_NAMES), "scales": self.scales.tolist(),
                "threshold": self.threshold, "training": self.training,
                "score": self.score_config.to_dict()}

    @classmethod
    def from_dict(cls, obj):
        required = {"schema_version", "detector", "version", "feature_config", "normal",
                    "feature_names", "scales", "threshold", "training"}
        if set(obj) - required - {"score"} or not required <= set(obj):
            raise ValueError("无效正常模型字段或版本")
        if obj["schema_version"] != 3 or obj["detector"] != cls.detector_type:
            raise ValueError("无效正常模型字段或版本")
        if obj["version"] != cls.version or obj["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("正常模型特征/版本不匹配，请重新训练")
        return cls(obj["scales"], obj["threshold"], FeatureConfig.from_dict(obj["feature_config"]),
                   NormalConfig.from_dict(obj["normal"]), obj["training"],
                   ScoreConfig.from_dict(obj["score"]) if "score" in obj else None)

    def score(self, bundle):
        self.feature_config.assert_compatible(bundle.config)
        return frame_score(normal_features(bundle, self.normal), self.scales, self.score_config)

    def predict_events(self, bundle, threshold=None, postprocess=None):
        """Compatibility API for anomaly consumers (including L2)."""
        return self.analyze(bundle, threshold, postprocess)[0]

    def predict_normal_regions(self, bundle, threshold=None, postprocess=None):
        """Primary normal-region output, without any labels as input."""
        return self.analyze(bundle, threshold, postprocess)[1]["normal_regions"]

    def predict_short_events(self, bundle, threshold=None, postprocess=None):
        """Impulse hints only; they never modify the normal regions."""
        return self.analyze(bundle, threshold, postprocess)[1]["short_events"]

    def analyze(self, bundle, threshold=None, postprocess=None):
        self.feature_config.assert_compatible(bundle.config)
        allowed = {"enter_s", "release_s", "min_duration", "merge_gap", "padding", "low_ratio",
                   "strong_ratio", "min_anomaly_s", "min_normal_s", "short_channel",
                   "short_peak_ratio", "short_window_s"}
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
        score = frame_score(values, self.scales, self.score_config)
        events, traces = normal_windows(score, bundle.times, bundle.duration_s, threshold, cfg,
                                         bundle.config.target_step)
        short = short_events(score, bundle.times, bundle.duration_s, threshold, cfg,
                             bundle.config.target_step, covered=events)
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
                      short_events=short, stable_segments=normal_regions,
                      normal_regions=normal_regions, primary_output="normal_regions",
                      output_mode="normal_model", score_mode=self.score_config.mode,
                      anomaly_source="learned_normal_deviation",
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
    After merging, fragment control removes pieces shorter than
    ``min_anomaly_s`` (unless strong) and bridges normal gaps shorter than
    ``min_normal_s``, so boundary jitter cannot split one disturbance in two.
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
    # Fragment control: drop short leftovers, then bridge short normal gaps.
    kept, dropped = [], []
    for event in merged:
        short = event["end_s"] - event["start_s"] < cfg.min_anomaly_s - 1e-9
        peak = float(score[event["i0"]:event["i1"] + 1].max())
        if short and peak < threshold * cfg.strong_ratio:
            dropped.append({"start_s": event["start_s"], "end_s": event["end_s"],
                            "reason": "shorter_than_min_anomaly", "peak_score": peak})
            continue
        kept.append(event)
    bridged = []
    for event in kept:
        if bridged and event["start_s"] - bridged[-1]["end_s"] < cfg.min_normal_s - 1e-9:
            bridged[-1].update(end_s=event["end_s"], i1=event["i1"])
        else:
            bridged.append(event)
    output = []
    for event in bridged:
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
    return output, {"rejected_triggers": rejected + dropped}


def short_events(score, times, duration, threshold, cfg, step, covered=()):
    """Impulse channel: short, high-contrast peaks for review only.

    The main detector needs a sustained or very strong excursion; genuine
    sub-second impacts often fall just below that.  This channel reports them
    as presence hints, deliberately excluded from ``events``, the normal
    regions and every duration metric.
    """
    if not cfg.short_channel:
        return []
    level = threshold * cfg.short_peak_ratio
    above = np.flatnonzero(score >= level)
    if not len(above):
        return []
    width = max(1, round(cfg.short_window_s / step))
    groups = np.split(above, np.flatnonzero(np.diff(above) > width) + 1)
    spans = [(e["start_s"], e["end_s"]) for e in covered]
    out = []
    for group in groups:
        peak = int(group[np.argmax(score[group])])
        a = max(0.0, float(times[peak]) - cfg.short_window_s / 2)
        b = min(duration, float(times[peak]) + cfg.short_window_s / 2)
        if any(a < end and b > start for start, end in spans):
            continue
        out.append({"start_s": a, "end_s": b, "duration_s": b - a,
                    "peak_score": float(score[peak]), "peak_time_s": float(times[peak]),
                    "channel": "short_impulse", "threshold": level})
    return out


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


def event_level_threshold(detector, calibration, target_fp, *, normal_cfg=None, floor=None):
    """Pick the smallest threshold whose calibration normal-time alarm <= target.

    ``calibration`` is a list of dicts with ``times``, ``values``, ``duration``
    and ``normal`` (explicit normal spans).  Frame quantiles do not transfer to
    decision-level false alarms (hysteresis, backdating and merging multiply
    them), so the operating point is searched on the metric that matters.
    """
    if not 0 < target_fp < 1:
        raise ValueError("目标误报率需在 (0,1) 内")
    cfg = normal_cfg or detector.normal
    step = detector.feature_config.target_step
    items = []
    for entry in calibration:
        score = frame_score(entry["values"], detector.scales, detector.score_config)
        items.append((entry["times"], score, float(entry["duration"]), entry["normal"]))
    pooled = np.concatenate([s for _, s, _, _ in items])
    candidates = np.unique(np.maximum(1.0, np.quantile(pooled, np.linspace(0.50, 0.9995, 80),
                                                      method="higher")))
    # ``floor`` keeps the frame-quantile rule as a sensitivity cap: the event
    # budget may tighten the threshold but never loosen it below the alpha rule.
    if floor is not None:
        candidates = candidates[candidates >= float(floor) - 1e-12]

    def alarm_fraction(threshold):
        normal_s = alarmed_s = 0.0
        for times, score, duration, spans in items:
            events, _ = normal_windows(score, times, duration, threshold, cfg, step)
            regions = normal_complement(events, duration)
            for a, b in spans:
                normal_s += b - a
                alarmed_s += (b - a) - sum(max(0.0, min(b, r["end_s"]) - max(a, r["start_s"]))
                                           for r in regions)
        return alarmed_s / normal_s if normal_s > 0 else 0.0

    if not len(candidates):
        return float(floor), {"target_fp": target_fp, "achieved_calibration_fp": None,
                              "note": "没有高于灵敏度上限的候选阈值"}
    measured = [(float(t), alarm_fraction(float(t))) for t in candidates]
    # Smallest threshold that still meets the budget, and that no larger
    # candidate breaks (guards against a noisy non-monotone measurement).
    chosen, achieved = float(candidates[-1]), measured[-1][1]
    for i, (threshold, fraction) in enumerate(measured):
        if all(f <= target_fp + 1e-12 for _, f in measured[i:]):
            chosen, achieved = threshold, fraction
            break
    return chosen, {"target_fp": target_fp, "achieved_calibration_fp": achieved,
                    "feasible": achieved is not None and achieved <= target_fp + 1e-12,
                    "candidates": [[t, f] for t, f in measured]}
