"""Frame scores -> contiguous events on the real time axis, plus features.

Thresholding is hysteresis: enter above ``threshold``, leave below
``threshold * low_ratio``.  Intervals separated by <= ``merge_gap`` seconds are
merged; events shorter than ``min_duration`` are dropped.  A short event can
therefore be lost by design -- the settings are recorded in the model metadata
and should be checked against annotated validation recordings.
"""
from __future__ import annotations

import numpy as np

from .features import FeatureBundle


def _as_float(value, default=None):
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def extract_events(
    score: np.ndarray,
    times: np.ndarray,
    threshold: float,
    threshold_low: float | None = None,
    min_duration: float = 0.2,
    merge_gap: float = 0.3,
    duration_s: float | None = None,
    step_s: float | None = None,
) -> list[dict]:
    score = np.asarray(score, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if len(score) != len(times):
        raise ValueError("score 与 times 长度不一致")
    if len(score) == 0:
        return []
    score = np.nan_to_num(score, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    if threshold <= 0:
        raise ValueError("threshold 必须 > 0（分数阈值不能为负）")
    if threshold_low is None:
        threshold_low = threshold * 0.6
    if threshold_low >= threshold:
        raise ValueError("threshold_low 必须小于 threshold")
    if step_s is None:
        step_s = float(np.median(np.diff(times))) if len(times) > 1 else 0.1
    if duration_s is None:
        duration_s = float(times[-1] + step_s / 2)

    intervals: list[list[int]] = []
    active = False
    start = 0
    for i, value in enumerate(score):
        if not active and value >= threshold:
            active = True
            start = i
        elif active and value < threshold_low:
            intervals.append([start, i - 1])
            active = False
    if active:
        intervals.append([start, len(score) - 1])

    merged: list[list[int]] = []
    for a, b in intervals:
        # Compare interval edges, not centres: adjacent cells have zero gap.
        if merged and times[a] - times[merged[-1][1]] - step_s <= merge_gap + 1e-9:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    events: list[dict] = []
    for a, b in merged:
        start_s = max(0.0, float(times[a] - step_s / 2))
        end_s = min(float(duration_s), float(times[b] + step_s / 2))
        if end_s <= start_s:
            continue
        if end_s - start_s < min_duration - 1e-9:
            continue
        seg = score[a : b + 1]
        peak_idx = int(a + np.argmax(seg))
        events.append(
            {
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": end_s - start_s,
                "peak_score": float(np.max(seg)),
                "mean_score": float(np.mean(seg)),
                "peak_time_s": float(times[peak_idx]),
                "i0": int(a),
                "i1": int(b),
                "at_edge": bool(start_s <= step_s / 2 + 1e-9 or end_s >= duration_s - step_s / 2 - 1e-9),
            }
        )
    return events


def event_features(
    bundle: FeatureBundle,
    event: dict,
    score: np.ndarray,
    score_type: str = "score",
) -> dict:
    """Spectral/temporal descriptors + traceable references for L2."""
    a, b = event["i0"], event["i1"]
    spec_db = bundle.spec_db
    n = len(spec_db)
    ft = bundle.mel_times[:n]
    hit = np.flatnonzero((ft >= event["start_s"]) & (ft <= event["end_s"]))
    if len(hit):
        fa, fb = int(hit[0]), int(hit[-1])
    else:
        fa = int(np.clip(np.searchsorted(ft, event["start_s"]), 0, n - 1))
        fb = int(np.clip(np.searchsorted(ft, event["end_s"]), fa, n - 1))
    power = np.mean(10.0 ** (spec_db[fa : fb + 1] / 10.0), axis=0)
    power = np.maximum(power, 1e-12)
    total = float(power.sum())
    freqs = bundle.freqs
    centroid = float((power * freqs).sum() / total)
    flatness = float(np.exp(np.mean(np.log(power))) / np.mean(power))
    peak_freq = float(freqs[int(np.argmax(power))])
    harmonicity_db = float(10.0 * np.log10(power.max() / max(np.median(power), 1e-12)))

    def share(lo: float, hi: float) -> float:
        m = (freqs >= lo) & (freqs < hi)
        return float(power[m].sum() / total) if m.any() else 0.0

    t = bundle.times
    seg = np.asarray(score[a : b + 1], dtype=np.float64)
    k = min(15, len(seg))
    onset_slope = None
    if k >= 3:
        tt = t[a : a + k] - t[a]
        onset_slope = _as_float(np.polyfit(tt, seg[:k], 1)[0])

    return {
        "score_type": score_type,
        "peak_score": event["peak_score"],
        "mean_score": event["mean_score"],
        "peak_time_s": event["peak_time_s"],
        "duration_s": event["duration_s"],
        "at_edge": event["at_edge"],
        "centroid_hz": centroid,
        "flatness": flatness,
        "peak_freq_hz": peak_freq,
        "harmonicity_db": harmonicity_db,
        "share_0_200": share(0, 200),
        "share_200_1k": share(200, 1000),
        "share_1k_4k": share(1000, 4000),
        "share_4k_10k": share(4000, 10000),
        "share_10k_up": share(10000, freqs[-1] + 1),
        "onset_slope_per_s": onset_slope,
    }
