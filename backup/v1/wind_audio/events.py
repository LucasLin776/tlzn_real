"""Frame scores -> contiguous anomaly events, plus event-level features.

Event-level features are exactly what Layer 2 will consume later: they describe
*shape* (onset, duration, harmonicity, band distribution), not just loudness.
"""
from __future__ import annotations

import numpy as np


def extract_events(
    score: np.ndarray,
    frame_dt: float,
    thr_high: float,
    thr_low: float | None = None,
    min_dur: float = 0.2,
    merge_gap: float = 0.3,
) -> list[dict]:
    """Hysteresis thresholding + gap merging + minimum duration."""
    if len(score) == 0:
        return []
    if thr_low is None:
        thr_low = 0.6 * thr_high
    if thr_low >= thr_high:
        raise ValueError("thr_low must be smaller than thr_high")

    intervals: list[list[int]] = []
    active = False
    start = 0
    for i, value in enumerate(score):
        if not active and value >= thr_high:
            active = True
            start = i
        elif active and value < thr_low:
            intervals.append([start, i - 1])
            active = False
    if active:
        intervals.append([start, len(score) - 1])

    gap_frames = int(round(merge_gap / frame_dt))
    merged: list[list[int]] = []
    for a, b in intervals:
        if merged and a - merged[-1][1] <= gap_frames:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    events = []
    n = len(score)
    for a, b in merged:
        duration = (b - a + 1) * frame_dt
        if duration < min_dur:
            continue
        seg = score[a : b + 1]
        events.append(
            {
                "i0": int(a),
                "i1": int(b),
                "start": float(a * frame_dt),
                "end": float((b + 1) * frame_dt),
                "duration": float(duration),
                "peak_z": float(seg.max()),
                "mean_z": float(seg.mean()),
                "at_edge": bool(a == 0 or b == n - 1),
            }
        )
    return events


def event_features(
    power: np.ndarray,
    freqs: np.ndarray,
    score: np.ndarray,
    event: dict,
    frame_dt: float,
) -> dict:
    """Spectral + temporal descriptors for one event."""
    a, b = event["i0"], event["i1"]
    P = np.maximum(power[a : b + 1].mean(axis=0), 1e-12)

    total = float(P.sum())
    centroid = float((P * freqs).sum() / total)
    flatness = float(np.exp(np.mean(np.log(P))) / np.mean(P))
    peak_freq = float(freqs[int(np.argmax(P))])
    harmonicity_db = float(10.0 * np.log10(P.max() / max(np.median(P), 1e-12)))

    def share(lo: float, hi: float) -> float:
        m = (freqs >= lo) & (freqs < hi)
        return float(P[m].sum() / total)

    # onset slope in z per second, over the first 0.15 s of the event
    k = min(15, b - a + 1)
    if k >= 3:
        t = np.arange(k) * frame_dt
        onset_slope = float(np.polyfit(t, score[a : a + k], 1)[0])
    else:
        onset_slope = float("nan")

    return {
        "start": event["start"],
        "end": event["end"],
        "duration": event["duration"],
        "peak_z": event["peak_z"],
        "mean_z": event["mean_z"],
        "centroid_hz": centroid,
        "flatness": flatness,
        "peak_freq_hz": peak_freq,
        "harmonicity_db": harmonicity_db,
        "share_0_200": share(0, 200),
        "share_200_1k": share(200, 1000),
        "share_1k_4k": share(1000, 4000),
        "share_4k_10k": share(4000, 10000),
        "share_10k_22k": share(10000, freqs[-1] + 1),
        "onset_slope_z_per_s": onset_slope,
        "at_edge": bool(event.get("at_edge", False)),
    }
