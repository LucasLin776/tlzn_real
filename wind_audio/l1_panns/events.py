"""Residual-only hysteresis; confirmed normal runs are hard merge barriers."""
import math

import numpy as np


def runs(mask):
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def segment_scores(times, scores, valid, duration, thresholds, cfg, step):
    times, scores, valid = np.asarray(times), np.asarray(scores), np.asarray(valid, bool)
    if times.shape != scores.shape or times.shape != valid.shape:
        raise ValueError("时间轴/评分/有效掩码不一致")
    if np.any(~np.isfinite(scores[valid])):
        raise ValueError("有效区域网络残差不是有限数值")
    low, high, impulse = (thresholds[k] for k in ("low", "high", "impulse"))
    if not all(math.isfinite(v) for v in (low, high, impulse)) or not 0 < low <= high <= impulse:
        raise ValueError("需要正常校准得到的 0 < low <= high <= impulse 阈值")
    normal = np.zeros(len(times), bool)
    for a, b in runs(valid & (scores <= low)):
        if (b - a) * step + 1e-9 >= cfg.confirm_s:
            normal[a:b] = True
    seed = valid & (scores >= impulse)
    for a, b in runs(valid & (scores >= high)):
        if b - a >= cfg.trigger_frames:
            seed[a:b] = True
    # Low residual is never a seed, even when a quantile has a tied value.
    seed &= scores > low
    active = valid & (scores > low)
    spans = runs(active)
    merged = []
    for a, b in spans:
        if (merged and (a - merged[-1][1]) * step <= cfg.merge_gap_s + 1e-9
                and valid[merged[-1][1]:a].all() and not normal[merged[-1][1]:a].any()):
            merged[-1][1] = b
        else:
            merged.append([a, b])
    alarm = np.zeros(len(times), bool)
    for a, b in merged:
        if seed[a:b].any():
            alarm[a:b] = True
    # Values between thresholds without a strong/sustained seed remain unknown,
    # not a hidden spectral alarm and not silently certified normal.
    state = np.full(len(times), -1, np.int8)
    state[normal] = 0
    state[alarm] = 1

    def spans_for(label):
        result = []
        for a, b in runs(state == label):
            start = max(0.0, float(times[a] - step / 2))
            end = min(duration, float(times[b - 1] + step / 2))
            result.append({"start_s": start, "end_s": end, "duration_s": end - start})
        return result

    events = spans_for(1)
    for event, (a, b) in zip(events, runs(alarm)):
        event.update(peak_score=float(scores[a:b].max()), mean_score=float(scores[a:b].mean()),
                     peak_time_s=float(times[a + np.argmax(scores[a:b])]),
                     score_type="panns_temporal_reconstruction_rms", threshold=high,
                     boundary_reason="network_residual", at_edge=False,
                     left_censored=a == 0 or not valid[a - 1],
                     right_censored=b == len(valid) or not valid[b])
    stable = spans_for(0)
    # Complement only of EXPLICIT normal + anomaly; everything else stays unknown.
    known = sorted([*stable, *events], key=lambda x: x["start_s"])
    unknown, previous = [], 0.0
    for item in known:
        if item["start_s"] > previous + 1e-9:
            unknown.append({"start_s": previous, "end_s": item["start_s"], "label": "unknown"})
        previous = item["end_s"]
    if previous < duration:
        unknown.append({"start_s": previous, "end_s": duration, "label": "unknown"})
    return {"events": events, "stable_segments": stable, "normal_regions": stable,
            "unknown_regions": unknown, "states": state}
