"""Event tracking after a spectral-change trigger; no fitted model.

Freeze a local pre-event spectrum while a foreground event is active. A drop
in frame-to-frame change alone does NOT end an event. Release on a sustained
return to background, or a stable gain-equivalent / tonal operating state.
"""
from __future__ import annotations

import numpy as np

from .events import extract_events
from .l1 import spectral_change


def _distance(a, b, dynamic_range_db, positive=False):
    energy = np.maximum(a, b)
    floor = np.maximum(energy.max(axis=-1, keepdims=True) * 10 ** (-dynamic_range_db / 10), 1e-12)
    delta = 10 * np.log10(np.maximum(a, floor) / np.maximum(b, floor))
    if positive:
        delta = np.maximum(delta, 0)
    weight = np.sqrt(energy) * (energy >= floor)
    return np.sqrt(np.sum(weight * delta ** 2, axis=-1) / np.maximum(weight.sum(axis=-1), 1e-12))


def _grid(bundle):
    cfg = bundle.config
    half = cfg.n_fft / cfg.sr / 2
    valid = (bundle.mel_times >= half) & (bundle.mel_times <= bundle.duration_s - half)
    if not valid.any():
        return None
    times = bundle.mel_times[valid]
    spectra = 10.0 ** (bundle.spec_db[valid].astype(np.float64) / 10)
    weights = np.full(spectra.shape[1], 2.0)
    weights[0] = 1
    if cfg.n_fft % 2 == 0:
        weights[-1] = 1
    energy = (spectra @ weights) / (cfg.n_fft * np.sum(np.hanning(cfg.n_fft) ** 2))
    keep = (bundle.freqs >= cfg.fmin) & (bundle.freqs <= cfg.fmax)
    if not keep.any():
        keep[np.argmin(np.abs(bundle.freqs - (cfg.fmin + cfg.fmax) / 2))] = True
    band = np.maximum(spectra[:, keep], 1e-12)
    flatness = np.exp(np.mean(np.log(band), axis=1)) / np.maximum(band.mean(axis=1), 1e-12)
    values = np.column_stack([bundle.mel_power[valid], energy, flatness])
    prefix = np.vstack([np.zeros(values.shape[1]), np.cumsum(values, axis=0)])
    # A short averaging window suppresses individual noisy FFT bins.
    half_s = max(cfg.smooth_s / 2, cfg.hop / cfg.sr / 2)
    lo = np.searchsorted(times, bundle.times - half_s)
    hi = np.searchsorted(times, bundle.times + half_s, side="right")
    grid = np.maximum((prefix[hi] - prefix[lo]) / np.maximum((hi - lo)[:, None], 1), 0)
    empty = hi == lo
    nearest = np.clip(np.searchsorted(times, bundle.times), 0, len(times) - 1)
    grid[empty] = values[nearest[empty]]
    return grid[:, :-2], 10 * np.log10(np.maximum(grid[:, -2], 1e-12)), grid[:, -1]


def track_windows(bundle, threshold, pp, change=None):
    cfg, times = bundle.config, bundle.times
    change = spectral_change(bundle) if change is None else np.asarray(change)
    n = len(times)
    traces = {"change_db": change, "activity_db": np.zeros(n), "prominence": np.zeros(n),
              "level_dbfs": np.full(n, -120.0), "level_floor_dbfs": -120.0,
              "rejected_triggers": []}
    grid = _grid(bundle)
    if grid is None:
        return [], traces
    power, level, flatness = grid
    traces["level_dbfs"] = level
    # Robust loudness reference, not a fixed microphone-gain-dependent dBFS cutoff.
    # Median of the top frames totalling 100 ms (not necessarily contiguous).
    # This is relative salience in this recording, not calibrated sound pressure.
    k = min(n, max(1, round(0.1 / cfg.target_step)))
    loud = np.sort(level)[-k:]
    floor_db = float(np.median(loud) - pp["relative_floor_db"])
    traces["level_floor_dbfs"] = floor_db
    seeds = extract_events(change, times, threshold, threshold * pp["low_ratio"],
                           0.0, 0.0, bundle.duration_s, cfg.target_step)
    hold = max(1, int(np.ceil(pp["quiet_s"] / cfg.target_step)))
    settle = max(2, int(np.ceil(pp["settle_s"] / cfg.target_step)))
    # Stability spans a whole dwell interval, including level drift in decaying tails.
    stationary = np.zeros(n, dtype=bool)
    tonal = np.zeros(n, dtype=bool)
    norm = power / np.maximum(power.sum(axis=1, keepdims=True), 1e-12)
    for j in range(settle - 1, n):
        a = j - settle + 1
        shape = _distance(norm[a:j + 1], np.median(norm[a:j + 1], axis=0), cfg.dynamic_range_db)
        stationary[j] = (np.ptp(level[a:j + 1]) <= pp["stability_db"]
                         and np.quantile(shape, 0.9) <= pp["stability_db"])
        tonal[j] = np.quantile(flatness[a:j + 1], 0.9) <= 0.1
    windows = []
    covered_until = -1
    for seed in seeds:
        if seed["i0"] <= covered_until:
            continue
        onset = seed["i0"]
        # Trigger uses lookahead; the pre-trigger interval stays outside that lookahead.
        end_ref = times[onset] - cfg.compare_s
        ref_indices = np.flatnonzero((times >= end_ref - pp["reference_s"]) & (times < end_ref))
        if not len(ref_indices):
            ref_indices = np.arange(max(1, onset))
        reference = np.median(power[ref_indices], axis=0)
        activity = _distance(power, reference, cfg.dynamic_range_db, positive=True)
        prominence = np.abs(power - reference).sum(axis=1) / np.maximum(
            (power + reference).sum(axis=1), 1e-12)
        shape_ref = reference / max(reference.sum(), 1e-12)
        shape_delta = _distance(norm, shape_ref, cfg.dynamic_range_db)
        # Admission rejects weak independent events; release must still follow a
        # genuine event's quiet decay. Do not apply the admission floor to its tail.
        foreground = activity >= pp["release_db"]
        # Grow back through a continuous foreground rise. Small padding is applied later.
        start = onset
        while start > covered_until + 1 and foreground[start - 1]:
            start -= 1
        end, reason = n - 1, "end_of_file"
        quiet = 0
        for j in range(max(onset, seed["i1"]), n):
            quiet = quiet + 1 if not foreground[j] and change[j] < threshold * pp["low_ratio"] else 0
            if quiet >= hold:
                end, reason = max(seed["i1"], j - hold), "background_return"
                break
            a = j - settle + 1
            if a > seed["i1"] and stationary[j]:
                # Stable broadband excess remains foreground. A new horizontal-line
                # state (or same spectrum at a new gain) may end without matching old level.
                same_shape = np.quantile(shape_delta[a:j + 1], 0.9) <= pp["stability_db"] * 2
                if tonal[j] or same_shape:
                    end, reason = a, "new_stable_state"
                    break
        end = max(start, end)
        covered_until = end
        stop = end + 1
        traces["activity_db"][start:stop] = activity[start:stop]
        traces["prominence"][start:stop] = prominence[start:stop]
        # Separate admission from release: significant peak AND spectral-energy change.
        peak_prominence = float(prominence[start:stop].max())
        peak_level = float(level[start:stop].max())
        if peak_prominence < pp["min_prominence"] or peak_level < floor_db:
            traces["rejected_triggers"].append({"start_s": float(times[start] - cfg.target_step / 2),
                "end_s": min(bundle.duration_s, float(times[end] + cfg.target_step / 2)),
                "reason": "below_prominence" if peak_prominence < pp["min_prominence"] else "below_level_floor",
                "peak_prominence": peak_prominence, "peak_level_dbfs": peak_level})
            continue
        start_s = max(0.0, float(times[start] - cfg.target_step / 2))
        end_s = min(bundle.duration_s, float(times[end] + cfg.target_step / 2))
        if reason == "end_of_file":
            end_s = bundle.duration_s
        window = {"start_s": start_s, "end_s": end_s, "i0": start, "i1": end,
                  "boundary_reason": reason, "right_censored": reason == "end_of_file",
                  "peak_activity_db": float(activity[start:stop].max()),
                  "peak_prominence": peak_prominence, "peak_level_dbfs": peak_level,
                  "level_floor_dbfs": floor_db}
        hits = [s for s in seeds if s["i0"] <= end and s["i1"] >= start]
        window["triggers"] = [{"start_s": s["start_s"], "end_s": s["end_s"]} for s in hits]
        window["trigger_start_s"] = hits[0]["start_s"]
        window["trigger_end_s"] = hits[-1]["end_s"]
        windows.append(window)
    # Merge short quiet gaps, but never cut an active event at a fixed maximum duration.
    merged = []
    for window in windows:
        if merged and window["start_s"] - merged[-1]["end_s"] <= pp["merge_gap"] + 1e-9:
            merge_window(merged[-1], window)
        else:
            merged.append(window)
    return [w for w in merged if w["end_s"] - w["start_s"] >= pp["min_duration"] - 1e-9], traces


def merge_window(left, right):
    if right["end_s"] >= left["end_s"]:
        left.update(end_s=right["end_s"], i1=right["i1"],
                    boundary_reason=right["boundary_reason"], right_censored=right["right_censored"])
    left["trigger_end_s"] = max(left["trigger_end_s"], right["trigger_end_s"])
    left["triggers"].extend(right["triggers"])
    for key in ("peak_activity_db", "peak_prominence", "peak_level_dbfs"):
        left[key] = max(left[key], right[key])
