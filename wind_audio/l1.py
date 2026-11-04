"""Training-free spectral fluctuation score, measured in weighted dB change."""
from __future__ import annotations

import numpy as np

from .audio import median_filter
from .features import FeatureBundle


def spectral_change(bundle: FeatureBundle):
    """Compare adjacent local mean-power spectra; never fit a global reference.

    At time t compare [t-window,t) and [t,t+window). Only full real STFT
    frames are used when available, avoiding reflection-padding edge alarms.
    The common local floor and weights suppress numerically unstable weak bins.
    Constant gain cancels in the local ratios; a gain transition remains a candidate.
    """
    cfg = bundle.config
    half_fft = cfg.n_fft / cfg.sr / 2
    valid = (bundle.mel_times >= half_fft) & (bundle.mel_times <= bundle.duration_s - half_fft)
    if not valid.any():
        # No complete FFT window: there is insufficient temporal evidence.
        return np.zeros(len(bundle.times), dtype=np.float32)
    times = bundle.mel_times[valid]
    power = bundle.mel_power[valid].astype(np.float64)
    prefix = np.vstack([np.zeros(power.shape[1]), np.cumsum(power, axis=0)])
    t = bundle.times
    lo = np.searchsorted(times, t - cfg.compare_s, side="left")
    mid = np.searchsorted(times, t, side="left")
    hi = np.searchsorted(times, t + cfg.compare_s, side="left")
    left_count, right_count = mid - lo, hi - mid
    left = (prefix[mid] - prefix[lo]) / np.maximum(left_count[:, None], 1)
    right = (prefix[hi] - prefix[mid]) / np.maximum(right_count[:, None], 1)
    energy = np.maximum(left, right)
    floor = np.maximum(energy.max(axis=1, keepdims=True) * 10 ** (-cfg.dynamic_range_db / 10), 1e-12)
    delta = 10 * np.log10(np.maximum(right, floor) / np.maximum(left, floor))
    weights = np.sqrt(energy) * (energy >= floor)
    score = np.sqrt(np.sum(weights * delta ** 2, axis=1) /
                    np.maximum(weights.sum(axis=1), 1e-12))
    score[(left_count == 0) | (right_count == 0)] = 0
    kernel = max(1, round(cfg.smooth_s / cfg.target_step))
    return median_filter(score, kernel).astype(np.float32)
