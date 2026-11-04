"""Multi-timescale spectral stability analysis (no fitted model).

The business premise is that a sound whose spectrum stays stable over a long
enough observation window is normal -- whatever its shape, band position or
loudness.  An anomaly is therefore a *clear* time-frequency change that exceeds
the local noise tolerance, not a particular timbre.

Every statistic is causal (it only looks backwards) and is computed on the
20 ms output grid of ``FeatureBundle`` with robust statistics of the recording
itself.  Nothing is compared with a global loudness reference.

==================  ==========  =========================================
quantity            unit        meaning
==================  ==========  =========================================
``L``               dB          log-mel power per band
``sigma``           dB          local per-band noise scale: robust MAD of
                                the fast difference, clamped to a sane range
``change_z``        noise sigma energy-weighted RMS of |dL|/sigma
``exceed``          0..1        power-weighted share of bands with
                                |dL|/sigma > ``band_z``
``slow_ratio``      ratio       consecutive 1 s block change / tolerance
``fluct_frac``      0..1        power-weighted share of bands whose
                                short-term fluctuation is inside tolerance
``drift_frac``      0..1        same for the local mean drift
``stable``          bool        fluct/drift inside tolerance *and* no fast
                                change above ``stable_change_z`` in the window
==================  ==========  =========================================

Bands more than ``active_range_db`` below the loudest band are ignored: they
carry no audible content and their dB value is dominated by the numerical
floor, which would otherwise create permanent instability in quiet files.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np

from .features import FeatureBundle


@dataclass(frozen=True)
class StabilityConfig:
    """Observation scales (seconds) and noise tolerances (noise sigmas / dB)."""

    fast_s: float = 0.10            # short scale: transient / step detection
    smooth_s: float = 0.03          # pre-smoothing before differencing
    noise_s: float = 2.00           # window for the local noise scale
    stable_s: float = 0.60          # window used to judge "is it stable now"
    drift_s: float = 0.40           # half window of the local mean-drift test
    slow_s: float = 1.00            # block length of the slow-ramp test
    slow_cal_s: float = 20.0        # window that calibrates the slow-ramp tolerance
    slow_scale: float = 2.0         # allowed 1 s block change, x its own variability
    slow_floor_db: float = 0.60
    slow_cap_db: float = 8.0
    bg_s: float = 2.00              # background median span
    bg_lag_s: float = 0.20
    excess_cal_s: float = 5.00      # window that calibrates the excess tolerance
    excess_scale: float = 3.0       # allowed deviation, x its own slow variability
    excess_floor_db: float = 0.80   # below this a level deviation is not audible
    excess_cap_db: float = 12.0     # never let a long event inflate the tolerance away

    band_z: float = 3.0             # per-band exceedance, in noise sigma
    active_range_db: float = 30.0   # bands this far below the loudest are ignored
    weight_exp: float = 0.25        # band weight = power ** weight_exp (0 = count bands)
    sigma_floor_db: float = 0.35    # protect against near-silent amplification
    sigma_cap_db: float = 6.0       # never normalise a clear change away
    fluct_scale: float = 1.20       # allowed fluctuation, x local sigma
    fluct_floor_db: float = 0.60
    drift_scale: float = 1.20       # allowed mean drift, x local sigma
    drift_floor_db: float = 0.60
    stable_band_frac: float = 0.75  # power-weighted share that must be inside
    stable_change_z: float = 1.50   # no fast change above this inside the window

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not np.isfinite(value):
                raise ValueError(f"{key} 必须是有限数值")
        if not 0 <= self.smooth_s < self.fast_s <= 1.0:
            raise ValueError("要求 0 <= smooth_s < fast_s <= 1 秒")
        if not 0.2 <= self.noise_s <= 30:
            raise ValueError("noise_s 需在 0.2–30 秒")
        if not 0.1 <= self.stable_s <= 10:
            raise ValueError("stable_s 需在 0.1–10 秒")
        if not 0 < self.drift_s <= self.stable_s:
            raise ValueError("drift_s 需在 (0, stable_s]")
        if not 0.2 <= self.slow_s <= 10:
            raise ValueError("slow_s 需在 0.2–10 秒")
        if not 5.0 <= self.slow_cal_s <= 120:
            raise ValueError("slow_cal_s 需在 5–120 秒")
        if not 0.5 <= self.slow_scale <= 10:
            raise ValueError("slow_scale 需在 0.5–10")
        if not 0 < self.slow_floor_db < self.slow_cap_db:
            raise ValueError("要求 0 < slow_floor_db < slow_cap_db")
        if not 0 <= self.bg_lag_s <= 5:
            raise ValueError("bg_lag_s 需在 0–5 秒")
        if not 1.0 <= self.excess_cal_s <= 30:
            raise ValueError("excess_cal_s 需在 1–30 秒")
        if not 0.5 <= self.excess_scale <= 10:
            raise ValueError("excess_scale 需在 0.5–10")
        if not 0 < self.excess_floor_db < self.excess_cap_db:
            raise ValueError("要求 0 < excess_floor_db < excess_cap_db")
        if not 1.0 <= self.band_z <= 10:
            raise ValueError("band_z 需在 1–10 个噪声单位")
        if not 20 <= self.active_range_db <= 90:
            raise ValueError("active_range_db 需在 20–90 dB")
        if not 0 <= self.weight_exp <= 1:
            raise ValueError("weight_exp 需在 0–1")
        if not 0 < self.sigma_floor_db < self.sigma_cap_db:
            raise ValueError("要求 0 < sigma_floor_db < sigma_cap_db")
        if not 0.2 <= self.fluct_scale <= 10 or not 0.1 <= self.drift_scale <= 10:
            raise ValueError("fluct_scale / drift_scale 需在合理范围")
        if not 0.05 <= self.fluct_floor_db <= 12 or not 0.05 <= self.drift_floor_db <= 12:
            raise ValueError("波动/漂移下限需在 0.05–12 dB")
        if not 0.5 <= self.stable_band_frac < 1:
            raise ValueError("stable_band_frac 需在 [0.5, 1)")
        if not 0.1 <= self.stable_change_z <= 10:
            raise ValueError("stable_change_z 需在 0.1–10")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data) -> "StabilityConfig":
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"不支持的稳定性参数 {sorted(unknown)}")
        return cls(**data)


@dataclass
class StabilityFrames:
    times: np.ndarray
    change_z: np.ndarray
    exceed: np.ndarray
    slow_ratio: np.ndarray
    fluct_frac: np.ndarray
    drift_frac: np.ndarray
    stable: np.ndarray
    level_db: np.ndarray
    sigma_db: np.ndarray
    excess_db: np.ndarray
    excess_ratio: np.ndarray
    valid: np.ndarray
    z: np.ndarray
    fluct_ratio: np.ndarray
    drift_ratio: np.ndarray
    peak_change: np.ndarray

    def on_grid(self, target: np.ndarray) -> "StabilityFrames":
        """Nearest-sample every series onto another grid (keeps report alignment)."""
        if len(target) == len(self.times) and np.allclose(target, self.times):
            return self
        index = np.clip(np.searchsorted(self.times, target), 0, len(self.times) - 1)
        return StabilityFrames(
            times=np.asarray(target, dtype=np.float64),
            **{name: getattr(self, name)[index] for name in
               ("change_z", "exceed", "slow_ratio", "fluct_frac", "drift_frac",
                "stable", "level_db", "sigma_db", "excess_db", "excess_ratio", "valid",
                "z", "fluct_ratio", "drift_ratio", "peak_change")})


# ------------------------------------------------------------------ helpers
def _trailing_median(x: np.ndarray, k: int, idx: np.ndarray) -> np.ndarray:
    """Median over ``[i-k+1, i]`` for every ``i`` in ``idx`` (x is T x K)."""
    k = max(1, int(k))
    out = np.empty((len(idx), x.shape[1]), dtype=np.float64)
    for row, i in enumerate(idx):
        out[row] = np.median(x[max(0, i - k + 1): i + 1], axis=0)
    return out


def _trailing_mad(x: np.ndarray, k: int, idx: np.ndarray) -> np.ndarray:
    """Robust local scale (1.4826 x MAD) over ``[i-k+1, i]``."""
    k = max(1, int(k))
    out = np.empty((len(idx), x.shape[1]), dtype=np.float64)
    for row, i in enumerate(idx):
        window = x[max(0, i - k + 1): i + 1]
        centre = np.median(window, axis=0)
        out[row] = 1.4826 * np.median(np.abs(window - centre), axis=0)
    return out


def _interp_rows(values: np.ndarray, idx: np.ndarray, n: int) -> np.ndarray:
    xs = np.arange(n, dtype=np.float64)
    if len(idx) == 1:
        return np.repeat(values, n, axis=0)
    return np.stack([np.interp(xs, idx, values[:, j]) for j in range(values.shape[1])], axis=1)


def _shift_right(x: np.ndarray, n: int) -> np.ndarray:
    n = int(n)
    if n <= 0:
        return np.asarray(x)
    pad = np.repeat(np.asarray(x)[:1], n, axis=0)
    return np.concatenate([pad, np.asarray(x)[:-n]], axis=0)


def _trailing_max(x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k))
    if k == 1:
        return np.asarray(x).copy()
    padded = np.pad(np.asarray(x, dtype=np.float64), (k - 1, 0), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(padded, k).max(axis=1)


def _block_mean_db(L: np.ndarray, step: int) -> np.ndarray:
    """Mean power inside blocks of ``step`` frames, back to dB (per band)."""
    n = len(L)
    step = max(1, int(step))
    power = 10.0 ** (L / 10.0)
    out = np.empty_like(L)
    for a in range(0, n, step):
        b = min(a + step, n)
        out[a:b] = 10.0 * np.log10(np.maximum(power[a:b].mean(axis=0), 1e-12))
    return out


def _masked_median(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    y = np.where(mask, x, np.nan)
    with np.errstate(all="ignore"):
        out = np.nanmedian(y, axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _level(power: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power.sum(axis=1), 1e-12))


# ------------------------------------------------------------------ analysis
def analyze(bundle: FeatureBundle, cfg: StabilityConfig | None = None) -> StabilityFrames:
    """Compute per-frame stability metrics on ``bundle.times``."""
    cfg = cfg or StabilityConfig()
    times = np.asarray(bundle.times, dtype=np.float64)
    if len(times) == 0:
        raise ValueError("空音频，无法分析稳定性")
    step = float(np.median(np.diff(times))) if len(times) > 1 else 0.02
    n = len(times)
    index = np.clip(np.searchsorted(bundle.mel_times, times), 0, len(bundle.mel_times) - 1)
    power = bundle.mel_power[index].astype(np.float64)
    L = 10.0 * np.log10(np.maximum(power, 1e-12))

    half_fft = bundle.config.n_fft / bundle.config.sr / 2
    valid = (times >= half_fft) & (times <= bundle.duration_s - half_fft)

    fast = max(1, int(round(cfg.fast_s / step)))
    smooth = max(1, int(round(cfg.smooth_s / step)) | 1)
    noise = max(3, int(round(cfg.noise_s / step)) | 1)
    stable = max(3, int(round(cfg.stable_s / step)) | 1)
    drift = max(3, int(round(cfg.drift_s / step)) | 1)
    slow = max(1, int(round(cfg.slow_s / step)))
    bg = max(3, int(round(cfg.bg_s / step)) | 1)
    lag = max(0, int(round(cfg.bg_lag_s / step)))
    stride = max(1, int(round(0.1 / step)))

    if n < max(noise, stable) + fast:
        zero = np.zeros(n)
        return StabilityFrames(times, zero, zero, zero, np.ones(n), np.ones(n),
                               np.zeros(n, bool), _level(power), np.zeros(n), zero, zero,
                               np.zeros(n, bool), np.zeros_like(power), np.ones_like(power),
                               np.ones_like(power), zero)

    Ls = L.copy()
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        Ls = np.stack([np.convolve(L[:, j], kernel, mode="same") for j in range(L.shape[1])], axis=1)
    d = np.zeros_like(Ls)
    d[fast:] = Ls[fast:] - Ls[:-fast]

    coarse = np.unique(np.r_[np.arange(0, n, stride), n - 1]).astype(int)
    sigma_c = np.clip(_trailing_mad(d, noise, coarse), cfg.sigma_floor_db, cfg.sigma_cap_db)
    sigma = _interp_rows(sigma_c, coarse, n)
    z = np.abs(d) / sigma

    # Bands far below the loudest one are dominated by the numerical floor and
    # must not be judged: a stable tone leaves them at -100 dB.
    active = L >= (L.max(axis=1, keepdims=True) - cfg.active_range_db)
    band_weight = np.where(active, np.maximum(power, 1e-12) ** cfg.weight_exp, 0.0)
    weight_sum = np.maximum(band_weight.sum(axis=1), 1e-12)
    exceed = ((z > cfg.band_z) * band_weight).sum(axis=1) / weight_sum
    weights = np.where(active, np.maximum(power, 0.0) ** (2 * cfg.weight_exp), 0.0)
    energy_sum = np.maximum(weights.sum(axis=1), 1e-12)
    change_z = np.sqrt((weights * z ** 2).sum(axis=1) / energy_sum)

    # Slow ramp: consecutive one-second blocks, power weighted median.
    blocks = _block_mean_db(L, slow)
    slow_delta = np.abs(blocks - _shift_right(blocks, slow))
    # Same principle as the excess test: a slow change is only unusual when it
    # is larger than this band's own slow variability (wind swells are not
    # events, a real ramp is).
    slow_cal = max(3, int(round(cfg.slow_cal_s / step)) | 1)
    slow_scale_band = _trailing_mad(slow_delta, slow_cal, coarse)
    slow_tol = np.clip(cfg.slow_scale * _interp_rows(slow_scale_band, coarse, n),
                       cfg.slow_floor_db, cfg.slow_cap_db)
    slow_ratio = _masked_median(slow_delta / slow_tol, active)

    fluct_c = _trailing_mad(Ls, stable, coarse)
    fluct = _interp_rows(fluct_c, coarse, n)
    fluct_tol = np.maximum(cfg.fluct_floor_db, cfg.fluct_scale * sigma)
    fluct_frac = ((fluct <= fluct_tol) * band_weight).sum(axis=1) / weight_sum

    mean_c = _trailing_median(Ls, drift, coarse)
    mean_now = _interp_rows(mean_c, coarse, n)
    mean_before = _shift_right(mean_now, drift)
    drift_tol = np.maximum(cfg.drift_floor_db, cfg.drift_scale * sigma)
    drift_frac = ((np.abs(mean_now - mean_before) <= drift_tol) * band_weight).sum(axis=1) / weight_sum

    warmup = max(stable, fast + smooth + 2)
    change_z[:warmup] = 0.0
    exceed[:warmup] = 0.0
    peak = _trailing_max(change_z, stable)
    stable_flag = ((fluct_frac >= cfg.stable_band_frac)
                   & (drift_frac >= cfg.stable_band_frac)
                   & (peak <= cfg.stable_change_z)
                   & valid)
    stable_flag[:warmup] = False

    background_c = _trailing_median(Ls, bg, coarse)
    background = _shift_right(_interp_rows(background_c, coarse, n), lag)
    residual = Ls - background
    # The excess test asks "is the level back around its recent background?".
    # Its tolerance must therefore reflect how much *this deviation* normally
    # wobbles, not the 0.1 s difference noise: LF rumble drifts a lot with a
    # tiny frame-to-frame difference, and a fast-sigma tolerance would then
    # never be met (stuck alarms) while quiet bands would trigger it constantly.
    excess_cal = max(3, int(round(cfg.excess_cal_s / step)) | 1)
    excess_scale_band = _trailing_mad(residual, excess_cal, coarse)
    excess_tol = np.clip(cfg.excess_scale * _interp_rows(excess_scale_band, coarse, n),
                         cfg.excess_floor_db, cfg.excess_cap_db)
    excess_band = np.abs(residual) > excess_tol
    excess = (np.abs(residual) * band_weight).sum(axis=1) / weight_sum
    excess_ratio = (excess_band * band_weight).sum(axis=1) / weight_sum

    change_z[~valid] = 0.0
    exceed[~valid] = 0.0
    excess_ratio[~valid] = 0.0
    fluct_ratio = fluct / np.maximum(sigma, 1e-9)
    drift_ratio = np.abs(mean_now - mean_before) / np.maximum(sigma, 1e-9)
    return StabilityFrames(times, change_z, exceed, slow_ratio, fluct_frac,
                           drift_frac, stable_flag, _level(power),
                           np.median(sigma, axis=1), excess, excess_ratio, valid, z,
                           fluct_ratio, drift_ratio, peak)


def audio_quality(waveform: np.ndarray, sr: int) -> dict:
    """Non-destructive acquisition diagnostics; never used to alter the audio."""
    x = np.asarray(waveform, dtype=np.float64)
    if not len(x):
        return {"clip_ratio": 0.0, "crest_db": None, "rms_dbfs": None}
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x ** 2)))
    return {
        "clip_ratio": float(np.mean(np.abs(x) >= 0.999)),
        "crest_db": float(20 * np.log10(peak + 1e-12) - 20 * np.log10(rms + 1e-12)),
        "rms_dbfs": float(20 * np.log10(rms + 1e-12)),
    }
