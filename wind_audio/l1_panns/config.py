"""Versioned architecture and physical time support (no interpolated precision)."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class NetworkConfig:
    sample_rate: int = 32000
    n_fft: int = 1024
    hop: int = 320
    mel_bins: int = 64
    level_center_db: float = -40.0
    short_lag: int = 1
    long_lag: int = 4
    local_radius: int = 4
    hidden: int = 96
    bottleneck: int = 24
    scale_floor: float = 0.01
    core_frames: int = 32

    def __post_init__(self):
        if (self.sample_rate, self.n_fft, self.hop, self.mel_bins) != (32000, 1024, 320, 64):
            raise ValueError("v1 仅支持官方 CNN14 32kHz / FFT1024 / hop320 / mel64")
        ints = (self.short_lag, self.long_lag, self.local_radius,
                self.hidden, self.bottleneck, self.core_frames)
        if any(type(v) is not int or v <= 0 for v in ints):
            raise ValueError("网络维度/时间步必须为正整数")
        if not self.short_lag < self.long_lag or not self.bottleneck < self.hidden < self.input_dim:
            raise ValueError("需要短/长差分和严格压缩瓶颈")
        if not math.isfinite(self.scale_floor) or self.scale_floor <= 0 or not math.isfinite(self.level_center_db):
            raise ValueError("非法标准化配置")

    @property
    def input_dim(self):
        return (64 + 128 + 256) * 3

    @property
    def stride_samples(self):
        return self.hop * 8

    @property
    def step_s(self):
        return self.stride_samples / self.sample_rate

    @property
    def offset_s(self):
        # center=False STFT center + three successive 2x pooling centers.
        return (self.n_fft / 2 + 3.5 * self.hop) / self.sample_rate

    @property
    def encoder_radius_s(self):
        # B3 receptive field = 36 STFT frames; B1/B2 aligned by pooling, not interpolation.
        return (self.n_fft + 35 * self.hop) / self.sample_rate / 2

    @property
    def context_radius_s(self):
        # Symmetric upper bound, including backward-only differences.
        return self.encoder_radius_s + (max(self.long_lag, self.local_radius) + 6) * self.step_s

    @property
    def halo(self):
        return math.ceil(self.context_radius_s / self.step_s)

    def samples_for_frames(self, frames):
        return frames * self.stride_samples + self.n_fft - self.hop

    def timing(self):
        return {"native_step_s": self.step_s, "first_center_s": self.offset_s,
                "encoder_receptive_field_s": 2 * self.encoder_radius_s,
                "score_receptive_field_bound_s": 2 * self.context_radius_s,
                "resampler_support": "add 16/native_sample_rate on each side",
                "interpolation": False,
                "note": "80ms is the sampling grid, NOT proven event-boundary accuracy"}

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class EventConfig:
    confirm_s: float = 0.32
    merge_gap_s: float = 0.16
    trigger_frames: int = 2

    def __post_init__(self):
        if not math.isfinite(self.confirm_s) or self.confirm_s <= 0:
            raise ValueError("confirm_s 必须为正数")
        if not math.isfinite(self.merge_gap_s) or self.merge_gap_s < 0:
            raise ValueError("merge_gap_s 必须非负")
        if type(self.trigger_frames) is not int or self.trigger_frames < 1:
            raise ValueError("trigger_frames 必须为正整数")

    def to_dict(self):
        return asdict(self)
