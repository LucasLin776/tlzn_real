"""Unfitted spectrogram features shared by L1 scoring and reports."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np

from .audio import frame_times, mel_filterbank, stft


@dataclass(frozen=True)
class FeatureConfig:
    sr: int = 44100
    n_fft: int = 2048
    hop: int = 441
    n_mels: int = 64
    fmin: float = 20.0
    fmax: float = 16000.0
    target_step: float = 0.02
    compare_s: float = 0.1
    smooth_s: float = 0.06
    dynamic_range_db: float = 40.0

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not np.isfinite(value):
                raise ValueError(f"{key} 必须是有限数值")
        if any(not isinstance(x, int) for x in (self.sr, self.n_fft, self.hop, self.n_mels)):
            raise ValueError("采样率、FFT、帧移、mel 数必须为整数")
        if self.sr < 8000 or self.n_fft < 256 or not 0 < self.hop <= self.n_fft or self.n_mels < 4:
            raise ValueError("无效频谱配置")
        if not 0 <= self.fmin < self.fmax <= self.sr / 2:
            raise ValueError("要求 0 <= fmin < fmax <= sr/2")
        if not self.hop / self.sr <= self.target_step <= self.compare_s <= 2:
            raise ValueError("要求 帧移 <= 输出步长 <= 比较窗口 <= 2 秒")
        if not 0 <= self.smooth_s <= 1 or not 10 <= self.dynamic_range_db <= 80:
            raise ValueError("平滑需在 0–1 秒，频谱动态范围需在 10–80 dB")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"不支持的特征配置 {sorted(unknown)}；旧 L1 模型不能作为新检测配置")
        return cls(**data)

    def assert_compatible(self, other):
        if self != other:
            raise ValueError("频谱配置不一致")


@dataclass
class FeatureBundle:
    audio_id: str
    duration_s: float
    sr: int
    times: np.ndarray
    mel_power: np.ndarray
    mel_times: np.ndarray
    freqs: np.ndarray
    spec_db: np.ndarray
    config: FeatureConfig


def extract_features(x, sr, cfg: FeatureConfig, audio_id):
    x = np.asarray(x, dtype=np.float32)
    if not len(x) or x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError(f"{audio_id}: 需要非空、有限、单声道音频")
    if sr != cfg.sr:
        raise ValueError("请先重采样到配置中的采样率")
    duration = len(x) / sr
    spec = np.abs(stft(x, cfg.n_fft, cfg.hop, center=True)) ** 2
    mel = spec @ mel_filterbank(sr, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax).T
    times = (np.arange(max(1, int(np.ceil(duration / cfg.target_step)))) + 0.5) * cfg.target_step
    times = times[times < duration]
    if not len(times):
        times = np.array([duration / 2])
    return FeatureBundle(
        audio_id, duration, sr, times, mel,
        frame_times(len(spec), cfg.hop, sr), np.fft.rfftfreq(cfg.n_fft, 1 / sr),
        (10 * np.log10(spec + 1e-12)).astype(np.float32), cfg,
    )
