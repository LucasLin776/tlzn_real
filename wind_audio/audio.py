"""WAV I/O, resampling, STFT and log-mel features.

Only numpy + the standard library are required.  All frames are centred on
their nominal timestamps: frame i is centred at i * hop / sr seconds, with
n_fft // 2 samples of reflection padding at both ends.  This makes event
boundaries and plotting coordinates line up with the real file duration
instead of "frames * hop".
"""
from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class WavInfo:
    path: str
    sample_rate: int
    channels: int
    sample_width: int
    n_frames: int
    duration_s: float


def wav_info(path: str | Path) -> WavInfo:
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as w:
            sr = w.getframerate()
            n = w.getnframes()
            return WavInfo(
                path=str(path),
                sample_rate=sr,
                channels=w.getnchannels(),
                sample_width=w.getsampwidth(),
                n_frames=n,
                duration_s=(n / sr) if sr else 0.0,
            )
    except wave.Error as exc:
        raise ValueError(f"不是可读的 PCM WAV: {path} ({exc})") from exc
    except FileNotFoundError:
        raise FileNotFoundError(f"音频不存在: {path}") from None


def load_wav(path: str | Path, target_sr: int = 44100) -> tuple[np.ndarray, int]:
    """Read a PCM wav as mono float32 in [-1, 1]."""
    path = Path(path)
    info = wav_info(path)
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(info.n_frames)

    if info.sample_width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif info.sample_width == 1:
        x = (np.frombuffer(raw, dtype="u1").astype(np.float32) - 128.0) / 128.0
    elif info.sample_width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"不支持的位深: {info.sample_width * 8} bit")

    if info.channels > 1:
        x = x.reshape(-1, info.channels).mean(axis=1)

    if info.sample_rate != target_sr:
        x = resample_fft(x, info.sample_rate, target_sr)
    return np.ascontiguousarray(x, dtype=np.float32), int(target_sr)


def write_wav(path: str | Path, x: np.ndarray, sr: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    pcm = (y * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    return path


def resample_fft(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """FFT resampling. Good enough for features, not for critical listening."""
    if sr == target_sr:
        return np.asarray(x, dtype=np.float32)
    n = len(x)
    if n == 0:
        return np.asarray(x, dtype=np.float32)
    m = int(round(n * target_sr / sr))
    X = np.fft.rfft(x)
    Y = np.zeros(m // 2 + 1, dtype=np.complex128)
    k = min(len(X), len(Y))
    Y[:k] = X[:k]
    y = np.fft.irfft(Y, m)
    return (y * (m / n)).astype(np.float32)


def _pad_centred(x: np.ndarray, n_fft: int) -> np.ndarray:
    pad = n_fft // 2
    if len(x) == 0:
        raise ValueError("空音频，无法计算 STFT")
    if len(x) > pad:
        return np.pad(x, (pad, pad), mode="reflect")
    return np.pad(x, (pad, pad), mode="constant")


def stft(
    x: np.ndarray,
    n_fft: int = 2048,
    hop: int = 441,
    center: bool = True,
) -> np.ndarray:
    """Complex STFT, shape (frames, n_fft//2+1)."""
    x = np.asarray(x, dtype=np.float32)
    if center:
        padded = _pad_centred(x, n_fft)
    else:
        padded = np.pad(x, (0, max(0, n_fft - len(x)))) if len(x) < n_fft else x
    window = np.hanning(n_fft).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop]
    return np.fft.rfft(frames * window, axis=1).astype(np.complex64)


def frame_times(n_frames: int, hop: int, sr: int, center: bool = True, n_fft: int = 2048) -> np.ndarray:
    if center:
        return np.arange(n_frames, dtype=np.float64) * hop / sr
    return (np.arange(n_frames, dtype=np.float64) * hop + n_fft / 2.0) / sr


def frame_rms_db(x: np.ndarray, n_fft: int = 2048, hop: int = 441, center: bool = True) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    padded = _pad_centred(x, n_fft) if center else (
        np.pad(x, (0, max(0, n_fft - len(x)))) if len(x) < n_fft else x
    )
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return (20.0 * np.log10(rms + 1e-10)).astype(np.float32)


def hz_to_mel(f: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int = 128,
    fmin: float = 20.0,
    fmax: float = 16000.0,
) -> np.ndarray:
    """Triangular mel filters, area-normalised, shape (n_mels, n_fft//2+1)."""
    fmax = min(float(fmax), sr / 2.0)
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * edges / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, centre, right = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        centre = max(centre, left + 1)
        right = max(right, centre + 1)
        if centre > n_fft // 2:
            break
        for k in range(left, min(centre, n_fft // 2 + 1)):
            fb[i, k] = (k - left) / max(centre - left, 1)
        for k in range(centre, min(right, n_fft // 2 + 1)):
            fb[i, k] = (right - k) / max(right - centre, 1)
    total = fb.sum(axis=1, keepdims=True)
    total[total == 0] = 1.0
    return fb / total


def log_mel(
    x: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop: int = 441,
    n_mels: int = 128,
    fmin: float = 20.0,
    fmax: float = 16000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (log-mel dB [frames, mels], stft freqs, frame timestamps)."""
    spec = np.abs(stft(x, n_fft, hop, center=True)) ** 2
    fb = mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
    mel = spec @ fb.T
    mel_db = (10.0 * np.log10(mel + 1e-10)).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    times = frame_times(mel_db.shape[0], hop, sr, center=True, n_fft=n_fft)
    return mel_db, freqs, times


def median_filter(x: np.ndarray, k: int) -> np.ndarray:
    """1-D median filter, edge padded, odd kernel."""
    if k <= 1 or len(x) == 0:
        return np.asarray(x).copy()
    if k % 2 == 0:
        k += 1
    pad = k // 2
    padded = np.pad(np.asarray(x), (pad, pad), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, k)
    return np.median(windows, axis=1).astype(np.float32)
