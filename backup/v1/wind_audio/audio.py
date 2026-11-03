"""WAV I/O, resampling and time-frequency features.

Only numpy + the standard library are required, so this runs on a bare
macOS Python without installing librosa/scipy/torch.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def load_wav(path: str | Path, target_sr: int = 44100) -> tuple[np.ndarray, int]:
    """Read a PCM wav as mono float32 in [-1, 1]."""
    path = Path(path)
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        n = w.getnframes()
        raw = w.readframes(n)

    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        x = (np.frombuffer(raw, dtype="u1").astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {width}")

    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)

    if sr != target_sr:
        x = resample_fft(x, sr, target_sr)
        sr = target_sr
    return np.ascontiguousarray(x, dtype=np.float32), sr


def resample_fft(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """FFT resampling. Good enough for feature extraction, not for listening."""
    if sr == target_sr:
        return x
    n = len(x)
    m = int(round(n * target_sr / sr))
    X = np.fft.rfft(x)
    Y = np.zeros(m // 2 + 1, dtype=np.complex128)
    k = min(len(X), len(Y))
    Y[:k] = X[:k]
    y = np.fft.irfft(Y, m)
    return (y * (m / n)).astype(np.float32)


def stft(x: np.ndarray, n_fft: int = 2048, hop: int = 441) -> np.ndarray:
    """Complex STFT, shape (frames, n_fft//2+1)."""
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    window = np.hanning(n_fft).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
    return np.fft.rfft(frames * window, axis=1).astype(np.complex64)


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
    """Return (log-mel dB [frames, mels], stft freqs, frame centre times)."""
    spec = np.abs(stft(x, n_fft, hop)) ** 2
    fb = mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
    mel = spec @ fb.T
    mel_db = (10.0 * np.log10(mel + 1e-10)).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    times = np.arange(mel_db.shape[0], dtype=np.float32) * hop / sr
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
