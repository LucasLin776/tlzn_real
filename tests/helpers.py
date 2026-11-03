from __future__ import annotations

from pathlib import Path

import numpy as np

from wind_audio.audio import write_wav
from wind_audio.data import Annotation, AudioRecord


def make_wav(path: Path, sr: int = 8000, duration: float = 6.0, seed: int = 0,
             level: float = 0.01, burst: tuple[float, float, float] | None = None,
             gain: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * duration)
    x = level * rng.standard_normal(n).astype(np.float32)
    if burst:
        start, dur, amp = burst
        a, b = int(start * sr), int(min(n, (start + dur) * sr))
        noise = rng.standard_normal(b - a).astype(np.float32)
        noise = np.diff(noise, prepend=noise[0])
        x[a:b] += amp * noise
    x = (gain * x).astype(np.float32)
    write_wav(path, x, sr)
    return x


def record(audio_id: str, path: str, duration: float, source: str = "", device: str = "",
           fully_reviewed: bool = True) -> AudioRecord:
    return AudioRecord(
        audio_id=audio_id, path=path, source_id=source or audio_id,
        device_id=device, duration_s=duration, sample_rate=8000, channels=1,
        fully_reviewed=fully_reviewed,
    )


def ann(annotation_id: str, audio_id: str, a: float, b: float, label: str = "anomaly",
        precision: str = "interval") -> Annotation:
    return Annotation(
        annotation_id=annotation_id, audio_id=audio_id, start_s=a, end_s=b,
        label=label, precision=precision,
    )
