"""PNG plots: waveform, spectrogram and score curve on a real time axis.

Low frequencies are at the bottom of the spectrogram.  All x coordinates use
``t / duration * (width - 1)`` so the three plots, event boxes and annotation
strips line up exactly.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

_RAMP = (
    np.array(
        [
            [0, 0, 4], [22, 11, 57], [66, 10, 104], [106, 23, 110],
            [147, 38, 103], [188, 55, 84], [221, 81, 58], [243, 120, 25],
            [250, 165, 15], [252, 210, 60], [252, 255, 164],
        ],
        dtype=np.float64,
    )
    / 255.0
)
_RAMP_POS = np.linspace(0.0, 1.0, len(_RAMP))

REGION_COLORS = {
    "normal": (224, 244, 230),
    "anomaly": (255, 226, 226),
    "uncertain": (236, 236, 236),
}
EVENT_COLOR = (226, 116, 20)


def write_png(path: str | Path, rgb: np.ndarray) -> Path:
    path = Path(path)
    rgb = np.ascontiguousarray(rgb.astype(np.uint8))
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(
            ">I", zlib.crc32(kind + data) & 0xFFFFFFFF
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
    return path


def colorize(values: np.ndarray) -> np.ndarray:
    v = np.clip(values, 0.0, 1.0)
    out = np.stack([np.interp(v, _RAMP_POS, _RAMP[:, c]) for c in range(3)], axis=-1)
    return (out * 255.0 + 0.5).astype(np.uint8)


def _x_of(t: float, duration: float, width: int) -> int:
    if duration <= 0:
        return 0
    return int(round(np.clip(t / duration, 0.0, 1.0) * (width - 1)))


def _vline(img: np.ndarray, x: int, color, width: int = 2) -> None:
    h, w, _ = img.shape
    if 0 <= x < w:
        img[:, max(0, x) : min(w, x + width)] = color


def _shade(img: np.ndarray, regions: list[dict], duration: float) -> None:
    h, w, _ = img.shape
    for r in regions or []:
        x0 = _x_of(r["start_s"], duration, w)
        x1 = _x_of(r["end_s"], duration, w)
        color = REGION_COLORS.get(r.get("label"), REGION_COLORS["uncertain"])
        img[:, max(0, x0) : min(w, x1 + 1)] = color


def _region_strip(img: np.ndarray, regions: list[dict], duration: float, y0: int = 0, y1: int = 7) -> None:
    h, w, _ = img.shape
    for r in regions or []:
        x0 = _x_of(r["start_s"], duration, w)
        x1 = _x_of(r["end_s"], duration, w)
        color = REGION_COLORS.get(r.get("label"), REGION_COLORS["uncertain"])
        img[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1 + 1)] = color


def _event_boxes(img: np.ndarray, events: list[dict], duration: float) -> None:
    h, w, _ = img.shape
    for ev in events or []:
        x0 = _x_of(ev["start_s"], duration, w)
        x1 = _x_of(ev["end_s"], duration, w)
        _vline(img, x0, EVENT_COLOR, 2)
        _vline(img, x1, EVENT_COLOR, 2)
        img[0:4, max(0, x0) : min(w, x1 + 1)] = EVENT_COLOR
        img[-4:, max(0, x0) : min(w, x1 + 1)] = EVENT_COLOR


def waveform_png(
    x: np.ndarray,
    duration: float,
    events: list[dict],
    regions: list[dict],
    path: str | Path,
    width: int = 1000,
    height: int = 170,
) -> Path:
    canvas = np.full((height, width, 3), 252, dtype=np.uint8)
    _shade(canvas, regions, duration)
    mid = height // 2
    canvas[mid : mid + 1, :] = (210, 214, 220)
    n = len(x)
    if n > 0:
        edges = np.linspace(0, n, width + 1).astype(np.int64)
        for col in range(width):
            a, b = int(edges[col]), int(edges[col + 1])
            if b <= a:
                continue
            seg = x[a:b]
            lo = float(seg.min())
            hi = float(seg.max())
            y0 = int(round((1.0 - (hi + 1.0) / 2.0) * (height - 1)))
            y1 = int(round((1.0 - (lo + 1.0) / 2.0) * (height - 1)))
            y0, y1 = max(0, min(height - 1, y0)), max(0, min(height - 1, y1))
            if y1 < y0:
                y0, y1 = y1, y0
            canvas[y0 : y1 + 1, col] = (32, 62, 130)
    _event_boxes(canvas, events, duration)
    return write_png(path, canvas)


def spectrogram_png(
    spec_db: np.ndarray,
    freqs: np.ndarray,
    mel_times: np.ndarray,
    duration: float,
    events: list[dict],
    regions: list[dict],
    path: str | Path,
    f_max: float = 20000.0,
    width: int = 1000,
    height: int = 380,
) -> Path:
    n_frames, n_bins = spec_db.shape
    keep = max(4, min(int(np.searchsorted(freqs, f_max)), n_bins))
    frame_grid = np.linspace(0.0, max(duration - 1e-6, 0.0), width)
    frame_dt = float(np.median(np.diff(mel_times))) if len(mel_times) > 1 else 0.01
    frame_idx = np.clip(np.rint(frame_grid / max(frame_dt, 1e-6)).astype(int), 0, n_frames - 1)
    freq_grid = np.linspace(0.0, min(f_max, freqs[keep - 1]), height)
    freq_idx = np.clip(np.searchsorted(freqs[:keep], freq_grid), 0, keep - 1)
    pooled = spec_db[np.ix_(frame_idx, freq_idx)].T  # row 0 = 0 Hz
    pooled = pooled[::-1]  # image top = high frequency

    vmin = float(np.percentile(pooled, 5))
    vmax = float(np.percentile(pooled, 99.5))
    if vmax <= vmin:
        vmax = vmin + 1.0
    rgb = colorize((pooled - vmin) / (vmax - vmin))
    _region_strip(rgb, regions, duration, 0, 7)
    _event_boxes(rgb, events, duration)
    return write_png(path, rgb)


def score_png(
    times: np.ndarray,
    score: np.ndarray,
    threshold: float,
    events: list[dict],
    regions: list[dict],
    duration: float,
    path: str | Path,
    width: int = 1000,
    height: int = 170,
) -> Path:
    canvas = np.full((height, width, 3), 252, dtype=np.uint8)
    _shade(canvas, regions, duration)
    n = len(score)
    if n == 0:
        return write_png(path, canvas)
    lo = float(min(np.nanmin(score), threshold))
    hi = float(max(np.nanmax(score), threshold * 1.2, lo + 1e-6))
    margin = 0.08 * (hi - lo)
    lo -= margin
    hi += margin

    def y_of(v: float) -> int:
        return int(round((1.0 - (v - lo) / (hi - lo)) * (height - 3))) + 1

    yt = y_of(threshold)
    for xx in range(0, width, 14):
        canvas[max(0, yt) : min(height, yt + 2), xx : min(width, xx + 7)] = (205, 55, 55)

    xs = np.linspace(0, width - 1, n).astype(np.int64)
    ys = np.array([y_of(float(v)) for v in np.nan_to_num(score, nan=lo)], dtype=np.int64)
    for xx, yy in zip(xs, ys):
        canvas[max(0, yy - 1) : min(height, yy + 2), max(0, xx - 1) : min(width, xx + 2)] = (32, 62, 130)
    _event_boxes(canvas, events, duration)
    return write_png(path, canvas)
