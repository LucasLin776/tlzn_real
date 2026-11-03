"""Tiny PNG renderer for spectrograms and anomaly-score curves.

No matplotlib / Pillow dependency: we write RGB PNGs directly with zlib.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

# inferno-ish colour ramp
_RAMP = (
    np.array(
        [
            [0, 0, 4],
            [22, 11, 57],
            [66, 10, 104],
            [106, 23, 110],
            [147, 38, 103],
            [188, 55, 84],
            [221, 81, 58],
            [243, 120, 25],
            [250, 165, 15],
            [252, 210, 60],
            [252, 255, 164],
        ],
        dtype=np.float64,
    )
    / 255.0
)
_RAMP_POS = np.linspace(0.0, 1.0, len(_RAMP))


def write_png(path: str | Path, rgb: np.ndarray) -> Path:
    path = Path(path)
    rgb = np.ascontiguousarray(rgb.astype(np.uint8))
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
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


def _vline(img: np.ndarray, x: int, color: tuple[int, int, int], width: int = 2) -> None:
    h, w, _ = img.shape
    if 0 <= x < w:
        img[:, max(0, x) : min(w, x + width)] = color


def _hline(img: np.ndarray, y: int, color: tuple[int, int, int], width: int = 2) -> None:
    h, w, _ = img.shape
    if 0 <= y < h:
        img[max(0, y) : min(h, y + width), :] = color


def spectrogram_png(
    spec_db: np.ndarray,
    freqs: np.ndarray,
    events: list[dict],
    path: str | Path,
    f_max: float = 20000.0,
    width: int = 1000,
    height: int = 420,
) -> Path:
    """spec_db: (frames, bins) dB. Event boxes are drawn as bright outlines."""
    n_frames, n_bins = spec_db.shape
    keep = int(np.searchsorted(freqs, f_max))
    keep = max(4, min(keep, n_bins))
    img = spec_db[:, :keep]

    rows = np.linspace(0, img.shape[0] - 1, width).astype(np.int64)
    cols = np.linspace(0, img.shape[1] - 1, height).astype(np.int64)
    pooled = img[np.ix_(rows, cols)].T  # (height, width)

    vmin = float(np.percentile(pooled, 5))
    vmax = float(np.percentile(pooled, 99.5))
    if vmax <= vmin:
        vmax = vmin + 1.0
    rgb = colorize((pooled - vmin) / (vmax - vmin))

    box = (80, 255, 120)
    for event in events:
        x0 = int(round(event["i0"] / max(n_frames - 1, 1) * (width - 1)))
        x1 = int(round(event["i1"] / max(n_frames - 1, 1) * (width - 1)))
        _vline(rgb, x0, box, 2)
        _vline(rgb, x1, box, 2)
        _hline(rgb, 0, box, 3)

    return write_png(path, rgb)


def score_png(
    score: np.ndarray,
    events: list[dict],
    threshold: float,
    path: str | Path,
    width: int = 1000,
    height: int = 180,
) -> Path:
    n = len(score)
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    canvas[:, :, 1] = 249
    canvas[:, :, 2] = 252

    for event in events:
        x0 = int(round(event["i0"] / max(n - 1, 1) * (width - 1)))
        x1 = int(round(event["i1"] / max(n - 1, 1) * (width - 1)))
        canvas[:, max(0, x0) : min(width, x1 + 1)] = (255, 236, 236)

    lo = float(min(score.min(), 0.0))
    hi = float(max(score.max(), threshold * 1.3, lo + 1.0))
    margin = 0.08 * (hi - lo)
    lo -= margin
    hi += margin

    def y_of(v: float) -> int:
        return int(round((1.0 - (v - lo) / (hi - lo)) * (height - 3))) + 1

    # threshold, dashed
    yt = y_of(threshold)
    for x in range(0, width, 14):
        canvas[max(0, yt) : min(height, yt + 2), x : min(width, x + 7)] = (210, 60, 60)

    # score line
    xs = np.linspace(0, width - 1, n).astype(np.int64)
    ys = np.array([y_of(float(v)) for v in score], dtype=np.int64)
    for x, y in zip(xs, ys):
        canvas[max(0, y - 1) : min(height, y + 2), max(0, x - 1) : min(width, x + 2)] = (32, 62, 130)

    return write_png(path, canvas)
