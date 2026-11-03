"""Deterministic synthetic dataset for pipeline/logic validation.

This is NOT a measure of real wind-turbine performance.  It covers the cases
the implementation must survive: normal background, a normal background
switch, a short impact, a sustained anomaly, a fixed recording gain change,
and anomalies touching the first/last sample.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .audio import write_wav
from .data import Annotation, AudioRecord, write_annotations, write_manifest


def _noise(rng, n: int, alpha: float = 1.0) -> np.ndarray:
    w = rng.standard_normal(n)
    spec = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1.0 / 44100)
    spec[1:] /= np.maximum(f[1:], 1.0) ** (alpha / 2.0)
    spec[0] = 0.0
    x = np.fft.irfft(spec, n)
    return (x / (np.std(x) + 1e-12)).astype(np.float32)


def _background(rng, n: int, level: float, tilt: float, hum: float = 0.002) -> np.ndarray:
    t = np.arange(n) / 44100
    x = level * _noise(rng, n, tilt)
    x += hum * np.sin(2 * np.pi * 50 * t) + 0.5 * hum * np.sin(2 * np.pi * 100 * t)
    x *= 1.0 + 0.08 * np.sin(2 * np.pi * 0.3 * t)
    return x.astype(np.float32)


def _burst(rng, n: int, sr: int, start: float, dur: float, amp: float, highpass: bool = True) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    a, b = int(start * sr), int(min(n, (start + dur) * sr))
    if b <= a:
        return out
    m = b - a
    t = np.arange(m) / sr
    noise = rng.standard_normal(m).astype(np.float32)
    if highpass:
        # crude high-pass: difference removes most low-frequency energy
        noise = np.diff(noise, prepend=noise[0])
    env = np.ones(m, dtype=np.float32)
    fade = max(1, int(0.01 * sr))
    if m > 2 * fade:
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
    out[a:b] = amp * noise * env
    return out


def generate(out_dir: str | Path, seed: int = 0) -> dict:
    out = Path(out_dir)
    audio = out / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    sr = 44100
    dur = 12.0
    n = int(sr * dur)
    rng = np.random.default_rng(seed)
    records: list[AudioRecord] = []
    anns: list[Annotation] = []
    counter = 0

    def add(audio_id: str, source: str, device: str, x: np.ndarray, anomalies: list[tuple[float, float]], note: str):
        nonlocal counter
        counter += 1
        path = audio / f"{audio_id}.wav"
        write_wav(path, x, sr)
        records.append(
            AudioRecord(
                audio_id=audio_id, path=str(path.relative_to(out)), source_id=source,
                device_id=device, duration_s=dur, sample_rate=sr, channels=1,
                fully_reviewed=True, notes=note,
            )
        )
        for a, b in anomalies:
            anns.append(
                Annotation(
                    annotation_id=f"syn_{counter:03d}_{len(anns):03d}",
                    audio_id=audio_id, start_s=float(a), end_s=float(b),
                    label="anomaly", precision="interval", note=note, version="synthetic",
                )
            )

    base = _background(rng, n, 0.01, 1.0)
    add("syn_normal_a", "S1", "D1", base, [], "normal background A")
    add("syn_normal_gain", "S2", "D1", 0.4 * _background(rng, n, 0.01, 1.0), [], "same background, fixed -8 dB gain")

    switch = base.copy()
    switch[int(5 * sr) :] = 0.7 * _background(rng, n - int(5 * sr), 0.01, 0.5)
    add("syn_switch", "S3", "D2", switch, [], "normal background switch at 5 s")

    short = base.copy()
    short += _burst(rng, n, sr, 2.0, 0.12, 0.25)
    add("syn_short_impact", "S4", "D2", short, [(2.0, 2.12)], "0.12 s impact")

    sustained = base.copy()
    sustained += _burst(rng, n, sr, 6.0, 3.0, 0.18)
    add("syn_sustained", "S5", "D3", sustained, [(6.0, 9.0)], "3 s broadband anomaly")

    edge_start = base.copy()
    edge_start[: int(0.4 * sr)] += _burst(rng, n, sr, 0.0, 0.4, 0.25)[: int(0.4 * sr)]
    add("syn_edge_start", "S6", "D3", edge_start, [(0.0, 0.4)], "anomaly at first sample")

    edge_end = base.copy()
    edge_end[int(11.6 * sr) :] += _burst(rng, n, sr, 11.6, 0.4, 0.25)[int(11.6 * sr) :]
    add("syn_edge_end", "S7", "D3", edge_end, [(11.6, 12.0)], "anomaly at last sample")

    mixed = switch.copy()
    mixed += _burst(rng, n, sr, 7.0, 0.12, 0.25)
    mixed += _burst(rng, n, sr, 9.0, 1.5, 0.18)
    add("syn_mixed", "S8", "D2", mixed, [(7.0, 7.12), (9.0, 10.5)], "switch + short + sustained")

    add("syn_normal_b", "S9", "D4", _background(rng, n, 0.01, 0.8), [], "normal background C")
    sustained_b = base.copy()
    sustained_b += _burst(rng, n, sr, 4.0, 5.0, 0.16)
    add("syn_sustained_b", "S10", "D4", sustained_b, [(4.0, 9.0)], "5 s sustained anomaly")
    short_b = base.copy()
    short_b += _burst(rng, n, sr, 5.0, 0.3, 0.22)
    add("syn_short_b", "S11", "D4", short_b, [(5.0, 5.3)], "0.3 s impact")
    mixed_b = switch.copy()
    mixed_b += _burst(rng, n, sr, 3.0, 2.0, 0.17)
    add("syn_mixed_b", "S12", "D4", mixed_b, [(3.0, 5.0)], "background switch + 2 s anomaly")

    write_manifest(records, out / "manifest.csv")
    write_annotations(anns, out / "annotations.csv")
    return {
        "out": str(out), "files": len(records), "annotations": len(anns),
        "duration_s": dur, "sr": sr, "seed": seed,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    print(generate(args.out, args.seed))
