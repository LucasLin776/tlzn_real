"""Layer-1 normal/abnormal scorer.

The model is fitted on NORMAL audio only: for every mel band it learns a
robust centre (median) and scale (MAD), then scores each 10 ms frame by how
far its spectral shape sits from that baseline.  A short median filter removes
single-frame noise before thresholding.

This is deliberately simple and data-efficient.  It is the v0 baseline; once
enough normal data is available it can be swapped for an autoencoder /
self-supervised embedding model behind the same interface.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .audio import median_filter


@dataclass
class NormalModel:
    mu: np.ndarray | None = None
    sigma: np.ndarray | None = None
    score_mu: float = 0.0
    score_sigma: float = 1.0
    threshold_z: float = 5.0
    version: str = "l1_v0"
    config: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        mel_db: np.ndarray,
        mask: np.ndarray | None = None,
        clip: float = 8.0,
    ) -> "NormalModel":
        X = mel_db if mask is None else mel_db[mask]
        if X.shape[0] < 10:
            raise ValueError(f"need at least 10 training frames, got {X.shape[0]}")

        self.mu = np.median(X, axis=0).astype(np.float32)
        mad = np.median(np.abs(X - self.mu), axis=0)
        self.sigma = np.maximum(1.4826 * mad, 0.5).astype(np.float32)

        raw = self._raw_score(X, clip)
        smooth = int(self.config.get("smooth_frames", 15))
        raw = median_filter(raw, smooth)
        self.score_mu = float(np.median(raw))
        self.score_sigma = float(max(1.4826 * np.median(np.abs(raw - self.score_mu)), 1e-3))
        return self

    # ---------------------------------------------------------------- score
    def _raw_score(self, mel_db: np.ndarray, clip: float = 8.0) -> np.ndarray:
        if self.mu is None or self.sigma is None:
            raise RuntimeError("model is not fitted")
        z = np.abs((mel_db - self.mu) / self.sigma)
        z = np.clip(z, 0.0, clip)
        k = max(4, min(z.shape[1], int(round(0.1 * z.shape[1]))))
        topk = np.partition(z, -k, axis=1)[:, -k:].mean(axis=1)
        return (0.5 * z.mean(axis=1) + 0.5 * topk).astype(np.float32)

    def score(self, mel_db: np.ndarray, clip: float = 8.0) -> np.ndarray:
        """Return per-frame anomaly z-score (smoothed)."""
        raw = self._raw_score(mel_db, clip)
        raw = median_filter(raw, int(self.config.get("smooth_frames", 15)))
        return ((raw - self.score_mu) / self.score_sigma).astype(np.float32)

    # ------------------------------------------------------------- persist
    def save(self, out_dir: str | Path, demo: bool = False, extra: dict | None = None) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if self.mu is None or self.sigma is None:
            raise RuntimeError("refusing to save an unfitted model")
        np.savez(
            out / "model.npz",
            mu=self.mu,
            sigma=self.sigma,
            score_mu=self.score_mu,
            score_sigma=self.score_sigma,
            threshold_z=self.threshold_z,
        )
        meta = {
            "version": self.version,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "demo": bool(demo),
            "n_mels": int(self.mu.shape[0]),
            "threshold_z": float(self.threshold_z),
            **self.config,
        }
        if extra:
            meta.update(extra)
        (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, model_dir: str | Path) -> "NormalModel":
        d = Path(model_dir)
        z = np.load(d / "model.npz")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        known = {"version", "created_at", "demo", "n_mels", "threshold_z"}
        config = {k: v for k, v in meta.items() if k not in known}
        config["demo"] = bool(meta.get("demo", False))
        config.setdefault("n_mels", int(meta.get("n_mels", 128)))
        config.setdefault("sr", 44100)
        return cls(
            mu=z["mu"].astype(np.float32),
            sigma=z["sigma"].astype(np.float32),
            score_mu=float(z["score_mu"]),
            score_sigma=float(z["score_sigma"]),
            threshold_z=float(z["threshold_z"]),
            version=str(meta.get("version", "l1_v0")),
            config=config,
        )
