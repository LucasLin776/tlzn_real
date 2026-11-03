"""NumPy logistic classifier for the lightweight L1 supervised control."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


class NumpyLogisticBackend:
    name = "numpy_logistic"

    def __init__(self, seed: int = 42, l2: float = 1e-3, iters: int = 250, lr: float = 0.2):
        self.seed = seed
        self.l2 = l2
        self.iters = iters
        self.lr = lr
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    def fit(self, X, y, sample_weight=None, **_) -> "NumpyLogisticBackend":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        self.sigma = np.where(sigma > 1e-9, sigma, 1.0)
        Z = (X - self.mu) / self.sigma
        if sample_weight is None:
            w = np.ones(len(y), dtype=np.float64)
        else:
            w = np.asarray(sample_weight, dtype=np.float64)
            w = w / max(w.mean(), 1e-12)
        self.w = np.zeros(Z.shape[1], dtype=np.float64)
        pos = max(float(w[y > 0.5].sum()), 1e-6)
        neg = max(float(w[y <= 0.5].sum()), 1e-6)
        self.b = float(np.log(pos / neg))
        n = max(len(y), 1)
        for _ in range(self.iters):
            p = _sigmoid(Z @ self.w + self.b)
            grad_w = Z.T @ ((p - y) * w) / n + self.l2 * self.w
            grad_b = float(np.sum((p - y) * w) / n)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.w is None or self.mu is None or self.sigma is None:
            raise RuntimeError("backend not fitted")
        Z = (np.asarray(X, dtype=np.float64) - self.mu) / self.sigma
        p = _sigmoid(Z @ self.w + self.b)
        return np.stack([1.0 - p, p], axis=1)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.savez(
            path / "model.npz",
            mu=self.mu, sigma=self.sigma, w=self.w, b=np.array([self.b]),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "NumpyLogisticBackend":
        z = np.load(Path(path) / "model.npz")
        obj = cls()
        obj.mu, obj.sigma, obj.w, obj.b = z["mu"], z["sigma"], z["w"], float(z["b"][0])
        return obj



@dataclass
class BackendSelection:
    backend: NumpyLogisticBackend
    requested: str = "numpy"
    used: str = "numpy"
    fallback: bool = False
    reason: str = ""


def select_backend(requested="numpy", allow_fallback=False, seed=42):
    if requested != "numpy":
        raise ValueError("精简版 L1 有监督对照仅支持 numpy；不自动切换或安装后端")
    return BackendSelection(NumpyLogisticBackend(seed=seed))


def backend_versions():
    return {"numpy": np.__version__}
