"""Classifier backends: XGBoost (preferred), sklearn HistGradientBoosting
(transparent fallback), and a small numpy logistic regression (last resort).

The chosen backend is always written into the model metadata.  A fallback is
never silent: ``BackendSelection.fallback`` and ``reason`` are recorded and
printed by the CLI.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


class SklearnBackend:
    name = "sklearn_hist_gradient_boosting"

    def __init__(self, seed: int = 42, max_iter: int = 200, learning_rate: float = 0.06):
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=seed,
            early_stopping=False,
        )

    def fit(self, X, y, sample_weight=None, **_) -> "SklearnBackend":
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "model.pkl").open("wb") as fh:
            pickle.dump(self.model, fh, protocol=4)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SklearnBackend":
        obj = cls.__new__(cls)
        with (Path(path) / "model.pkl").open("rb") as fh:
            obj.model = pickle.load(fh)
        return obj


class XGBBackend:
    name = "xgboost"

    def __init__(self, seed: int = 42, n_estimators: int = 300, max_depth: int = 4):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - depends on host
            raise ImportError(
                "未安装 xgboost。请运行 `pip install xgboost`，"
                "或显式使用 --algo sklearn 并确认回退算法。"
            ) from exc
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=4,
            random_state=seed,
            eval_metric="logloss",
        )

    def fit(self, X, y, sample_weight=None, **_) -> "XGBBackend":
        self.model.fit(X, y, sample_weight=sample_weight, verbose=False)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path / "model.json"))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "XGBBackend":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover
            raise ImportError("加载该模型需要 xgboost，请先 `pip install xgboost`") from exc
        obj = cls.__new__(cls)
        obj.model = XGBClassifier()
        obj.model.load_model(str(Path(path) / "model.json"))
        return obj


@dataclass
class BackendSelection:
    backend: Any
    requested: str
    used: str
    fallback: bool
    reason: str


def _make(name: str, seed: int):
    if name == "xgboost":
        return XGBBackend(seed=seed)
    if name == "sklearn":
        return SklearnBackend(seed=seed)
    if name == "numpy":
        return NumpyLogisticBackend(seed=seed)
    raise ValueError(f"未知 backend: {name}")


def select_backend(requested: str = "auto", allow_fallback: bool = True, seed: int = 42) -> BackendSelection:
    requested = (requested or "auto").lower()
    order = {
        "auto": ["xgboost", "sklearn", "numpy"],
        "xgboost": ["xgboost", "sklearn", "numpy"],
        "sklearn": ["sklearn", "numpy"],
        "numpy": ["numpy"],
    }
    if requested not in order:
        raise ValueError("--algo 只能是 auto / xgboost / sklearn / numpy")
    reasons = []
    for i, name in enumerate(order[requested]):
        try:
            backend = _make(name, seed)
            return BackendSelection(
                backend=backend,
                requested=requested,
                used=name,
                fallback=(name != requested and requested != "auto") or (requested == "auto" and i > 0),
                reason="; ".join(reasons) if reasons else "",
            )
        except Exception as exc:
            reasons.append(f"{name}: {exc}")
            if not allow_fallback:
                break
    raise RuntimeError(
        "没有可用的分类后端。\n" + "\n".join(reasons) +
        "\n安装建议：`pip install xgboost` 或 `pip install scikit-learn`。"
    )


def backend_versions() -> dict:
    out = {"numpy": np.__version__}
    try:
        import sklearn

        out["sklearn"] = sklearn.__version__
    except Exception:
        out["sklearn"] = None
    try:
        import xgboost

        out["xgboost"] = xgboost.__version__
    except Exception:
        out["xgboost"] = None
    return out
