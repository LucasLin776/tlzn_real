"""L1 detectors with one scoring/save/load interface.

* ``BaselineDetector`` keeps the original median/MAD ``NormalModel`` and can
  load legacy ``models/l1_demo`` directories.
* ``SupervisedDetector`` is the NumPy logistic-regression control.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .audio import load_wav, log_mel
from .backends import (
    BackendSelection,
    NumpyLogisticBackend,
    backend_versions,
    select_backend,
)
from .data import AudioRecord, Annotation, resolve_audio_path
from .events import event_features, extract_events
from .features import FeatureBundle, FeatureConfig, labels_at_times
from .l1 import NormalModel

DEFAULT_POSTPROCESS = {"min_duration": 0.2, "merge_gap": 0.3, "low_ratio": 0.6}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def _merge_postprocess(base: dict | None, override: dict | None) -> dict:
    out = dict(DEFAULT_POSTPROCESS)
    if base:
        out.update({k: v for k, v in base.items() if v is not None})
    if override:
        out.update({k: v for k, v in override.items() if v is not None})
    return out


class BaseDetector:
    detector_type = "base"
    score_type = "score"

    def __init__(
        self,
        *,
        version: str,
        feature_config: FeatureConfig,
        threshold: float,
        postprocess: dict | None = None,
        calibrated: bool = False,
        demo: bool = False,
        metadata: dict | None = None,
    ):
        self.version = version
        self.feature_config = feature_config
        self.threshold = float(threshold)
        self.postprocess = _merge_postprocess(postprocess, None)
        self.calibrated = bool(calibrated)
        self.demo = bool(demo)
        self.metadata = metadata or {}

    def score(self, bundle: FeatureBundle) -> np.ndarray:
        raise NotImplementedError

    def predict_events(
        self,
        bundle: FeatureBundle,
        threshold: float | None = None,
        postprocess: dict | None = None,
    ) -> list[dict]:
        score = self.score(bundle)
        thr = float(self.threshold if threshold is None else threshold)
        pp = _merge_postprocess(self.postprocess, postprocess)
        events = extract_events(
            score,
            bundle.times,
            threshold=thr,
            threshold_low=thr * float(pp["low_ratio"]),
            min_duration=float(pp["min_duration"]),
            merge_gap=float(pp["merge_gap"]),
            duration_s=bundle.duration_s,
            step_s=bundle.config.target_step,
        )
        out = []
        for i, ev in enumerate(events, 1):
            feats = event_features(bundle, ev, score, self.score_type)
            feats.update(
                {
                    "event_id": f"{bundle.audio_id}_ev{i:03d}",
                    "start_s": round(ev["start_s"], 4),
                    "end_s": round(ev["end_s"], 4),
                    "duration_s": round(ev["duration_s"], 4),
                    "peak_score": float(ev["peak_score"]),
                    "mean_score": float(ev["mean_score"]),
                    "score_type": self.score_type,
                    "threshold": thr,
                }
            )
            if self.score_type == "baseline_z":
                feats["peak_z"] = float(ev["peak_score"])
            out.append(feats)
        return out

    def save(self, out_dir: str | Path) -> Path:
        raise NotImplementedError


class BaselineDetector(BaseDetector):
    detector_type = "baseline"
    score_type = "baseline_z"

    def __init__(self, model: NormalModel, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    # ------------------------------------------------------------------ train
    @classmethod
    def train(
        cls,
        records: list[AudioRecord],
        anns: list[Annotation],
        audio_ids: list[str],
        manifest_path: str | Path,
        cfg: FeatureConfig,
        *,
        version: str = "l1_baseline",
        threshold: float = 5.0,
        max_frames_per_file: int = 20000,
        seed: int = 42,
        demo: bool = False,
    ) -> "BaselineDetector":
        by_id = {r.audio_id: r for r in records}
        rng = np.random.default_rng(seed)
        chunks = []
        used = []
        for audio_id in audio_ids:
            rec = by_id[audio_id]
            x, sr = load_wav(resolve_audio_path(rec, manifest_path), target_sr=cfg.sr)
            mel_db, _, mel_times = log_mel(
                x, sr, cfg.n_fft, cfg.hop, cfg.n_mels, cfg.fmin, cfg.fmax
            )
            if cfg.normalize == "per_file":
                mel = mel_db - np.median(mel_db, axis=0)
            else:
                mel = mel_db
            y = labels_at_times(rec, anns, mel_times, cfg.hop / cfg.sr)
            normal = mel[y == 0]
            if len(normal) == 0:
                continue
            if max_frames_per_file > 0 and len(normal) > max_frames_per_file:
                pick = rng.choice(len(normal), size=max_frames_per_file, replace=False)
                normal = normal[np.sort(pick)]
            chunks.append(normal)
            used.append({"audio_id": audio_id, "normal_frames": int(len(normal))})
        if not chunks:
            raise ValueError("没有已审核正常帧，无法训练 baseline；请先提供 normal 区间")
        mel = np.concatenate(chunks, axis=0)
        if len(mel) < 100:
            raise ValueError(f"已审核正常帧只有 {len(mel)}，至少需要 100 帧")
        model = NormalModel(
            threshold_z=threshold,
            version=version,
            config={**cfg.to_dict(), "demo": demo},
        )
        model.fit(mel)
        return cls(
            model,
            version=version,
            feature_config=cfg,
            threshold=threshold,
            postprocess=dict(DEFAULT_POSTPROCESS),
            calibrated=False,
            demo=demo,
            metadata={
                "training": {
                    "audio_ids": audio_ids,
                    "files": used,
                    "normal_frames": int(len(mel)),
                    "seed": seed,
                },
                "dependencies": backend_versions(),
            },
        )

    # ------------------------------------------------------------------ score
    def score(self, bundle: FeatureBundle) -> np.ndarray:
        self.feature_config.assert_compatible(bundle.config)
        mel = bundle.mel_norm if self.feature_config.normalize == "per_file" else bundle.mel_db
        frame_score = self.model.score(mel)
        if len(frame_score) == 0:
            return np.zeros(len(bundle.times), dtype=np.float32)
        return np.interp(bundle.times, bundle.mel_times, frame_score).astype(np.float32)

    # ------------------------------------------------------------- save/load
    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(
            out / "model.npz",
            mu=self.model.mu,
            sigma=self.model.sigma,
            score_mu=self.model.score_mu,
            score_sigma=self.model.score_sigma,
        )
        meta = {
            "schema_version": "2.0",
            "version": self.version,
            "detector": self.detector_type,
            "score_type": self.score_type,
            "created_at": _now(),
            "demo": self.demo,
            "calibrated": self.calibrated,
            "threshold": self.threshold,
            "postprocess": self.postprocess,
            "feature_config": self.feature_config.to_dict(),
            "feature_fingerprint": self.feature_config.fingerprint(),
            "model": {"score_mu": self.model.score_mu, "score_sigma": self.model.score_sigma},
            **self.metadata,
        }
        meta.update(
            version=self.version, threshold=self.threshold,
            calibrated=self.calibrated, demo=self.demo,
        )
        (out / "meta.json").write_text(
            json.dumps(_jsonable(meta), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return out

    @classmethod
    def load(cls, model_dir: str | Path) -> "BaselineDetector":
        d = Path(model_dir)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        legacy = "detector" not in meta
        if legacy:
            model = NormalModel.load(d)
            cfg = FeatureConfig(
                sr=int(meta.get("sr", 44100)),
                n_fft=int(meta.get("n_fft", 2048)),
                hop=int(meta.get("hop", 441)),
                n_mels=int(meta.get("n_mels", 128)),
                fmin=float(meta.get("fmin", 20.0)),
                fmax=float(meta.get("fmax", 16000.0)),
                target_step=0.1,
                context_s=1.0,
                normalize="none",
            )
            return cls(
                model,
                version=str(meta.get("version", "legacy")),
                feature_config=cfg,
                threshold=float(meta.get("threshold_z", 5.0)),
                postprocess=dict(DEFAULT_POSTPROCESS),
                calibrated=False,
                demo=bool(meta.get("demo", False)),
                metadata={"legacy": True, "raw_meta": meta},
            )
        z = np.load(d / "model.npz")
        model = NormalModel(
            mu=z["mu"], sigma=z["sigma"],
            score_mu=float(z["score_mu"]), score_sigma=float(z["score_sigma"]),
            threshold_z=float(meta.get("threshold", 5.0)),
            version=str(meta.get("version", "baseline")),
            config=dict(meta.get("feature_config", {})),
        )
        cfg = FeatureConfig.from_dict(meta["feature_config"])
        return cls(
            model,
            version=str(meta.get("version", "baseline")),
            feature_config=cfg,
            threshold=float(meta.get("threshold", 5.0)),
            postprocess=dict(meta.get("postprocess", DEFAULT_POSTPROCESS)),
            calibrated=bool(meta.get("calibrated", False)),
            demo=bool(meta.get("demo", False)),
            metadata={k: v for k, v in meta.items() if k not in {"feature_config", "postprocess"}},
        )


class SupervisedDetector(BaseDetector):
    detector_type = "supervised"
    score_type = "probability_uncalibrated"

    def __init__(self, backend: Any, feature_names: list[str], **kwargs):
        super().__init__(**kwargs)
        self.backend = backend
        self.feature_names = list(feature_names)

    @classmethod
    def train(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        cfg: FeatureConfig,
        *,
        version: str = "l1_supervised",
        seed: int = 42,
        algo: str = "numpy",
        allow_fallback: bool = False,
        metadata: dict | None = None,
    ) -> tuple["SupervisedDetector", BackendSelection]:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int8)
        if len(X) < 50:
            raise ValueError(f"训练样本只有 {len(X)}，至少需要 50")
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        if pos < 10 or neg < 10:
            raise ValueError(f"类别不足：normal={neg}, anomaly={pos}，各至少需要 10")
        selection = select_backend(algo, allow_fallback=allow_fallback, seed=seed)
        w = make_sample_weights(y, np.array([str(v) for v in metadata.get("file_ids", [])]) if metadata and metadata.get("file_ids") else None)
        selection.backend.fit(X, y, sample_weight=w)
        detector = cls(
            selection.backend,
            feature_names,
            version=version,
            feature_config=cfg,
            threshold=0.5,
            postprocess=dict(DEFAULT_POSTPROCESS),
            calibrated=False,
            demo=False,
            metadata={
                **(metadata or {}),
                "backend": {
                    "requested": selection.requested,
                    "used": selection.used,
                    "fallback": selection.fallback,
                    "reason": selection.reason,
                    "versions": backend_versions(),
                },
                "training": {
                    **(metadata or {}).get("training", {}),
                    "samples": int(len(X)),
                    "normal_samples": neg,
                    "anomaly_samples": pos,
                    "seed": seed,
                },
            },
        )
        return detector, selection

    def score(self, bundle: FeatureBundle) -> np.ndarray:
        self.feature_config.assert_compatible(bundle.config)
        if list(bundle.feature_names) != self.feature_names:
            raise ValueError("特征名称/顺序与模型不一致，拒绝推理")
        return self.backend.predict_proba(bundle.X)[:, 1].astype(np.float32)

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.backend.save(out / "model")
        meta = {
            "schema_version": "2.0",
            "version": self.version,
            "detector": self.detector_type,
            "score_type": self.score_type,
            "created_at": _now(),
            "demo": self.demo,
            "calibrated": self.calibrated,
            "threshold": self.threshold,
            "postprocess": self.postprocess,
            "feature_config": self.feature_config.to_dict(),
            "feature_fingerprint": self.feature_config.fingerprint(),
            "feature_names": self.feature_names,
            "backend": self.metadata.get("backend", {"used": getattr(self.backend, "name", "unknown")}),
            **{k: v for k, v in self.metadata.items() if k != "backend"},
        }
        meta.update(
            version=self.version, threshold=self.threshold,
            calibrated=self.calibrated, demo=self.demo,
        )
        (out / "meta.json").write_text(
            json.dumps(_jsonable(meta), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return out

    @classmethod
    def load(cls, model_dir: str | Path) -> "SupervisedDetector":
        d = Path(model_dir)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        used = meta.get("backend", {}).get("used", "")
        if used in {"numpy", "numpy_logistic"}:
            backend = NumpyLogisticBackend.load(d / "model")
        else:
            raise ValueError(f"未知 backend: {used}")
        cfg = FeatureConfig.from_dict(meta["feature_config"])
        return cls(
            backend,
            meta["feature_names"],
            version=str(meta.get("version", "supervised")),
            feature_config=cfg,
            threshold=float(meta.get("threshold", 0.5)),
            postprocess=dict(meta.get("postprocess", DEFAULT_POSTPROCESS)),
            calibrated=bool(meta.get("calibrated", False)),
            demo=bool(meta.get("demo", False)),
            metadata={k: v for k, v in meta.items() if k not in {"feature_config", "postprocess", "feature_names"}},
        )


def load_detector(model_dir: str | Path) -> BaseDetector:
    d = Path(model_dir)
    meta_path = d / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"模型目录没有 meta.json: {d}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    kind = meta.get("detector", "baseline")
    if kind == "supervised":
        return SupervisedDetector.load(d)
    return BaselineDetector.load(d)


def make_sample_weights(y: np.ndarray, file_ids: np.ndarray | None) -> np.ndarray:
    """Balance classes and recordings so one long file cannot dominate."""
    y = np.asarray(y)
    w = np.ones(len(y), dtype=np.float64)
    if file_ids is not None and len(file_ids) == len(y):
        counts = Counter(str(v) for v in file_ids)
        w *= np.array([1.0 / max(counts[str(v)], 1) for v in file_ids])
    counts = Counter(int(v) for v in y)
    w *= np.array([1.0 / max(counts[int(v)], 1) for v in y])
    w /= max(w.mean(), 1e-12)
    return w.astype(np.float32)


def cap_training_data(
    X, y, groups, file_ids, times,
    max_total: int = 50000,
    max_per_file: int = 5000,
    seed: int = 42,
) -> dict:
    """Deterministic per-file + global cap, preserving positives."""
    rng = np.random.default_rng(seed)
    keep = []
    for fid in sorted(set(str(v) for v in file_ids)):
        idx = np.flatnonzero(np.array([str(v) == fid for v in file_ids]))
        if len(idx) > max_per_file > 0:
            pos = idx[y[idx] == 1]
            neg = idx[y[idx] == 0]
            take_pos = pos if len(pos) <= max_per_file else rng.choice(pos, max_per_file, replace=False)
            room = max(0, max_per_file - len(take_pos))
            take_neg = neg if len(neg) <= room else rng.choice(neg, room, replace=False)
            idx = np.sort(np.concatenate([take_pos, take_neg]).astype(int))
        keep.append(idx)
    idx = np.concatenate(keep) if keep else np.array([], dtype=int)
    if len(idx) > max_total > 0:
        pos = idx[y[idx] == 1]
        neg = idx[y[idx] == 0]
        ratio = len(pos) / max(len(idx), 1)
        n_pos = min(len(pos), max(1, int(round(max_total * ratio))))
        n_neg = min(len(neg), max(1, max_total - n_pos))
        pos = rng.choice(pos, n_pos, replace=False) if len(pos) > n_pos else pos
        neg = rng.choice(neg, n_neg, replace=False) if len(neg) > n_neg else neg
        idx = np.sort(np.concatenate([pos, neg]).astype(int))
    return {
        "X": X[idx], "y": y[idx], "groups": groups[idx],
        "file_ids": file_ids[idx], "times": times[idx],
        "selected": int(len(idx)), "available": int(len(y)),
    }
