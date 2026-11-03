"""Threshold + event-postprocessing calibration on the validation split only."""
from __future__ import annotations

import itertools
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .audio import load_wav
from .data import AudioRecord, Annotation, resolve_audio_path
from .evaluate import aggregate_file_metrics, evaluate_file_events
from .events import extract_events
from .features import extract_features


def _candidate_thresholds(scores: np.ndarray, score_type: str) -> list[float]:
    if len(scores) == 0:
        return []
    if score_type == "baseline_z":
        hi = max(20.0, float(np.percentile(scores, 99.9)))
        vals = np.linspace(2.0, hi, 40)
    else:
        qs = np.quantile(scores, [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 0.995, 0.999])
        vals = np.unique(np.concatenate([qs, np.linspace(0.05, 0.95, 30)]))
    # Add midpoints between score clusters.  Without these, an over-confident
    # classifier with scores {0.0005, 0.9995} can pick a threshold just below
    # the low plateau, so normal frames never fall below the hysteresis exit.
    uniq = np.unique(scores)
    if len(uniq) > 1:
        mids = (uniq[:-1] + uniq[1:]) / 2.0
        if len(mids) > 300:
            mids = np.quantile(mids, np.linspace(0.0, 1.0, 300))
        eps = np.maximum(np.abs(uniq) * 1e-3, 1e-9)
        vals = np.unique(np.concatenate([vals, mids, uniq + eps, np.nextafter(uniq, np.inf)]))
    return [float(v) for v in vals if np.isfinite(v) and v > 0]


def calibrate_detector(
    detector,
    records: list[AudioRecord],
    anns: list[Annotation],
    audio_ids: list[str],
    manifest_path: str | Path,
    *,
    target_recall: float = 0.95,
    iou_threshold: float = 0.1,
    min_durations: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5),
    merge_gaps: tuple[float, ...] = (0.1, 0.3, 0.5),
    low_ratios: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9),
) -> dict:
    by_id = {r.audio_id: r for r in records}
    bundles = []
    for audio_id in audio_ids:
        rec = by_id[audio_id]
        x, sr = load_wav(resolve_audio_path(rec, manifest_path), target_sr=detector.feature_config.sr)
        bundles.append((rec, extract_features(x, sr, detector.feature_config, audio_id)))
    if not bundles:
        raise ValueError("验证集为空，无法校准")
    scores_by_file = [detector.score(b) for _, b in bundles]
    all_scores = np.concatenate(scores_by_file) if scores_by_file else np.array([])
    thresholds = _candidate_thresholds(all_scores, detector.score_type)
    if not thresholds:
        raise ValueError("没有可用分数，无法校准")

    postprocess_grid = []
    for md, mg, lr in itertools.product(min_durations, merge_gaps, low_ratios):
        postprocess_grid.append({"min_duration": float(md), "merge_gap": float(mg), "low_ratio": float(lr)})

    rows: list[dict] = []
    for threshold in thresholds:
        for pp in postprocess_grid:
            per_file = []
            for (rec, bundle), score in zip(bundles, scores_by_file):
                pred = extract_events(
                    score,
                    bundle.times,
                    threshold=threshold,
                    threshold_low=threshold * pp["low_ratio"],
                    min_duration=pp["min_duration"],
                    merge_gap=pp["merge_gap"],
                    duration_s=bundle.duration_s,
                    step_s=bundle.config.target_step,
                )
                per_file.append(evaluate_file_events(rec, anns, pred, iou_threshold))
            m = aggregate_file_metrics(per_file)
            recall = m["event_recall"]
            if recall is None:
                continue
            fa = m["fa_per_hour"]
            precision = m["event_precision"] or 0.0
            rows.append(
                {
                    "threshold": float(threshold),
                    "postprocess": pp,
                    "recall": float(recall),
                    "precision": float(precision),
                    "fa_per_hour": None if fa is None else float(fa),
                    "coverage": m["coverage"],
                }
            )

    if not rows:
        raise ValueError("验证集没有可评估的异常事件，无法按目标召回校准")

    feasible = [r for r in rows if r["recall"] >= target_recall]
    target_reached = bool(feasible)
    pool = feasible if feasible else rows
    pool.sort(
        key=lambda r: (
            (r["fa_per_hour"] if r["fa_per_hour"] is not None else 1e18),
            -r["precision"],
            -r["recall"],
            -r["threshold"],
        )
    )
    best = pool[0]
    result = {
        "target_recall": float(target_recall),
        "target_reached": target_reached,
        "iou_threshold": float(iou_threshold),
        "split": "val",
        "audio_ids": list(audio_ids),
        "candidates_evaluated": len(rows),
        "selected": {
            "threshold": best["threshold"],
            "postprocess": best["postprocess"],
        },
        "validation": {
            "event_recall": best["recall"],
            "event_precision": best["precision"],
            "fa_per_hour": best["fa_per_hour"],
            "coverage": best["coverage"],
        },
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return result


def apply_calibration(detector, calibration: dict) -> None:
    detector.threshold = float(calibration["selected"]["threshold"])
    detector.postprocess = dict(calibration["selected"]["postprocess"])
    detector.calibrated = True
    detector.metadata["calibration"] = calibration
