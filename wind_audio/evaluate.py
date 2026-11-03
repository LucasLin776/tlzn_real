"""Event-level evaluation with one-to-one matching.

Definitions used here:

* Only reviewed regions count.  A prediction that touches only unknown audio
  is excluded, never counted as a false positive or a true positive.
* GT events are the interval-level ``anomaly`` annotations (merged when they
  touch).  File-level coarse labels never become GT.
* Matching is greedy by IoU, one prediction per GT event.  A huge prediction
  covering the whole file has low IoU with every GT event and cannot "hit all
  of them".
* ``fa_per_hour`` counts unmatched predictions that overlap reviewed normal
  audio; its denominator is reviewed normal hours, not wall-clock time.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .audio import load_wav
from .data import (
    AudioRecord,
    Annotation,
    anomaly_events,
    label_mask,
    regions_from_mask,
    resolve_audio_path,
)
from .features import FeatureConfig, extract_features


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def iou(a: dict, b: dict) -> float:
    inter = _overlap(a["start_s"], a["end_s"], b["start_s"], b["end_s"])
    union = (a["end_s"] - a["start_s"]) + (b["end_s"] - b["start_s"]) - inter
    return inter / union if union > 0 else 0.0


def match_events(pred: list[dict], gt: list[dict], iou_threshold: float = 0.1) -> dict:
    pairs = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            v = iou(p, g)
            if v >= iou_threshold:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches = []
    for v, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matches.append({"pred_index": i, "gt_index": j, "iou": float(v)})
    return {
        "matches": matches,
        "unmatched_pred": [i for i in range(len(pred)) if i not in used_p],
        "unmatched_gt": [j for j in range(len(gt)) if j not in used_g],
    }


def evaluate_file_events(
    record: AudioRecord,
    anns: list[Annotation],
    pred_events: list[dict],
    iou_threshold: float = 0.1,
) -> dict:
    gt = anomaly_events(record, anns)
    mask = label_mask(record, anns, 0.1)
    regions = regions_from_mask(mask)
    normal_s = sum(r["end_s"] - r["start_s"] for r in regions if r["label"] == "normal")
    unknown_s = sum(r["end_s"] - r["start_s"] for r in regions if r["label"] == "uncertain")
    anomaly_s = sum(r["end_s"] - r["start_s"] for r in regions if r["label"] == "anomaly")
    reviewed_s = normal_s + anomaly_s

    reviewed: list[dict] = []
    excluded_unknown = 0
    for p in pred_events:
        a = max(0.0, float(p["start_s"]))
        b = min(record.duration_s, float(p["end_s"]))
        if b <= a:
            continue
        ov_n = sum(_overlap(a, b, r["start_s"], r["end_s"]) for r in regions if r["label"] == "normal")
        ov_a = sum(_overlap(a, b, r["start_s"], r["end_s"]) for r in regions if r["label"] == "anomaly")
        ov_u = sum(_overlap(a, b, r["start_s"], r["end_s"]) for r in regions if r["label"] == "uncertain")
        if ov_n + ov_a <= 1e-9 and ov_u > 1e-9:
            excluded_unknown += 1
            continue
        q = dict(p)
        q.update({"start_s": a, "end_s": b, "_ov_n": ov_n, "_ov_a": ov_a})
        reviewed.append(q)

    match = match_events(reviewed, gt, iou_threshold)
    matched_pred = {m["pred_index"] for m in match["matches"]}
    matched_gt = {m["gt_index"] for m in match["matches"]}
    fp_events = 0
    fp_normal_s = 0.0
    for i, p in enumerate(reviewed):
        if i in matched_pred:
            continue
        if p["_ov_n"] > 1e-9:
            fp_events += 1
            fp_normal_s += float(p["_ov_n"])

    gt_duration = sum(g["end_s"] - g["start_s"] for g in gt)
    inter = 0.0
    start_err = []
    end_err = []
    for m in match["matches"]:
        p = reviewed[m["pred_index"]]
        g = gt[m["gt_index"]]
        inter += _overlap(p["start_s"], p["end_s"], g["start_s"], g["end_s"])
        start_err.append(abs(p["start_s"] - g["start_s"]))
        end_err.append(abs(p["end_s"] - g["end_s"]))

    alarm_s = sum(p["end_s"] - p["start_s"] for p in reviewed)
    return {
        "audio_id": record.audio_id,
        "device_id": record.device_id,
        "source_id": record.source_id,
        "duration_s": record.duration_s,
        "normal_s": normal_s,
        "anomaly_s": anomaly_s,
        "unknown_s": unknown_s,
        "reviewed_s": reviewed_s,
        "n_gt_events": len(gt),
        "n_pred_events": len(pred_events),
        "n_pred_reviewed": len(reviewed),
        "n_matched": len(match["matches"]),
        "n_fp_events": fp_events,
        "excluded_unknown_events": excluded_unknown,
        "fp_normal_s": fp_normal_s,
        "alarm_s": alarm_s,
        "coverage_inter_s": inter,
        "gt_duration_s": gt_duration,
        "boundary_start_errors_s": start_err,
        "boundary_end_errors_s": end_err,
    }


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def aggregate_file_metrics(rows: list[dict]) -> dict:
    normal_s = sum(r["normal_s"] for r in rows)
    anomaly_s = sum(r["anomaly_s"] for r in rows)
    unknown_s = sum(r["unknown_s"] for r in rows)
    reviewed_s = normal_s + anomaly_s
    n_gt = sum(r["n_gt_events"] for r in rows)
    n_pred = sum(r["n_pred_events"] for r in rows)
    n_pred_reviewed = sum(r["n_pred_reviewed"] for r in rows)
    n_matched = sum(r["n_matched"] for r in rows)
    n_fp = sum(r["n_fp_events"] for r in rows)
    fp_normal_s = sum(r["fp_normal_s"] for r in rows)
    alarm_s = sum(r["alarm_s"] for r in rows)
    inter = sum(r["coverage_inter_s"] for r in rows)
    gt_duration = sum(r["gt_duration_s"] for r in rows)
    start_err = [v for r in rows for v in r["boundary_start_errors_s"]]
    end_err = [v for r in rows for v in r["boundary_end_errors_s"]]
    normal_hours = normal_s / 3600.0
    devices = {r["device_id"] for r in rows if r.get("device_id")}
    sources = {r["source_id"] for r in rows if r.get("source_id")}
    return {
        "files": len(rows),
        "devices": len(devices),
        "sources": len(sources),
        "reviewed_normal_s": round(normal_s, 3),
        "reviewed_anomaly_s": round(anomaly_s, 3),
        "unknown_s": round(unknown_s, 3),
        "reviewed_s": round(reviewed_s, 3),
        "n_gt_events": n_gt,
        "n_pred_events": n_pred,
        "n_pred_reviewed": n_pred_reviewed,
        "n_matched": n_matched,
        "n_fp_events": n_fp,
        "event_recall": (n_matched / n_gt) if n_gt else None,
        "event_precision": (n_matched / n_pred_reviewed) if n_pred_reviewed else None,
        "fa_per_hour": (n_fp / normal_hours) if normal_hours > 0 else None,
        "fp_alarm_s_per_hour": (fp_normal_s / normal_hours) if normal_hours > 0 else None,
        "coverage": (inter / gt_duration) if gt_duration > 0 else None,
        "alarm_ratio": (alarm_s / reviewed_s) if reviewed_s > 0 else None,
        "boundary_start_median_s": _median(start_err),
        "boundary_end_median_s": _median(end_err),
        "excluded_unknown_events": sum(r["excluded_unknown_events"] for r in rows),
        "normal_hours": round(normal_hours, 4),
    }


def evaluate_detector(
    detector,
    records: list[AudioRecord],
    anns: list[Annotation],
    audio_ids: list[str],
    manifest_path: str | Path,
    cfg: FeatureConfig,
    *,
    threshold: float | None = None,
    postprocess: dict | None = None,
    iou_threshold: float = 0.1,
) -> dict:
    by_id = {r.audio_id: r for r in records}
    rows = []
    for audio_id in audio_ids:
        rec = by_id[audio_id]
        x, sr = load_wav(resolve_audio_path(rec, manifest_path), target_sr=cfg.sr)
        bundle = extract_features(x, sr, cfg, audio_id)
        pred = detector.predict_events(bundle, threshold=threshold, postprocess=postprocess)
        rows.append(evaluate_file_events(rec, anns, pred, iou_threshold))
    overall = aggregate_file_metrics(rows)
    overall["threshold"] = float(detector.threshold if threshold is None else threshold)
    overall["postprocess"] = dict(detector.postprocess if postprocess is None else postprocess)
    overall["iou_threshold"] = iou_threshold
    overall["per_file"] = rows
    return overall


def _null_safe(obj):
    if isinstance(obj, dict):
        return {k: _null_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_null_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def write_eval_outputs(result: dict, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = _null_safe(result)
    (out / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 评估简报",
        "",
        f"- 文件数：{result['files']}，设备数：{result['devices']}，来源数：{result['sources']}",
        f"- 已审核正常：{result['reviewed_normal_s']:.1f}s；已审核异常：{result['reviewed_anomaly_s']:.1f}s；未审核：{result['unknown_s']:.1f}s",
        f"- GT 事件：{result['n_gt_events']}；预测事件：{result['n_pred_events']}（其中落在已审核区 {result['n_pred_reviewed']}）",
        f"- 事件召回：{_fmt(result['event_recall'])}；事件精确率：{_fmt(result['event_precision'])}",
        f"- 每小时误报：{_fmt(result['fa_per_hour'])} 次；每小时误报时长：{_fmt(result['fp_alarm_s_per_hour'])} s",
        f"- 异常覆盖率：{_fmt(result['coverage'])}；报警时长占比：{_fmt(result['alarm_ratio'])}",
        f"- 边界误差（中位）：起点 {_fmt(result['boundary_start_median_s'])}s，终点 {_fmt(result['boundary_end_median_s'])}s",
        f"- 只落在未审核区、被排除的预测：{result['excluded_unknown_events']}",
        "",
        "> 未审核区域不计入召回、误报或正常时长；文件级粗标不参与评估。",
    ]
    (out / "brief.md").write_text("\n".join(lines), encoding="utf-8")
    return out / "metrics.json", out / "brief.md"


def _fmt(v) -> str:
    if v is None:
        return "不可计算（无对应样本）"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)
