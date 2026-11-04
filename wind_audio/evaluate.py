"""Event-level evaluation with one-to-one matching.

Definitions used here:

* Only reviewed regions count.  A prediction that touches only unknown audio
  is excluded, never counted as a false positive or a true positive.
* GT events are the interval-level ``anomaly`` annotations (merged when they
  touch). File-level coarse labels count only toward file accuracy.
* Matching is greedy by IoU, one prediction per GT event.  A huge prediction
  covering the whole file has low IoU with every GT event and cannot "hit all
  of them".
* ``fa_per_hour`` counts unmatched predictions that overlap reviewed normal
  audio; its denominator is reviewed normal hours, not wall-clock time.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .audio import load_wav
from .data import (
    AudioRecord,
    Annotation,
    anomaly_events,
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


def _reviewed_regions(record, anns):
    """Continuous annotation boundaries: do not quantise short events to 100 ms."""
    mine = [a for a in anns if a.audio_id == record.audio_id]
    intervals = [a for a in mine if a.precision == "interval"]
    points = sorted({0.0, record.duration_s} |
                    {max(0.0, min(record.duration_s, t)) for a in intervals for t in (a.start_s, a.end_s)})
    infer_normal = record.fully_reviewed and not any(
        a.precision == "file" and a.label in {"anomaly", "uncertain"} for a in mine)
    regions = []
    for start, end in zip(points, points[1:]):
        mid = (start + end) / 2
        labels = {a.label for a in intervals if a.start_s <= mid < a.end_s}
        label = (next(iter(labels)) if len(labels) == 1 else
                 "normal" if not labels and infer_normal else "uncertain")
        regions.append({"start_s": start, "end_s": end, "label": label})
    return regions


def binary_metrics(tp, tn, fp, fn):
    """Counts for files, seconds for time; undefined denominators stay null."""
    total = tp + tn + fp + fn
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "support": total,
        "accuracy": (tp + tn) / total if total else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": recall, "specificity": specificity,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "balanced_accuracy": (recall + specificity) / 2
        if recall is not None and specificity is not None else None,
    }


def _file_label(record, anns, regions):
    """An observed anomaly proves presence; normal requires complete coverage."""
    mine = [a for a in anns if a.audio_id == record.audio_id]
    coarse = {a.label for a in mine if a.precision == "file" and a.label != "uncertain"}
    if len(coarse) > 1 or ("normal" in coarse and any(a.label == "anomaly" for a in mine)):
        raise ValueError(f"{record.audio_id}: 文件标签冲突")
    if coarse:
        return next(iter(coarse))
    if any(r["label"] == "anomaly" for r in regions):
        return "anomaly"
    if regions and all(r["label"] == "normal" for r in regions):
        return "normal"
    return None


def evaluate_file_events(record, anns, pred_events, iou_threshold=0.1):
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU 门槛需要在 (0,1] 内")
    gt = anomaly_events(record, anns)
    regions = _reviewed_regions(record, anns)
    normal_s = sum(r["end_s"] - r["start_s"] for r in regions if r["label"] == "normal")
    anomaly_s = sum(r["end_s"] - r["start_s"] for r in regions if r["label"] == "anomaly")
    reviewed_s = normal_s + anomaly_s
    reviewed, excluded = [], 0
    for pred_index, p in enumerate(pred_events):
        a, b = max(0.0, p["start_s"]), min(record.duration_s, p["end_s"])
        if b <= a:
            continue
        ov_n = sum(_overlap(a, b, r["start_s"], r["end_s"]) for r in regions if r["label"] == "normal")
        ov_a = sum(_overlap(a, b, r["start_s"], r["end_s"]) for r in regions if r["label"] == "anomaly")
        if ov_n + ov_a <= 1e-12:
            excluded += 1
            continue
        reviewed.append({**p, "start_s": a, "end_s": b, "_ov_n": ov_n,
                         "_reviewed": ov_n + ov_a, "_pred_index": pred_index})
    # IoU uses only reviewed prediction duration; unknown is neither TP nor FP.
    pairs = []
    for i, p in enumerate(reviewed):
        for j, g in enumerate(gt):
            overlap = _overlap(p["start_s"], p["end_s"], g["start_s"], g["end_s"])
            union = p["_reviewed"] + g["end_s"] - g["start_s"] - overlap
            value = overlap / union if union > 0 else 0.0
            if value >= iou_threshold:
                pairs.append((value, i, j))
    used_p, used_g, matches, match_details = set(), set(), [], []
    for value, i, j in sorted(pairs, reverse=True):
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            matches.append((i, j))
            match_details.append({"pred_index": reviewed[i]["_pred_index"], "gt_index": j, "iou": value})
    # Alarm time uses the UNION of windows and includes normal overlap even for matched events.
    merged = []
    for p in sorted(reviewed, key=lambda r: r["start_s"]):
        if merged and p["start_s"] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], p["end_s"])
        else:
            merged.append([p["start_s"], p["end_s"]])
    alarm_s = sum(_overlap(a, b, r["start_s"], r["end_s"]) for a, b in merged
                  for r in regions if r["label"] in {"normal", "anomaly"})
    normal_alarm = sum(_overlap(a, b, r["start_s"], r["end_s"]) for a, b in merged
                       for r in regions if r["label"] == "normal")
    tp_s = max(0.0, min(anomaly_s, alarm_s - normal_alarm))
    fp_s = max(0.0, min(normal_s, normal_alarm))
    time = binary_metrics(tp_s, max(0.0, normal_s - fp_s), fp_s, max(0.0, anomaly_s - tp_s))
    truth = _file_label(record, anns, regions)
    predicted = "anomaly" if pred_events else "normal"
    outcome = ("excluded" if truth is None else
               "TP" if truth == predicted == "anomaly" else
               "TN" if truth == predicted == "normal" else
               "FP" if predicted == "anomaly" else "FN")
    return {
        "audio_id": record.audio_id, "device_id": record.device_id, "source_id": record.source_id,
        "duration_s": record.duration_s, "normal_s": normal_s, "anomaly_s": anomaly_s,
        "unknown_s": max(0.0, record.duration_s - reviewed_s), "reviewed_s": reviewed_s,
        "n_gt_events": len(gt), "n_pred_events": len(pred_events), "n_pred_reviewed": len(reviewed),
        "n_matched": len(matches),
        "n_fp_events": sum(i not in used_p and p["_ov_n"] > 1e-12 for i, p in enumerate(reviewed)),
        "excluded_unknown_events": excluded, "fp_normal_s": normal_alarm, "alarm_s": alarm_s,
        "coverage_inter_s": sum(_overlap(reviewed[i]["start_s"], reviewed[i]["end_s"],
                                         gt[j]["start_s"], gt[j]["end_s"]) for i, j in matches),
        "gt_duration_s": sum(g["end_s"] - g["start_s"] for g in gt),
        "boundary_start_errors_s": [abs(reviewed[i]["start_s"] - gt[j]["start_s"]) for i, j in matches],
        "boundary_end_errors_s": [abs(reviewed[i]["end_s"] - gt[j]["end_s"]) for i, j in matches],
        "file_true_label": truth, "file_predicted_label": predicted, "file_outcome": outcome,
        "time_metrics": time, "time_accuracy": time["accuracy"],
        "predicted_events": pred_events, "ground_truth_events": gt, "matches": match_details,
        "annotations": [a.to_row() for a in anns if a.audio_id == record.audio_id],
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
    file_metrics = binary_metrics(*(sum(r["file_outcome"] == k for r in rows) for k in ("TP", "TN", "FP", "FN")))
    file_metrics["excluded"] = sum(r["file_outcome"] == "excluded" for r in rows)
    time_metrics = binary_metrics(*(sum(r["time_metrics"][k] for r in rows) for k in ("tp", "tn", "fp", "fn")))
    return {
        "file_accuracy": file_metrics["accuracy"], "file_metrics": file_metrics,
        "time_accuracy": time_metrics["accuracy"], "time_metrics": time_metrics,
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
        "event_f1": 2 * n_matched / (n_gt + n_pred_reviewed) if n_gt + n_pred_reviewed else None,
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
    if not audio_ids:
        raise ValueError("评估集合为空")
    if len(audio_ids) != len(set(audio_ids)):
        raise ValueError("评估集合包含重复 audio_id")
    missing = set(audio_ids) - set(by_id)
    if missing:
        raise ValueError(f"评估集合引用不存在的 audio_id: {sorted(missing)}")
    rows = []
    for audio_id in audio_ids:
        rec = by_id[audio_id]
        x, sr = load_wav(resolve_audio_path(rec, manifest_path), target_sr=cfg.sr)
        bundle = extract_features(x, sr, cfg, audio_id)
        pred = detector.predict_events(bundle, threshold=threshold, postprocess=postprocess)
        row = evaluate_file_events(rec, anns, pred, iou_threshold)
        row["audio_path"] = str(resolve_audio_path(rec, manifest_path))
        rows.append(row)
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
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    fields = ["audio_id", "audio_path", "duration_s", "file_true_label", "file_predicted_label",
              "file_outcome", "time_accuracy", "n_gt_events", "n_pred_events", "n_matched",
              "n_fp_events", "reviewed_s", "unknown_s", "tp_s", "tn_s", "fp_s", "fn_s"]
    with (out / "per_file.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in result["per_file"]:
            writer.writerow({**row, **{k + "_s": row["time_metrics"][k] for k in ("tp", "tn", "fp", "fn")}})
    with (out / "events.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["audio_id", "kind", "index", "start_s", "end_s", "matched_index", "iou"])
        writer.writeheader()
        for row in result["per_file"]:
            for kind, key, own, other in (("prediction", "predicted_events", "pred_index", "gt_index"),
                                          ("label", "ground_truth_events", "gt_index", "pred_index")):
                matches = {m[own]: m for m in row["matches"]}
                for i, event in enumerate(row[key]):
                    match = matches.get(i, {})
                    writer.writerow({"audio_id": row["audio_id"], "kind": kind, "index": i,
                                     "start_s": event["start_s"], "end_s": event["end_s"],
                                     "matched_index": match.get(other), "iou": match.get("iou")})
    fm, tm = result["file_metrics"], result["time_metrics"]
    lines = [
        "# L1 评估简报",
        "",
        f"- 文件级准确率：{_percent(result['file_accuracy'])}（有任意候选窗口即预测为异常）",
        f"- 文件混淆矩阵：TP={fm['tp']}、TN={fm['tn']}、FP={fm['fp']}、FN={fm['fn']}；排除未知标签 {fm['excluded']} 条",
        f"- 文件精确率：{_percent(fm['precision'])}；召回率：{_percent(fm['recall'])}；F1：{_fmt(fm['f1'])}；平衡准确率：{_percent(fm['balanced_accuracy'])}",
        f"- 时长准确率：{_percent(result['time_accuracy'])}；TP={tm['tp']:.3f}s、TN={tm['tn']:.3f}s、FP={tm['fp']:.3f}s、FN={tm['fn']:.3f}s",
        f"- 异常时长召回率：{_percent(tm['recall'])}；时长精确率：{_percent(tm['precision'])}；时长 F1：{_fmt(tm['f1'])}",
        f"- 文件数：{result['files']}，设备数：{result['devices']}，来源数：{result['sources']}",
        f"- 已审核正常：{result['reviewed_normal_s']:.1f}s；已审核异常：{result['reviewed_anomaly_s']:.1f}s；未审核：{result['unknown_s']:.1f}s",
        f"- GT 事件：{result['n_gt_events']}；预测事件：{result['n_pred_events']}（其中落在已审核区 {result['n_pred_reviewed']}）",
        f"- 事件召回：{_fmt(result['event_recall'])}；事件精确率：{_fmt(result['event_precision'])}",
        f"- 每小时误报：{_fmt(result['fa_per_hour'])} 次；每小时误报时长：{_fmt(result['fp_alarm_s_per_hour'])} s",
        f"- 异常覆盖率：{_fmt(result['coverage'])}；报警时长占比：{_fmt(result['alarm_ratio'])}",
        f"- 边界误差（中位）：起点 {_fmt(result['boundary_start_median_s'])}s，终点 {_fmt(result['boundary_end_median_s'])}s",
        f"- 只落在未审核区、被排除的预测：{result['excluded_unknown_events']}",
        "",
        f"- 事件 F1：{_fmt(result['event_f1'])}；匹配 IoU 门槛：{result['iou_threshold']}",
        "",
        "> 文件准确率 = (TP+TN)/已知文件数；时长准确率 = (TP秒+TN秒)/已审核秒数。",
        "> 未审核区间不计时长或事件指标；整段粗标只参与文件指标，不展开为精确异常区间。",
        "> 正常样本占比高会抬高准确率，请结合异常召回率、误报和 F1 判断效果。",
        "",
        "逐文件对比见 per_file.csv；预测/真值窗口及一对一匹配见 events.csv（index 从 0 开始）。",
        "完整参数、原始标签、预测及匹配细节见 metrics.json。",
    ]
    (out / "brief.md").write_text("\n".join(lines), encoding="utf-8")
    return out / "metrics.json", out / "brief.md"


def _fmt(v) -> str:
    if v is None:
        return "不可计算（无对应样本）"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _percent(value):
    return f"{value:.2%}" if value is not None else "不可计算（无对应样本）"
