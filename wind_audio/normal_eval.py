"""Evaluate NORMAL regions directly; never reinterpret normals as anomalies."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .labeling import collapse_anomalies, label_text
from .evaluate import _overlap, _percent, _reviewed_regions, binary_metrics, merge_intervals


def evaluate_normal_regions(record, annotations, predictions, iou_threshold=0.1,
                            boundary_tolerance=0.0):
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU 门槛需要在 (0,1] 内")
    regions = _reviewed_regions(record, annotations, boundary_tolerance)
    # Use actual reviewed normal ranges, not inversion of the anomaly-event list.
    gt = [{"start_s": a, "end_s": b} for a, b in merge_intervals(
        [r for r in regions if r["label"] == "normal"])]
    normal_s = sum(r["end_s"] - r["start_s"] for r in regions if r["label"] == "normal")
    anomaly_s = sum(r["end_s"] - r["start_s"] for r in regions if r["label"] == "anomaly")
    pred = [{"start_s": max(0., a), "end_s": min(record.duration_s, b)}
            for a, b in merge_intervals(predictions) if min(record.duration_s, b) > max(0., a)]
    overlap = lambda event, label: sum(_overlap(event["start_s"], event["end_s"], r["start_s"], r["end_s"])
                                      for r in regions if r["label"] == label)
    normal_overlap = [overlap(p, "normal") for p in pred]
    anomaly_overlap = [overlap(p, "anomaly") for p in pred]
    reviewed_duration = [a + b for a, b in zip(normal_overlap, anomaly_overlap)]
    candidates = []
    for i, event in enumerate(pred):
        for j, label in enumerate(gt):
            inter = _overlap(event["start_s"], event["end_s"], label["start_s"], label["end_s"])
            union = reviewed_duration[i] + label["end_s"] - label["start_s"] - inter
            iou = inter / union if union > 0 else 0
            if iou >= iou_threshold:
                candidates.append((iou, i, j))
    used_pred, used_gt, matches = set(), set(), []
    for iou, i, j in sorted(candidates, reverse=True):
        if i not in used_pred and j not in used_gt:
            used_pred.add(i)
            used_gt.add(j)
            matches.append({"pred_index": i, "gt_index": j, "iou": iou})
    tp = min(normal_s, sum(normal_overlap))
    fp = min(anomaly_s, sum(anomaly_overlap))
    tm = binary_metrics(tp, max(0., anomaly_s - fp), fp, max(0., normal_s - tp))
    def _complement(regions):
        """Deviation events = recording minus the normal regions."""
        out, prev = [], 0.0
        for r in sorted(regions, key=lambda r: r["start_s"]):
            if r["start_s"] > prev + 1e-9:
                out.append({"start_s": prev, "end_s": r["start_s"]})
            prev = max(prev, r["end_s"])
        if record.duration_s > prev + 1e-9:
            out.append({"start_s": prev, "end_s": record.duration_s})
        return out

    return {"audio_id": record.audio_id, "duration_s": record.duration_s, "target_label": "normal",
            "predicted_events": _complement(pred), "ground_truth_events": _complement(gt),
            "predicted_label": collapse_anomalies(_complement(pred), record.duration_s),
            "normal_s": normal_s, "anomaly_s": anomaly_s, "reviewed_s": normal_s + anomaly_s,
            "unknown_s": max(0., record.duration_s - normal_s - anomaly_s),
            "ground_truth_normal_regions": gt, "predicted_normal_regions": pred,
            "n_gt_regions": len(gt), "n_pred_regions": len(pred), "n_matched": len(matches),
            "n_pred_reviewed": sum(v > 1e-12 for v in reviewed_duration),
            "excluded_unknown_regions": sum(v <= 1e-12 for v in reviewed_duration),
            "matches": matches, "iou_threshold": iou_threshold,
            "time_metrics": tm, "time_accuracy": tm["accuracy"]}


def aggregate_normal_metrics(rows):
    tm = binary_metrics(*(sum(r["time_metrics"][k] for r in rows) for k in ("tp", "tn", "fp", "fn")))
    gt, pred, matched = (sum(r[k] for r in rows) for k in ("n_gt_regions", "n_pred_reviewed", "n_matched"))
    return {"schema_version": "normal_regions_eval_v1", "target_label": "normal", "files": len(rows),
            "time_accuracy": tm["accuracy"], "time_metrics": tm, "normal_time_metrics": tm,
            "normal_region_metrics": {"n_gt": gt, "n_pred": sum(r["n_pred_regions"] for r in rows),
                                      "n_pred_reviewed": pred, "n_matched": matched,
                                      "precision": matched / pred if pred else None,
                                      "recall": matched / gt if gt else None,
                                      "f1": 2 * matched / (gt + pred) if gt + pred else None},
            "reviewed_normal_s": sum(r["normal_s"] for r in rows),
            "reviewed_anomaly_s": sum(r["anomaly_s"] for r in rows),
            "unknown_s": sum(r["unknown_s"] for r in rows), "per_file": rows,
            "metric_definitions": {"positive_label": "normal", "tp": "真实正常且预测正常的秒数",
                                   "fp": "真实异常却预测正常的秒数（异常漏检）",
                                   "fn": "真实正常却未预测正常的秒数（误报异常）",
                                   "tn": "真实异常且未预测正常的秒数"}}


def write_normal_outputs(result, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (out / "normal_regions.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        fields = ["audio_id", "label", "kind", "index", "start_s", "end_s", "matched_index", "iou"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in result["per_file"]:
            for kind, key, own, other in (("prediction", "predicted_normal_regions", "pred_index", "gt_index"),
                                          ("label", "ground_truth_normal_regions", "gt_index", "pred_index")):
                matches = {m[own]: m for m in row["matches"]}
                for i, event in enumerate(row[key]):
                    m = matches.get(i, {})
                    writer.writerow({"audio_id": row["audio_id"], "label": "normal", "kind": kind,
                                     "index": i, **event, "matched_index": m.get(other), "iou": m.get("iou")})
    with (out / "labels.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["audio_id", "n_intervals", "rule", "intervals_s"])
        for row in result["per_file"]:
            lab = row["predicted_label"]
            writer.writerow([row["audio_id"], lab["count"], lab["rule"], label_text(lab)])
    with (out / "per_file.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        fields = ["audio_id", "target_label", "duration_s", "n_gt_regions", "n_pred_regions", "n_matched",
                  "time_accuracy", "precision", "recall", "tp", "tn", "fp", "fn"]
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in result["per_file"]:
            writer.writerow({**row, **row["time_metrics"]})
    tm, rm = result["time_metrics"], result["normal_region_metrics"]
    lines = ["# 正常区域测试简报", "", "本报告直接比较人工正常区域与模型正常区域；所有主指标的正类均为 normal。",
             "模型正常区域来自已训练模型的判断，不读取本批标签、不重新训练。", "",
             f"- 时长准确率：{_percent(tm['accuracy'])}",
             f"- 正常时长精确率：{_percent(tm['precision'])}；正常时长召回率：{_percent(tm['recall'])}",
             f"- 正常区域：人工 {rm['n_gt']} 段，预测 {rm['n_pred']} 段，匹配 {rm['n_matched']} 段",
             f"- 正常区域精确率：{_percent(rm['precision'])}；召回率：{_percent(rm['recall'])}",
             f"- 异常错当正常（FP）：{tm['fp']:.3f} 秒；正常被排除（FN）：{tm['fn']:.3f} 秒", "",
             "注意：正常召回率高并不意味着异常检出率高，必须同时看异常错当正常的时长。",
             "逐段范围见 normal_regions.csv；逐文件指标见 per_file.csv；完整结果见 metrics.json。"]
    tolerant = result.get("boundary_tolerant")
    if tolerant:
        tm = tolerant["time_metrics"]
        lines += ["", f"- 标注边界 ±{tolerant['tolerance_s']:.1f} 秒不计分的容差口径："
                      f"时长准确率 {_percent(tm['accuracy'])}，"
                      f"正常时长精确率 {_percent(tm['precision'])}、召回率 {_percent(tm['recall'])}"]
    if result.get("visual_report"):
        lines += [f"PDF：comparison.pdf，共 {result['visual_report']['pages']} 页，上行为人工正常，下行为模型正常。"]
    if result.get("data_reuse_warning"):
        lines += ["", result["data_reuse_warning"]]
    (out / "brief.md").write_text("\n".join(lines), encoding="utf-8")
    return out / "metrics.json", out / "brief.md"
