"""Fit only explicit normal interiors; calibrate and check on separate files.

Two input shapes are supported and can be mixed into one training run:

* one flat audio/label directory pair (the historical entry), and
* a dataset root whose children are batches (``B05/音频`` + ``B05/标注``),
  merged into a single manifest before splitting.

``split_mode="temporal"`` keeps the historical time-ordered 60/20/20 split.
``split_mode="shuffle"`` shuffles all recordings (across batches) with a fixed
seed before splitting.  ``holdout_batches`` additionally moves whole batches
into the check set, which is the only split that measures cross-device
generalisation.

Every report also carries a leave-one-batch-out section: the same model is
refitted without each batch and scored on that batch.  Shuffled holdout mixes
the same device into train and check, so its numbers are optimistic; the
cross-batch numbers are the honest ones.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from .audio import load_wav
from .evaluate import aggregate_file_metrics, evaluate_file_events
from .features import FeatureConfig, extract_features
from .l2.common import file_hash
from .markers import _natural_key, import_batches, import_markers
from .normal import (
    NormalConfig,
    NormalModelDetector,
    context_margin,
    event_level_threshold,
    frame_score,
    normal_complement,
    normal_features,
    normal_windows,
    short_events,
)

SPLIT_MODES = ("temporal", "shuffle")


def split_records(records, calibration_fraction=0.2, test_fraction=0.2, *,
                  mode="temporal", seed=0, holdout_ids=()):
    """Split recordings into train / calibration / test.

    ``temporal`` sorts by path only (the historical behaviour: the last files
    become the held-out check).  ``shuffle`` mixes all recordings first, which
    is required when several batches are pooled: otherwise the held-out check
    is just "the newest batch" instead of a random sample of everything.
    ``holdout_ids`` forces whole recordings (one or more complete batches) into
    the check set and splits only the remainder.
    """
    fractions = np.asarray([calibration_fraction, test_fraction])
    if not np.isfinite(fractions).all() or (fractions <= 0).any() or fractions.sum() >= 1:
        raise ValueError("校准/留出比例必须为正且合计小于 1")
    if mode not in SPLIT_MODES:
        raise ValueError(f"划分方式只支持 {' / '.join(SPLIT_MODES)}")
    ordered = sorted(records, key=lambda r: _natural_key(r.path))
    holdout = set(holdout_ids or ())
    known = {r.audio_id for r in ordered}
    unknown = sorted(holdout - known)
    if unknown:
        raise ValueError(f"留出的批次里有未知 audio_id: {unknown[:5]}")
    held = [r for r in ordered if r.audio_id in holdout]
    remaining = [r for r in ordered if r.audio_id not in holdout]
    if holdout:
        if mode == "shuffle":
            random.Random(int(seed)).shuffle(remaining)
        n = len(remaining)
        nc = max(1, round(n * calibration_fraction / max(1 - test_fraction, 1e-9)))
        if n - nc < 2 or nc < 2 or len(held) < 2:
            raise ValueError("留出批次后，训练/校准各需至少两条音频；请增加数据或调整比例")
        return {"train": remaining[:-nc], "calibration": remaining[-nc:], "test": held}
    if mode == "shuffle":
        random.Random(int(seed)).shuffle(ordered)
    n = len(ordered)
    nc, nt = max(1, round(n * calibration_fraction)), max(1, round(n * test_fraction))
    if n - nc - nt < 2 or nc < 2 or nt < 2:
        raise ValueError("训练、校准和留出检查各需至少两条音频；请增加数据或调整比例")
    return {"train": ordered[:-nc - nt], "calibration": ordered[-nc - nt:-nt], "test": ordered[-nt:]}


def normal_core(times, annotations, duration, margin):
    """Include a frame only if its ENTIRE feature context has explicit normal labels."""
    intervals = []
    for ann in sorted((a for a in annotations if a.label == "normal"), key=lambda a: a.start_s):
        a, b = max(0, ann.start_s), min(duration, ann.end_s)
        if intervals and a <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], b)
        elif b > a:
            intervals.append([a, b])
    mask = np.zeros(len(times), bool)
    for a, b in intervals:
        mask |= (times >= a + margin) & (times <= b - margin)
    for ann in annotations:
        if ann.label != "normal":
            mask &= (times < ann.start_s - margin) | (times > ann.end_s + margin)
    return mask


def _load_dataset(audio_dir, labels_dir, dataset_dir, out_dir, pairing):
    """Import one directory pair or every batch under a dataset root."""
    manifest, annotation_path = Path(out_dir) / "manifest.csv", Path(out_dir) / "annotations.csv"
    if dataset_dir:
        records, annotations, warnings, batch_of = import_batches(
            dataset_dir, manifest, annotation_path, pairing=pairing or "sorted", stable_markers=True)
        return records, annotations, warnings, batch_of, pairing or "sorted"
    if not (audio_dir and labels_dir):
        raise ValueError("请提供 --audio-dir/--labels，或 --dataset-dir（批次根目录）")
    records, annotations, warnings = import_markers(
        audio_dir, labels_dir, manifest, annotation_path,
        pairing=pairing or "stem", stable_markers=True)
    return records, annotations, warnings, {}, pairing or "stem"


def _normal_spans(annotations, audio_id, duration):
    spans = [(max(0.0, a.start_s), min(duration, a.end_s))
             for a in annotations if a.audio_id == audio_id and a.label == "normal"]
    return [(a, b) for a, b in spans if b > a]


def _normal_side(row):
    """Normal-side match rates from an anomaly-polarity evaluation row."""
    tm = row["time_metrics"]
    normal_s, anomaly_s = row["normal_s"], row["anomaly_s"]
    tn, fp, tp, fn = tm["tn"], tm["fp"], tm["tp"], tm["fn"]
    return {"normal_s": normal_s, "anomaly_s": anomaly_s,
            "normal_match": tn / (tn + fp) if tn + fp else None,
            "model_normal_precision": tn / (tn + fn) if tn + fn else None,
            "anomaly_recall": tp / (tp + fn) if tp + fn else None,
            "anomaly_precision": tp / (tp + fp) if tp + fp else None}


def _aggregate_normal(rows):
    n = len(rows)
    normal_s = sum(r["normal_s"] for r in rows)
    anomaly_s = sum(r["anomaly_s"] for r in rows)
    tn = sum(r["time_metrics"]["tn"] for r in rows)
    fp = sum(r["time_metrics"]["fp"] for r in rows)
    tp = sum(r["time_metrics"]["tp"] for r in rows)
    fn = sum(r["time_metrics"]["fn"] for r in rows)
    return {"files": n, "normal_s": normal_s, "anomaly_s": anomaly_s,
            "normal_match": tn / (tn + fp) if tn + fp else None,
            "model_normal_precision": tn / (tn + fn) if tn + fn else None,
            "anomaly_recall": tp / (tp + fn) if tp + fn else None,
            "anomaly_precision": tp / (tp + fp) if tp + fp else None,
            "alarmed_normal_s": fp, "missed_anomaly_s": fn}


def _short_summary(records, annotations, shorts):
    """Presence-only statistics for the impulse channel (never in time metrics)."""
    near = {}
    for ann in annotations:
        if ann.label == "anomaly":
            near.setdefault(ann.audio_id, []).append((ann.start_s, ann.end_s, ann.end_s - ann.start_s < 1.0))
    short_anomalies = sum(1 for spans in near.values() for _, _, short in spans if short)
    hit, normal_alarm_s, unmatched = set(), 0.0, 0
    for record, events in zip(records, shorts):
        normal = _normal_spans(annotations, record.audio_id, record.duration_s)
        spans = near.get(record.audio_id, [])
        for event in events:
            a, b = event["start_s"], event["end_s"]
            touched = {(record.audio_id, ga, gb) for ga, gb, short in spans if short and a < gb and b > ga}
            if touched:
                hit |= touched
            else:
                unmatched += 1
            normal_alarm_s += sum(max(0.0, min(b, y) - max(a, x)) for x, y in normal)
    return {"n_events": sum(len(s) for s in shorts),
            "short_anomalies": short_anomalies, "short_anomalies_hit": len(hit),
            "recall_under_1s": len(hit) / short_anomalies if short_anomalies else None,
            "unmatched_events": unmatched, "normal_alarm_s": normal_alarm_s,
            "note": "脉冲通道只做有无提示，不进入 events / 正常区域 / 时长指标"}


def _cross_batch(records, annotations, cached, batches, groups, *, feature_config, normal,
                 alpha, score_mode, target_fp, iou_threshold):
    """Leave-one-batch-out: refit without a batch, score that batch."""
    names = sorted(set(batches.values()))
    if len(names) < 2:
        return {}
    out = {}
    for batch in names:
        train = [r for r in records if batches[r.audio_id] != batch and groups[r.audio_id] == "train"]
        calib = [r for r in records if batches[r.audio_id] != batch and groups[r.audio_id] == "calibration"]
        held = [r for r in records if batches[r.audio_id] == batch]
        if len(train) < 2 or len(calib) < 2 or not held:
            out[batch] = {"skipped": "训练或校核剩余文件不足两条"}
            continue
        train_values = np.concatenate([cached[r.audio_id][1][cached[r.audio_id][2]] for r in train])
        calib_values = np.concatenate([cached[r.audio_id][1][cached[r.audio_id][2]] for r in calib])
        try:
            detector = NormalModelDetector.fit(train_values, calib_values, feature_config=feature_config,
                                               normal=normal, alpha=alpha, score_mode=score_mode)
            if target_fp is not None:
                detector = _calibrate(detector, calib, cached, annotations, normal, target_fp)
        except ValueError as exc:                     # too little normal context for this fold
            out[batch] = {"skipped": str(exc)}
            continue
        rows = []
        for record in held:
            times, values, _ = cached[record.audio_id]
            score = frame_score(values, detector.scales, detector.score_config)
            events, _ = normal_windows(score, times, record.duration_s, detector.threshold,
                                       normal, feature_config.target_step)
            rows.append(evaluate_file_events(record, annotations, events, iou_threshold))
        summary = _aggregate_normal(rows)
        summary.update(threshold=detector.threshold, scales=detector.scales.tolist(),
                       train_files=len(train), calibration_files=len(calib),
                       held_files=len(held))
        out[batch] = summary
    return out


def _calibrate(detector, calibration_records, cached, annotations, normal, target_fp):
    entries = []
    for record in calibration_records:
        times, values, _ = cached[record.audio_id]
        entries.append({"times": times, "values": values, "duration": record.duration_s,
                        "normal": _normal_spans(annotations, record.audio_id, record.duration_s)})
    threshold, info = event_level_threshold(detector, entries, target_fp, normal_cfg=normal,
                                            floor=detector.threshold)
    calibrated = detector.with_threshold(threshold)
    calibrated.training.update(threshold_policy="event_level", **info)
    return calibrated


def fit_normal(audio_dir=None, labels_dir=None, out_dir=None, *, dataset_dir=None, pairing=None,
               fully_reviewed=False, feature_config=None, normal=None, alpha=0.01,
               calibration_fraction=0.2, test_fraction=0.2, split_mode="temporal", seed=0,
               holdout_batches=(), target_fp=None, score_mode="max",
               boundary_tolerance=0.2, verbose=True):
    if not fully_reviewed:
        raise ValueError("请先确认全部正常区间已标出，再加 --fully-reviewed-normal；未标注不自动等于异常")
    if not np.isfinite(alpha) or not 0 < alpha < 0.5:
        raise ValueError("alpha 需在 (0, 0.5) 内")
    if target_fp is not None and not 0 < target_fp < 1:
        raise ValueError("--target-fp 需在 (0,1) 内；0 表示关闭并退回帧分位口径")
    if dataset_dir and (audio_dir or labels_dir):
        raise ValueError("--dataset-dir 与 --audio-dir/--labels 不能同时使用，请二选一")
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"划分方式只支持 {' / '.join(SPLIT_MODES)}")
    out_dir = Path(out_dir)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ValueError("训练输出目录非空，请使用新目录，避免覆盖模型或报告")
    cfg, normal = feature_config or FeatureConfig(), normal or NormalConfig()
    records, annotations, warnings, batch_of, pairing = _load_dataset(
        audio_dir, labels_dir, dataset_dir, out_dir, pairing)
    # Do not silently mix old anomaly-marker batches with normal-marker data.
    wrong = sorted({a.fault_type for a in annotations if a.label == "normal"
                    and not (a.fault_type.startswith("正常") or a.fault_type.lower().startswith("normal"))})
    if wrong:
        raise ValueError(f"仅接收名称为 正常/normal（可带编号）的正常区间，发现: {wrong}")
    batches = {}
    for name in batch_of.values():
        batches[name] = batches.get(name, 0) + 1
    if batches:
        warnings.append(f"合并 {len(batches)} 个批次: " + "，".join(f"{k} {v} 条" for k, v in sorted(batches.items())))
    holdout = []
    if holdout_batches:
        missing = sorted(set(holdout_batches) - set(batches))
        if missing:
            raise ValueError(f"--holdout-batch 未在数据中找到: {missing}；现有批次 {sorted(batches)}")
        holdout = [r.audio_id for r in records if batch_of.get(r.audio_id) in set(holdout_batches)]
        warnings.append(f"留出批次 {sorted(set(holdout_batches))} 作为检查集，不参与尺度/阈值拟合")
    splits = split_records(records, calibration_fraction, test_fraction, mode=split_mode, seed=seed,
                           holdout_ids=holdout)
    by_id = {r.audio_id: [a for a in annotations if a.audio_id == r.audio_id] for r in records}
    groups = {r.audio_id: name for name, items in splits.items() for r in items}
    cached, normals, hashes, files = {}, {"train": [], "calibration": []}, {}, []
    margin = context_margin(cfg, normal)
    for i, record in enumerate(records, 1):
        digest = file_hash(record.path)
        if digest in hashes:
            raise ValueError(f"重复音频: {record.audio_id} / {hashes[digest]}；请去重后训练，避免数据泄漏")
        hashes[digest] = record.audio_id
        waveform, sr = load_wav(record.path, target_sr=cfg.sr)
        bundle = extract_features(waveform, sr, cfg, record.audio_id)
        if bundle.duration_s < 2 * margin:
            raise ValueError(f"{record.audio_id}: 音频太短，缺少完整观察窗")
        values = normal_features(bundle, normal)
        mask = normal_core(bundle.times, by_id[record.audio_id], bundle.duration_s, margin)
        group = groups[record.audio_id]
        cached[record.audio_id] = (bundle.times, values, mask)
        if group in normals:
            normals[group].append(values[mask])
        files.append({"audio_id": record.audio_id, "path": record.path, "sha256": digest,
                      "batch": batch_of.get(record.audio_id, ""), "split": group,
                      "explicit_normal_core_frames": int(mask.sum()),
                      "normal_core_s": float(mask.sum() * cfg.target_step)})
        if verbose and (i % 12 == 0 or i == len(records)):
            print(f"正常特征 {i}/{len(records)}", flush=True)
    for group in normals:
        usable_files = sum(f["split"] == group and f["explicit_normal_core_frames"] > 0 for f in files)
        if usable_files < 2:
            raise ValueError(f"{group} 至少需要两条包含足够长正常片段的音频")
    detector = NormalModelDetector.fit(np.concatenate(normals["train"]), np.concatenate(normals["calibration"]),
                                       feature_config=cfg, normal=normal, alpha=alpha,
                                       score_mode=score_mode)
    threshold_info = {}
    if target_fp is not None:
        detector = _calibrate(detector, splits["calibration"], cached, annotations, normal,
                              target_fp)
        threshold_info = {k: detector.training[k] for k in ("target_fp", "achieved_calibration_fp")}
    detector.training.update(split_ids={k: [r.audio_id for r in v] for k, v in splits.items()},
                             context_margin_s=margin, label_semantics="all_normal_intervals_marked",
                             pairing=pairing, split_mode=split_mode, split_seed=int(seed),
                             holdout_batches=sorted(set(holdout_batches)),
                             batches=batches, score_mode=score_mode,
                             threshold_policy="event_level" if target_fp is not None else "frame_quantile",
                             dataset_dir=str(Path(dataset_dir).resolve()) if dataset_dir else None)
    report = {"model": detector.to_dict(), "files": files, "warnings": warnings,
              "split_policy": _split_policy(split_mode, seed, calibration_fraction, test_fraction,
                                            holdout_batches),
              "threshold_calibration": threshold_info,
              "limitations": _limitations(split_mode, bool(holdout_batches)),
              "metrics": {}}
    for group, items in splits.items():
        rows, shorts, main_events = [], [], []
        for record in items:
            times, values, _ = cached[record.audio_id]
            score = frame_score(values, detector.scales, detector.score_config)
            events, _ = normal_windows(score, times, record.duration_s, detector.threshold,
                                       normal, cfg.target_step)
            rows.append(evaluate_file_events(record, annotations, events))
            main_events.append(events)
            shorts.append(short_events(score, times, record.duration_s, detector.threshold,
                                       normal, cfg.target_step, covered=events))
        summary = aggregate_file_metrics(rows, with_groups=False)
        summary["per_file"] = rows
        summary.update(_aggregate_normal(rows))
        if boundary_tolerance > 0:
            tolerant_rows = [evaluate_file_events(record, annotations, events, 0.1, boundary_tolerance)
                             for record, events in zip(items, main_events)]
            tolerant = aggregate_file_metrics(tolerant_rows, with_groups=False)
            summary["boundary_tolerant"] = {"tolerance_s": boundary_tolerance,
                                            "time_accuracy": tolerant["time_accuracy"],
                                            "time_metrics": tolerant["time_metrics"],
                                            "event_recall": tolerant["event_recall"],
                                            "event_precision": tolerant["event_precision"]}
        summary["short_channel"] = _short_summary(items, annotations, shorts)
        report["metrics"][group] = summary
    report["cross_batch"] = _cross_batch(records, annotations, cached, batch_of, groups,
                                         feature_config=cfg, normal=normal, alpha=alpha,
                                         score_mode=score_mode, target_fp=target_fp,
                                         iou_threshold=0.1)
    report["cross_batch_note"] = ("每个批次单独留出重新拟合后的表现；这是跨设备泛化的诚实估计，"
                                  "打散留出集因为同设备进训练而偏乐观。")
    return detector, report


def _split_policy(split_mode, seed, calibration_fraction, test_fraction, holdout_batches):
    share = f"{1 - calibration_fraction - test_fraction:.0%}/{calibration_fraction:.0%}/{test_fraction:.0%}"
    if holdout_batches:
        return (f"留出批次 {sorted(set(holdout_batches))} 作为检查集，其余录音打散后按校准比例划分；"
                "这是跨设备口径，指标应视为对外可复现的下限")
    if split_mode == "shuffle":
        return (f"所有录音（跨批次）用 seed={seed} 全局打散后按 {share} 划分；"
                "每个集合都含各批次的随机样本，指标对批次混合后的分布更有代表性；"
                "同一设备/同一会话的不同录音会同时出现在训练与留出检查中，留出指标偏乐观")
    return (f"按文件路径自然顺序（时间顺序）做 {share} 划分：最后一组是留出检查；"
            "多批次合并时等于按批次顺序切分，最后一批几乎只进留出检查")


def _limitations(split_mode, holdout):
    common = ["正常片段较短时边界上下文被排除；alpha 不是事件误报率或准确率保证。",
              "同一会话的相邻录音共享工况，任何划分都不能代表跨设备/跨季节泛化。"]
    if holdout:
        return ["跨设备口径但只留出一台设备；不同季节/负荷仍需新数据验证。"] + common
    if split_mode == "shuffle":
        return ["打散划分让同一设备的录音同时进入训练和留出检查，留出指标是内插估计，"
                "跨设备表现请看 cross_batch。"] + common
    return ["时间顺序留出只检查最新的一段，不代表整体分布；多批次合并时更容易把整批留作测试。"] + common
