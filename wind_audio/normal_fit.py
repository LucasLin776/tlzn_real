"""Fit only explicit normal interiors; calibrate and check on separate files."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio import load_wav
from .evaluate import aggregate_file_metrics, evaluate_file_events
from .features import FeatureConfig, extract_features
from .l2.common import file_hash
from .markers import _natural_key, import_markers
from .normal import NormalConfig, NormalModelDetector, context_margin, normal_features, normal_windows


def split_records(records, calibration_fraction=0.2, test_fraction=0.2):
    fractions = np.asarray([calibration_fraction, test_fraction])
    if not np.isfinite(fractions).all() or (fractions <= 0).any() or fractions.sum() >= 1:
        raise ValueError("校准/留出比例必须为正且合计小于 1")
    ordered = sorted(records, key=lambda r: _natural_key(r.path))
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


def fit_normal(audio_dir, labels_dir, out_dir, *, pairing="stem", fully_reviewed=False,
               feature_config=None, normal=None, alpha=0.01,
               calibration_fraction=0.2, test_fraction=0.2, verbose=True):
    if not fully_reviewed:
        raise ValueError("请先确认全部正常区间已标出，再加 --fully-reviewed-normal；未标注不自动等于异常")
    if not np.isfinite(alpha) or not 0 < alpha < 0.5:
        raise ValueError("alpha 需在 (0, 0.5) 内")
    out_dir = Path(out_dir)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ValueError("训练输出目录非空，请使用新目录，避免覆盖模型或报告")
    cfg, normal = feature_config or FeatureConfig(), normal or NormalConfig()
    records, annotations, warnings = import_markers(
        audio_dir, labels_dir, out_dir / "manifest.csv", out_dir / "annotations.csv",
        pairing=pairing, stable_markers=True)
    # Do not silently mix old anomaly-marker batches with normal-marker data.
    wrong = sorted({a.fault_type for a in annotations if a.label == "normal"
                    and not (a.fault_type.startswith("正常") or a.fault_type.lower().startswith("normal"))})
    if wrong:
        raise ValueError(f"仅接收名称为 正常/normal（可带编号）的正常区间，发现: {wrong}")
    splits = split_records(records, calibration_fraction, test_fraction)
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
        cached[record.audio_id] = (bundle.times, values)
        if group in normals:
            normals[group].append(values[mask])
        files.append({"audio_id": record.audio_id, "path": record.path, "sha256": digest,
                      "split": group, "explicit_normal_core_frames": int(mask.sum()),
                      "normal_core_s": float(mask.sum() * cfg.target_step)})
        if verbose and (i % 12 == 0 or i == len(records)):
            print(f"正常特征 {i}/{len(records)}", flush=True)
    for group in normals:
        usable_files = sum(f["split"] == group and f["explicit_normal_core_frames"] > 0 for f in files)
        if usable_files < 2:
            raise ValueError(f"{group} 至少需要两条包含足够长正常片段的音频")
    detector = NormalModelDetector.fit(np.concatenate(normals["train"]), np.concatenate(normals["calibration"]),
                                       feature_config=cfg, normal=normal, alpha=alpha)
    detector.training.update(split_ids={k: [r.audio_id for r in v] for k, v in splits.items()},
                             context_margin_s=margin, label_semantics="all_normal_intervals_marked",
                             pairing=pairing)
    report = {"model": detector.to_dict(), "files": files, "warnings": warnings,
              "split_policy": "temporal 60/20/20 by default; no test-based parameter selection",
              "limitations": ["同一会话时间留出，不代表跨设备/跨会话泛化；B04/B05 存在重复音频，不能互作独立测试集。",
                              "历史版本已使用过 B05；本次留出未参与此模型拟合，但并非全项目从未看过的盲测集。",
                              "正常片段较短时边界上下文被排除；alpha 不是事件误报率或准确率保证。"],
              "metrics": {}}
    for group, items in splits.items():
        rows = []
        for record in items:
            times, values = cached[record.audio_id]
            score = (values / detector.scales).max(axis=1)
            events, _ = normal_windows(score, times, record.duration_s, detector.threshold, normal, cfg.target_step)
            rows.append(evaluate_file_events(record, annotations, events))
        summary = aggregate_file_metrics(rows, with_groups=False)
        summary["per_file"] = rows
        report["metrics"][group] = summary
    return detector, report
