"""Explicit input preflight and train -> normal calibration -> val/test reports.

Importing or checking this module never imports torch, extracts model features,
optimizes weights, fits a threshold, or evaluates a recording.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..audio import wav_info
from .data import (contained, from_directories, load_dataset, normal_crops, partition,
                   read_json, validate, write_json)


def training_data(args):
    direct = (args.audio_dir, args.labels, args.pairing, args.pairing_file, args.splits,
              args.semantics, args.marker_label)
    if args.dataset:
        if any(v is not None for v in direct):
            raise ValueError("--dataset 不能与音频/标签目录、配对、语义或划分参数混用")
        return load_dataset(args.dataset)
    if not args.audio_dir or not args.labels:
        raise ValueError("训练必须同时指定 --audio-dir 和 --labels（或明确指定 --dataset 快照）")
    semantics = read_json(args.semantics) if args.semantics else (
        {"all_markers": args.marker_label} if args.marker_label else None)
    return from_directories(args.audio_dir, args.labels, pairing=args.pairing, semantics=semantics,
                            splits_path=args.splits, pairing_file=args.pairing_file)


def preflight(data, cfg, *, reports=True, min_frames=1000, quantiles=(.95, .995, .9995)):
    """Read headers and count admissible normal contexts, without model features."""
    validate(data)
    required = ("train", "calibration", "val", "test") if reports else ("train",)
    for split in required:
        partition(data, split)  # fail before any training if a required split is missing
    if (len(quantiles) != 3 or not all(math.isfinite(q) for q in quantiles)
            or not 0 < quantiles[0] < quantiles[1] < quantiles[2] < 1
            or type(min_frames) is not int or min_frames < 2):
        raise ValueError("正常校准需要三个递增分位数及至少 2 帧；这里只检查配置，不拟合阈值")
    per_file, warnings = [], []
    for r in data["records"]:
        info = wav_info(r["audio_path"])
        margin = 0 if info.sample_rate == cfg.sample_rate else 16 / info.sample_rate
        crops = list(normal_crops(r, cfg, margin)) if r["split"] == "train" else []
        train_frames = sum((b - a - cfg.n_fft + cfg.hop) // cfg.stride_samples - 2 * cfg.halo for a, b in crops)
        cal_frames = 0
        if r["split"] == "calibration":
            samples = int(math.floor(info.n_frames * cfg.sample_rate / info.sample_rate))
            n = max(0, ((samples - cfg.n_fft) // cfg.hop + 1) // 8)
            times = cfg.offset_s + np.arange(n) * cfg.step_s
            radius = cfg.context_radius_s + margin
            mask = (times - radius >= 0) & (times + radius <= samples / cfg.sample_rate)
            cal_frames = int((mask & contained(times, r["intervals"], radius)).sum())
        per_file.append({"audio_id": r["audio_id"], "split": r["split"], "source_id": r["source_id"],
                         "duration_s": info.duration_s, "normal_training_crops": len(crops),
                         "normal_training_frames": train_frames, "normal_calibration_frames": cal_frames})
        if r["split"] == "train" and not train_frames:
            warnings.append(f"{r['audio_id']}: 没有完整正常上下文，训练将不使用该录音")
    train_count = sum(r["normal_training_frames"] for r in per_file)
    cal_count = sum(r["normal_calibration_frames"] for r in per_file)
    if not train_count:
        raise ValueError("train 集没有完整特征及网络上下文均落在明确正常区间内的样本")
    if reports and cal_count < min_frames:
        raise ValueError(f"calibration 集完整正常上下文仅 {cal_count} 帧，少于 --min-frames {min_frames}")
    if reports:
        for split in ("val", "test"):
            intervals = [a for r in partition(data, split) for a in r["intervals"]]
            if not any(a["label"] != "unknown" for a in intervals):
                raise ValueError(f"{split} 集没有明确审核区间，无法生成有效评估报告")
            if not any(a["label"] == "anomaly" for a in intervals):
                warnings.append(f"{split}: 没有明确异常标注，异常召回率不可计算；未标注区间仍为 unknown")
    return {"status": "valid", "features_extracted": False, "thresholds_fitted": False,
            "splits": {s: sum(r["split"] == s for r in per_file) for s in required},
            "normal_training_frames": train_count, "normal_calibration_frames": cal_count,
            "warnings": warnings, "files": per_file,
            "note": "检查通过不代表模型训练完成；未加载 PANNs，权重仅做存在性检查"}


def run_training(data, checkpoint, output, *, cfg, event_cfg, validation,
                 epochs=20, lr=.001, noise_std=0, augmentation_reviewed=False, seed=0,
                 device="cpu", train_only=False, quantiles=(.95, .995, .9995), min_frames=1000,
                 font_path=None):
    """Called ONLY by the user's explicit training command, never check-only."""
    from .cli import model_module
    from .reporting import export_inputs, report_dependencies, write_evaluation_report
    if not train_only:
        report_dependencies(font_path)
    implementation = model_module()
    out = implementation.new_directory(output)
    write_json(out / "dataset.json", data)
    write_json(out / "input_validation.json", validation)
    export_inputs(data, out)
    trained = implementation.train(out / "dataset.json", checkpoint, out / "trained", cfg=cfg,
                                   event_cfg=event_cfg, epochs=epochs, lr=lr, noise_std=noise_std,
                                   augmentation_reviewed=augmentation_reviewed, seed=seed, device=device)
    write_json(out / "training.json", trained.meta["training"])
    with (out / "training.md").open("x", encoding="utf-8") as f:
        f.write("# L1 PANNs 训练简报\n\n仅使用 train 中完整正常上下文；PANNs 和 BatchNorm 均冻结。\n\n"
                f"优化器步数：{trained.meta['training']['optimizer_steps']}\n\n"
                "训练 MSE 仅代表重建损失，不代表检测准确率。模型先保存至 trained/（未校准）。\n")
    del trained
    if train_only:
        write_json(out / "run.json", {"status": "trained_uncalibrated", "model": "trained/model.json",
                                      "reports": None, "dataset": "dataset.json"})
        return
    calibrated = implementation.calibrate(out / "trained/model.json", out / "dataset.json", out / "calibrated",
                                         checkpoint=checkpoint, device=device, quantiles=quantiles, min_frames=min_frames)
    summaries = {}
    for split in ("val", "test"):
        result = write_evaluation_report(calibrated, data, split, out / split, font_path=font_path)
        summaries[split] = {key: result[key] for key in ("duration_metrics", "normal_metrics", "reviewed_coverage",
                                                        "anomaly_recall_including_abstentions", "visual_report")}
    write_json(out / "metrics.json", summaries)
    with (out / "brief.md").open("x", encoding="utf-8") as f:
        f.write("# L1 PANNs 完整运行简报\n\n"
                "训练：trained/；独立正常校准后模型：calibrated/。\n\n"
                "验证报告：val/comparison.pdf；测试报告：test/comparison.pdf。\n\n"
                "每个集合同时输出 metrics.json、per_file.csv、normal_regions.csv、events.csv、unknown_regions.csv、"
                "labels.csv、brief.md 和 plots/index.csv。\n\n"
                "划分由用户明确指定；val/test 不参与模型或阈值拟合。训练损失不代表检测准确率。\n")
    write_json(out / "run.json", {"status": "complete", "model": "calibrated/model.json", "dataset": "dataset.json",
                                  "reports": {s: f"{s}/comparison.pdf" for s in ("val", "test")}})


def validate_command(args):
    """Validate cheap command settings before opening audio or creating output."""
    if Path(args.out).exists():
        raise ValueError("输出路径已存在；请指定新目录，禁止覆盖历史模型或报告")
    path = Path(args.checkpoint)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"本地 PANNs 权重不存在或为空: {path}；不会自动下载")
    if args.epochs <= 0 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("epochs/lr 必须为正数")
    if not math.isfinite(args.noise_std) or not 0 <= args.noise_std <= .05:
        raise ValueError("noise-std 必须在 [0,0.05] 内")
    if args.noise_std and not args.augmentation_reviewed:
        raise ValueError("增强必须显式 --augmentation-reviewed 确认符合正常定义")
