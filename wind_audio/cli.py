"""L1 training, inference, review, and the shared command-line entry point."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np

from .audio import load_wav, wav_info, write_wav
from .backends import backend_versions
from .calibrate import apply_calibration, calibrate_detector
from .data import (
    AudioRecord,
    Annotation,
    anomaly_events,
    dataset_summary,
    import_simple,
    label_mask,
    migrate_legacy_labels,
    read_annotations,
    read_manifest,
    regions_from_mask,
    resolve_audio_path,
    validate_dataset,
    write_annotations,
)
from .detectors import (
    DEFAULT_POSTPROCESS,
    BaselineDetector,
    SupervisedDetector,
    cap_training_data,
    load_detector,
)
from .evaluate import evaluate_detector, write_eval_outputs
from .events import extract_events
from .features import FeatureConfig, build_matrix, extract_features, load_or_extract
from .render import score_png, spectrogram_png, waveform_png
from .report import write_report
from .review import (
    export_template,
    import_reviews,
    read_reviews,
    review_summary,
    reviews_to_annotations,
)
from .splits import assert_no_leakage, load_splits, make_splits, save_splits


# --------------------------------------------------------------------- utils
def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _write_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", name)[:80] or "audio"


def _feature_overrides(args) -> dict:
    out = {}
    for key in ("sr", "n_fft", "hop", "n_mels", "fmin", "fmax", "target_step", "context_s", "normalize"):
        value = getattr(args, key, None)
        if value is not None:
            out[key] = value
    return out


def _cfg_from_args(args) -> FeatureConfig:
    base = FeatureConfig().to_dict()
    base.update(_feature_overrides(args))
    return FeatureConfig.from_dict(base)


def _assert_model_cfg(detector, args) -> None:
    overrides = _feature_overrides(args)
    if not overrides:
        return
    cfg = detector.feature_config
    for key, value in overrides.items():
        if isinstance(value, float) and isinstance(getattr(cfg, key), float):
            if abs(value - getattr(cfg, key)) > 1e-9:
                raise SystemExit(
                    f"--{key} 与模型特征配置不一致（模型 {getattr(cfg, key)}，命令 {value}）。"
                    "推理不允许临时改特征参数；请用训练时的配置。"
                )
        elif value != getattr(cfg, key):
            raise SystemExit(
                f"--{key} 与模型特征配置不一致（模型 {getattr(cfg, key)}，命令 {value}）。"
                "推理不允许临时改特征参数；请用训练时的配置。"
            )


def _load_dataset(args, check_audio: bool = True):
    manifest = Path(args.manifest)
    records = read_manifest(manifest)
    anns = read_annotations(args.annotations) if args.annotations else []
    errors, warnings = validate_dataset(records, anns, manifest, check_audio=check_audio)
    return records, anns, errors, warnings


def _require_clean(errors: list[str]) -> None:
    if errors:
        raise SystemExit("数据校验失败：\n- " + "\n- ".join(errors))


def _anomaly_count(ids: list[str], records: list[AudioRecord], anns: list[Annotation]) -> int:
    by_id = {r.audio_id: r for r in records}
    return sum(len(anomaly_events(by_id[i], anns)) for i in ids if i in by_id)


# ---------------------------------------------------------------- data checks
def cmd_check(args) -> int:
    records, anns, errors, warnings = _load_dataset(args)
    summary = dataset_summary(records, anns)
    payload = {
        "manifest": str(args.manifest),
        "annotations": str(args.annotations),
        "summary": summary,
        "errors": errors,
        "warnings": warnings,
        "backend_versions": backend_versions(),
    }
    if args.json:
        print(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print(f"文件 {summary['files']} 个 · 来源 {summary['sources']} 个 · 设备 {summary['devices']} 个")
        print(f"已审核正常 {summary['normal_s']:.1f}s · 已审核异常 {summary['anomaly_s']:.1f}s "
              f"· 未审核 {summary['unknown_s']:.1f}s")
        print(f"异常事件 {summary['anomaly_events']} 个 · 完整审核文件 {summary['fully_reviewed_files']} "
              f"· 含粗标文件 {summary['coarse_label_files']}")
        if summary["reasons"]:
            print("\n不足以训练/评估的原因：")
            for reason in summary["reasons"]:
                print(f"- {reason}")
        if warnings:
            print("\n警告：")
            for w in warnings:
                print(f"- {w}")
        if errors:
            print("\n错误：")
            for e in errors:
                print(f"- {e}")
    return 1 if errors else 0


def cmd_import_simple(args) -> int:
    reviewed = None
    if args.reviewed_list:
        reviewed = {
            line.strip()
            for line in Path(args.reviewed_list).read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
    records, anns, warnings = import_simple(
        args.audio_dir, args.labels, args.manifest, args.annotations,
        assume_unlabeled_normal=args.assume_unlabeled_normal,
        reviewed_files=reviewed,
    )
    print(f"已导入 {len(records)} 个音频，{len(anns)} 条区间标注")
    print(f"  manifest: {args.manifest}")
    print(f"  annotations: {args.annotations}")
    if args.assume_unlabeled_normal:
        mode = "全部文件视为已完整审核，未标注区域推导为 normal"
    elif reviewed:
        mode = f"{len(reviewed)} 个文件按 reviewed-list 推导 normal；其余未标注区域保持 unknown"
    else:
        mode = "保持 unknown"
    print(f"  未标注区域: {mode}")
    for w in warnings:
        print(f"警告: {w}")
    return 0


def cmd_migrate(args) -> int:
    records, anns, warnings = migrate_legacy_labels(
        args.labels, args.audio_dir, args.manifest, args.annotations
    )
    print(f"迁移完成：{len(records)} 个音频，{len(anns)} 条标注 -> {args.manifest}, {args.annotations}")
    print("旧 labels.csv 未修改；file 级粗标不会变成逐帧真值。")
    for w in warnings:
        print(f"警告: {w}")
    return 0


# -------------------------------------------------------------------- split
def cmd_split(args) -> int:
    records, anns, errors, _ = _load_dataset(args)
    _require_clean(errors)
    ratios = tuple(float(x) for x in args.ratios.split(","))
    if len(ratios) != 3:
        raise SystemExit("--ratios 需要三个数，例如 0.6,0.2,0.2")
    split = make_splits(records, anns, group_by=args.group_by, ratios=ratios, seed=args.seed)
    assert_no_leakage(split)
    save_splits(split, args.out)
    print(f"划分已写入 {args.out}（group_by={args.group_by}, seed={args.seed}）")
    for name in ("train", "val", "test"):
        ids = split.ids(name)
        print(f"  {name:5s}: {len(ids):3d} 文件, {_anomaly_count(ids, records, anns):3d} 个异常事件")
    return 0


# -------------------------------------------------------------------- train
def _train_new(args) -> int:
    records, anns, errors, _ = _load_dataset(args)
    _require_clean(errors)
    if not Path(args.splits).exists():
        raise SystemExit(f"划分文件不存在: {args.splits}（先运行 split）")
    split = load_splits(args.splits)
    assert_no_leakage(split)
    cfg = _cfg_from_args(args)
    if args.detector == "baseline":
        detector = BaselineDetector.train(
            records, anns, split.train, args.manifest, cfg,
            version=args.version, threshold=args.threshold,
            max_frames_per_file=args.max_samples_per_file, seed=args.seed,
        )
    else:
        data = build_matrix(
            records, anns, split.train, args.manifest, cfg, args.cache_dir
        )
        if len(data["X"]) == 0:
            raise SystemExit("训练集没有已审核样本；检查标注覆盖范围。")
        capped = cap_training_data(
            data["X"], data["y"], data["groups"], data["file_ids"], data["times"],
            max_total=args.max_total_samples, max_per_file=args.max_samples_per_file, seed=args.seed,
        )
        meta = {
            "audio_ids": split.train,
            "file_ids": capped["file_ids"].tolist(),
            "training": {
                "available": capped["available"],
                "selected": capped["selected"],
            },
        }
        detector, selection = SupervisedDetector.train(
            capped["X"], capped["y"], data["feature_names"], cfg,
            version=args.version, seed=args.seed, metadata=meta,
        )
        print(f"分类后端：请求 {selection.requested}，实际 {selection.used}"
              + ("（回退）" if selection.fallback else ""))
        if selection.reason:
            print(f"  原因: {selection.reason}")

    detector.metadata["split"] = {
        "train": split.train, "val": split.val, "test": split.test,
        "seed": split.seed, "group_by": split.group_by,
    }
    out = Path(args.out)
    detector.save(out)

    if not args.no_calibrate:
        try:
            calib = calibrate_detector(
                detector, records, anns, split.val, args.manifest,
                target_recall=args.target_recall,
            )
            apply_calibration(detector, calib)
            detector.save(out)
            _write_json(out / "calibration.json", calib)
            print(f"校准：阈值 {calib['selected']['threshold']:.4f}，"
                  f"目标召回 {args.target_recall:.2f}，"
                  f"{'达到' if calib['target_reached'] else '未达到'}；"
                  f"验证集召回 {calib['validation']['event_recall']}")
        except Exception as exc:
            print(f"校准失败（模型保留未校准状态）：{exc}", file=sys.stderr)
            detector.calibrated = False
            detector.save(out)

    print(f"模型已保存: {out}")
    print(f"  版本={detector.version} 检测器={detector.detector_type} "
          f"分数类型={detector.score_type} 阈值={detector.threshold}")
    return 0


def cmd_train(args) -> int:
    return _train_new(args)


# --------------------------------------------------------------- calibrate
def cmd_calibrate(args) -> int:
    if args.split != "val":
        raise ValueError("只能使用 val 校准；test 必须保留为独立测试集")
    detector = load_detector(args.model)
    records, anns, errors, _ = _load_dataset(args)
    _require_clean(errors)
    split = load_splits(args.splits)
    assert_no_leakage(split)
    audio_ids = split.val
    calib = calibrate_detector(
        detector, records, anns, audio_ids, args.manifest,
        target_recall=args.target_recall,
    )
    apply_calibration(detector, calib)
    detector.save(args.model)
    _write_json(Path(args.model) / "calibration.json", calib)
    print(f"校准完成：阈值 {calib['selected']['threshold']:.4f}，"
          f"目标召回 {args.target_recall:.2f}，"
          f"{'达到' if calib['target_reached'] else '未达到'}")
    print(f"验证集：召回 {calib['validation']['event_recall']}，"
          f"精确率 {calib['validation']['event_precision']}，"
          f"FA/h {calib['validation']['fa_per_hour']}")
    return 0


# ---------------------------------------------------------------- evaluate
def cmd_evaluate(args) -> int:
    detector = load_detector(args.model)
    records, anns, errors, _ = _load_dataset(args)
    _require_clean(errors)
    split = load_splits(args.splits)
    audio_ids = split.ids(args.split)
    assert_no_leakage(split)
    pp = {}
    for key in ("min_duration", "merge_gap", "low_ratio"):
        value = getattr(args, key, None)
        if value is not None:
            pp[key] = value
    if (args.threshold is not None or pp) and args.split == "test":
        print("警告：在 test 上覆盖阈值/后处理，结果不再是冻结模型的正式测试成绩。", file=sys.stderr)
    result = evaluate_detector(
        detector, records, anns, audio_ids, args.manifest, detector.feature_config,
        threshold=args.threshold, postprocess=pp or None, iou_threshold=args.match_iou,
    )
    out = Path(args.out)
    metrics_path, brief_path = write_eval_outputs(result, out)
    print(f"评估：{args.split} 集，{result['files']} 文件，"
          f"GT {result['n_gt_events']}，预测 {result['n_pred_events']}")
    print(f"  召回 {result['event_recall']}；精确率 {result['event_precision']}；"
          f"FA/h {result['fa_per_hour']}；覆盖 {result['coverage']}")
    print(f"  指标: {metrics_path}\n  简报: {brief_path}")
    return 0


# --------------------------------------------------------------------- run
def _copy_audio(src: Path, out_dir: Path, audio_id: str) -> str:
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    target = audio_dir / f"{_safe_name(audio_id)}{src.suffix.lower()}"
    if src.resolve() != target.resolve():
        shutil.copy2(src, target)
    return str(target.relative_to(out_dir))


def _make_clip(x, sr, start, end, path: Path, context: float, duration: float) -> dict:
    a = max(0.0, start - context)
    b = min(duration, end + context)
    ia, ib = int(round(a * sr)), int(round(b * sr))
    write_wav(path, x[ia:ib], sr)
    return {"path": str(path.name), "start_s": round(a, 3), "end_s": round(b, 3), "sr": sr}


def _events_csv(events: list[dict], path: Path) -> None:
    fields = [
        "event_id", "start_s", "end_s", "duration_s", "peak_score", "mean_score",
        "score_type", "threshold", "centroid_hz", "flatness", "peak_freq_hz",
        "harmonicity_db", "share_0_200", "share_200_1k", "share_1k_4k",
        "share_4k_10k", "share_10k_up", "onset_slope_per_s", "at_edge",
        "clip_path", "clip_start_s", "clip_end_s",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for ev in events:
            clip = ev.get("clip") or {}
            row = {k: ev.get(k) for k in fields}
            row["clip_path"] = clip.get("path")
            row["clip_start_s"] = clip.get("start_s")
            row["clip_end_s"] = clip.get("end_s")
            writer.writerow(row)


def _run_one(
    path: Path,
    detector,
    cfg: FeatureConfig,
    out_dir: Path,
    *,
    threshold: float,
    threshold_source: str,
    postprocess: dict,
    postprocess_source: str,
    fmax_plot: float,
    clip_context: float,
    records: list[AudioRecord],
    anns: list[Annotation],
    manifest_path: Path | None = None,
    reviews_by_id: dict | None = None,
) -> dict:
    x, sr = load_wav(path, target_sr=cfg.sr)
    if len(x) == 0:
        raise ValueError(f"空音频: {path}")
    duration = len(x) / sr
    audio_id = path.stem
    record = None
    if manifest_path:
        for rec in records:
            try:
                same = resolve_audio_path(rec, manifest_path).resolve() == path.resolve()
            except Exception:
                same = False
            if same:
                record = rec
                audio_id = rec.audio_id
                break
    bundle = extract_features(x, sr, cfg, audio_id)
    events = detector.predict_events(bundle, threshold=threshold, postprocess=postprocess)
    from .l2.common import digest, file_hash
    audio_hash = file_hash(path)
    for ev in events:
        ev["event_id"] = audio_id + "_" + digest(
            [audio_hash, ev["start_s"], ev["end_s"], detector.version,
             cfg.to_dict(), threshold, postprocess])[:20]
        ev["review"] = (reviews_by_id or {}).get(ev["event_id"])
    regions = regions_from_mask(label_mask(record, anns, cfg.target_step)) if record else []
    gt = anomaly_events(record, anns) if record else []

    if (out_dir / "review.csv").exists():
        raise ValueError("输出目录已有审核记录，请使用新目录，避免覆盖人工标注")
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_rel = _copy_audio(path, out_dir, audio_id)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for ev in events:
        clip = _make_clip(x, sr, ev["start_s"], ev["end_s"], clips_dir / f"{_safe_name(ev['event_id'])}.wav", clip_context, duration)
        clip["path"] = f"clips/{clip['path']}"
        ev["clip"] = clip

    waveform_png(x, duration, events, regions, out_dir / "waveform.png")
    spectrogram_png(
        bundle.spec_db, bundle.freqs, bundle.mel_times, duration, events, regions,
        out_dir / "spectrogram.png", f_max=min(fmax_plot, cfg.fmax, sr / 2),
    )
    score = detector.score(bundle)
    score_png(bundle.times, score, threshold, events, regions, duration, out_dir / "score.png")

    payload = {
        "schema_version": "2.0",
        "audio": {
            "id": audio_id,
            "file": path.name,
            "path": str(path.resolve()),
            "sha256": audio_hash,
            "duration_s": round(duration, 6),
            "sr": sr,
            "analyzed_fmax_hz": cfg.fmax,
        },
        "model": {
            "version": detector.version,
            "detector": detector.detector_type,
            "score_type": detector.score_type,
            "threshold": threshold,
            "threshold_source": threshold_source,
            "postprocess": postprocess,
            "postprocess_source": postprocess_source,
            "calibrated": detector.calibrated,
            "demo": detector.demo,
            "backend": detector.metadata.get("backend"),
        },
        "events": events,
        "annotations": {"regions": regions, "gt_events": gt, "source": str(record.audio_id) if record else None},
    }
    _write_json(out_dir / "events.json", payload)
    _events_csv(events, out_dir / "events.csv")

    export_template(payload, out_dir / "review.csv", out_dir / "review.json")
    report = write_report(
        out_dir,
        audio_id=audio_id,
        file_name=path.name,
        duration=duration,
        sr=sr,
        analyzed_fmax=min(fmax_plot, cfg.fmax, sr / 2),
        model_version=detector.version,
        detector_type=detector.detector_type,
        score_type=detector.score_type,
        threshold=threshold,
        threshold_source=threshold_source,
        calibrated=detector.calibrated,
        calibration=detector.metadata.get("calibration"),
        demo=detector.demo,
        backend=detector.metadata.get("backend"),
        feature_config=cfg.to_dict(),
        postprocess=postprocess,
        events=events,
        regions=regions,
        gt_events=gt,
        reviews=reviews_by_id,
        audio_relpath=audio_rel,
        review_relpath="review.csv",
    )
    return {"audio_id": audio_id, "events": len(events), "report": str(report), "out": str(out_dir)}


def cmd_run(args) -> int:
    input_path = Path(args.input)
    if input_path.is_dir():
        files = sorted(input_path.rglob("*.wav"))
    elif input_path.is_file():
        files = [input_path]
    else:
        print(f"输入不存在: {input_path}", file=sys.stderr)
        return 2
    if not files:
        print(f"没有 wav: {input_path}", file=sys.stderr)
        return 2

    detector = load_detector(args.model)
    _assert_model_cfg(detector, args)
    cfg = detector.feature_config

    records, anns = [], []
    if args.manifest:
        records = read_manifest(args.manifest)
        anns = read_annotations(args.annotations) if args.annotations else []
    reviews_by_id = {}
    if args.review:
        reviews_by_id = {r.event_id: r.to_row() for r in read_reviews(args.review)}

    threshold = None if args.threshold is None else float(args.threshold)
    threshold_source = "cli" if threshold is not None else (
        "calibration" if detector.calibrated else "model"
    )
    pp_override = {
        "min_duration": args.min_dur,
        "merge_gap": args.merge_gap,
        "low_ratio": args.low_ratio,
    }
    pp_override = {k: v for k, v in pp_override.items() if v is not None}
    pp_source = "cli" if pp_override else "model"

    out_base = Path(args.out) if args.out else Path("runs")
    results = []
    for path in files:
        det = detector
        thr = det.threshold if threshold is None else threshold
        pp = {**DEFAULT_POSTPROCESS, **det.postprocess, **pp_override}
        if len(files) == 1 and args.out:
            out_dir = out_base
        else:
            suffix = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:6]
            out_dir = out_base / f"{_safe_name(path.stem)}_{suffix}"
        try:
            result = _run_one(
                path, det, cfg, out_dir,
                threshold=thr, threshold_source=threshold_source,
                postprocess=pp, postprocess_source=pp_source,
                fmax_plot=args.fmax_plot, clip_context=args.clip_context,
                records=records, anns=anns,
                manifest_path=Path(args.manifest) if args.manifest else None,
                reviews_by_id=reviews_by_id,
            )
        except Exception as exc:
            print(f"{path.name}: 跳过（{exc}）", file=sys.stderr)
            continue
        results.append(result)
        print(f"{path.name}: {result['events']} 事件 -> {result['report']}")
    return 0 if len(results) == len(files) else 1


# ------------------------------------------------------------------- review
def cmd_review_export(args) -> int:
    payload = json.loads(Path(args.events).read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else Path(args.events).parent / "review.csv"
    rows = export_template(payload, out, out.with_suffix(".json"))
    print(f"已导出审核模板：{out}（{len(rows)} 个事件）")
    return 0


def cmd_review_import(args) -> int:
    out = Path(args.out)
    rows = import_reviews(args.review, out, args.json)
    print(f"已导入/合并 {len(rows)} 条审核记录 -> {out}")
    print(json.dumps(review_summary(rows), ensure_ascii=False))
    if args.emit_annotations:
        anns = reviews_to_annotations(rows)
        write_annotations(anns, args.emit_annotations)
        print(f"已生成 L1 反馈区间：{args.emit_annotations}（{len(anns)} 条）")
        print("注意：anomaly+nonfault 仍按 anomaly 正例写入；只有 l1_label=normal 才成为 normal。")
    return 0


# -------------------------------------------------------------------- probe
def cmd_probe(args) -> int:
    path = Path(args.input)
    try:
        info = wav_info(path)
        x, sr = load_wav(path)
    except Exception as exc:
        print(f"无法读取 {path}: {exc}", file=sys.stderr)
        return 2
    rms = 20 * math.log10(float(np.sqrt(np.mean(x**2))) + 1e-12) if len(x) else None
    payload = {
        "file": str(path),
        "duration_s": info.duration_s,
        "sample_rate": info.sample_rate,
        "channels": info.channels,
        "sample_width": info.sample_width,
        "n_frames": info.n_frames,
        "peak": float(np.max(np.abs(x))) if len(x) else None,
        "rms_dbfs": rms,
    }
    if args.json:
        print(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print(f"{path.name}")
        print(f"  sr={info.sample_rate} dur={info.duration_s:.6f}s "
              f"ch={info.channels} peak={payload['peak']:.4f} rms={rms:.1f} dBFS")
    return 0


# ------------------------------------------------------------------- parser
def _add_feature_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sr", type=int, default=None)
    p.add_argument("--n-fft", dest="n_fft", type=int, default=None)
    p.add_argument("--hop", type=int, default=None)
    p.add_argument("--n-mels", dest="n_mels", type=int, default=None)
    p.add_argument("--fmin", type=float, default=None)
    p.add_argument("--fmax", type=float, default=None)
    p.add_argument("--target-step", dest="target_step", type=float, default=None)
    p.add_argument("--context", dest="context_s", type=float, default=None)
    p.add_argument("--normalize", choices=["none", "per_file"], default=None)


def _add_data_args(p: argparse.ArgumentParser, required: bool = True) -> None:
    p.add_argument("--manifest", required=required)
    p.add_argument("--annotations", required=required)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wind_audio", description="风机音频两级检测与人工审核增量学习")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("probe")
    q.add_argument("--input", required=True)
    q.add_argument("--json", action="store_true")
    q.set_defaults(func=cmd_probe)

    c = sub.add_parser("check", help="数据清单/标注检查")
    _add_data_args(c)
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_check)

    imp = sub.add_parser("import-simple", help="只有音频+区间标注：自动生成 manifest")
    imp.add_argument("--audio-dir", required=True)
    imp.add_argument("--labels", required=True, help="CSV/JSON：file,start_s,end_s,label[,group,fault_type,note]")
    imp.add_argument("--manifest", default="data/manifest.csv")
    imp.add_argument("--annotations", default="data/annotations.csv")
    imp.add_argument("--assume-unlabeled-normal", action="store_true",
                     help="声明所有文件都已完整审核；未标注区域推导为 normal（默认 unknown）")
    imp.add_argument("--reviewed-list", default=None,
                     help="每行一个文件名/相对路径，表示该文件整段已审核（未标注部分为 normal）")
    imp.set_defaults(func=cmd_import_simple)

    m = sub.add_parser("migrate-labels", help="迁移旧 labels.csv（不修改原文件）")
    m.add_argument("--labels", default="data/labels.csv")
    m.add_argument("--audio-dir", default="data/anomaly")
    m.add_argument("--manifest", default="data/manifest.csv")
    m.add_argument("--annotations", default="data/annotations.csv")
    m.set_defaults(func=cmd_migrate)

    s = sub.add_parser("split", help="按录音/批次/设备分组划分")
    _add_data_args(s)
    s.add_argument("--out", default="data/splits.json")
    s.add_argument("--group-by", choices=["source", "device", "time", "audio"], default="source")
    s.add_argument("--ratios", default="0.6,0.2,0.2")
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_split)

    t = sub.add_parser("train", help="训练 L1 候选检测器")
    _add_data_args(t)
    t.add_argument("--splits", default="data/splits.json")
    t.add_argument("--detector", choices=["baseline", "supervised"], default="baseline")
    t.add_argument("--out", default="models/l1_v2")
    t.add_argument("--version", default="l1_v2")
    t.add_argument("--threshold", type=float, default=5.0)
    t.add_argument("--target-recall", type=float, default=0.95)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--max-total-samples", type=int, default=50000)
    t.add_argument("--max-samples-per-file", type=int, default=5000)
    t.add_argument("--cache-dir", default="cache/features")
    t.add_argument("--no-calibrate", action="store_true")
    _add_feature_args(t)
    t.set_defaults(func=cmd_train)

    cal = sub.add_parser("calibrate", help="在验证集上重选阈值与后处理")
    cal.add_argument("--model", required=True)
    _add_data_args(cal)
    cal.add_argument("--splits", default="data/splits.json")
    cal.add_argument("--split", choices=["val"], default="val")
    cal.add_argument("--target-recall", type=float, default=0.95)
    cal.set_defaults(func=cmd_calibrate)

    ev = sub.add_parser("evaluate", help="在冻结模型上评估")
    ev.add_argument("--model", required=True)
    _add_data_args(ev)
    ev.add_argument("--splits", default="data/splits.json")
    ev.add_argument("--split", choices=["train", "val", "test"], default="test")
    ev.add_argument("--out", default="reports/eval_test")
    ev.add_argument("--threshold", type=float, default=None)
    ev.add_argument("--min-dur", dest="min_duration", type=float, default=None)
    ev.add_argument("--merge-gap", dest="merge_gap", type=float, default=None)
    ev.add_argument("--low-ratio", dest="low_ratio", type=float, default=None)
    ev.add_argument("--match-iou", type=float, default=0.1)
    ev.set_defaults(func=cmd_evaluate)

    r = sub.add_parser("run", help="单条/批量推理并生成报告")
    r.add_argument("--input", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--out", default=None)
    r.add_argument("--manifest", default=None)
    r.add_argument("--annotations", default=None)
    r.add_argument("--threshold", type=float, default=None)
    r.add_argument("--min-dur", dest="min_dur", type=float, default=None)
    r.add_argument("--merge-gap", dest="merge_gap", type=float, default=None)
    r.add_argument("--low-ratio", type=float, default=None)
    r.add_argument("--fmax-plot", type=float, default=20000.0)
    r.add_argument("--clip-context", type=float, default=3.0)
    r.add_argument("--review", default=None, help="可选：已导入的 reviews.csv，用于报告显示人工结论")
    _add_feature_args(r)
    r.set_defaults(func=cmd_run)

    rex = sub.add_parser("review-export", help="从 events.json 导出审核模板")
    rex.add_argument("--events", required=True)
    rex.add_argument("--out", default=None)
    rex.set_defaults(func=cmd_review_export)

    rim = sub.add_parser("review-import", help="导入/合并人工审核 CSV")
    rim.add_argument("--review", required=True)
    rim.add_argument("--out", default="data/reviews/reviews.csv")
    rim.add_argument("--json", default=None)
    rim.add_argument("--emit-annotations", default=None)
    rim.set_defaults(func=cmd_review_import)
    from .l2.cli import register
    register(sub)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
