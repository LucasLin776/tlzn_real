"""Separate L1 detection (audio only) and testing (audio + labels) interfaces."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .audio import load_wav, wav_info, write_wav
from .data import (
    AudioRecord,
    Annotation,
    anomaly_events,
    dataset_summary,
    import_simple,
    read_annotations,
    read_manifest,
    validate_dataset,
    write_annotations,
)
from .detectors import SpectralChangeDetector, StabilityDetector, load_detector
from .normal import NormalConfig, NormalModelDetector
from .stability import StabilityConfig
from .state import EventConfig
from .evaluate import evaluate_detector, write_eval_outputs
from .features import FeatureConfig, extract_features
from .render import score_png, spectrogram_png, waveform_png
from .report import write_report, write_batch_report
from .review import (
    export_template,
    import_reviews,
    review_summary,
    reviews_to_annotations,
)
from .splits import assert_no_leakage, load_splits, make_splits, save_splits
from .stability import audio_quality


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


# Command-line flags that only make sense for one detector generation.
LEGACY_ONLY_ARGS = {"threshold", "merge_gap", "low_ratio", "reference_s", "release_db",
                    "quiet_s", "settle_s", "stability_db", "min_prominence",
                    "relative_floor_db"}
V3_ONLY_ARGS = {"trigger_change_z", "trigger_exceed", "trigger_frames", "slow_ratio_trigger",
                "slow_frames", "slow_min_duration", "confirm_s", "stable_s", "noise_s", "band_z",
                "init_s", "backdate_s", "fast_s", "drift_s", "sigma_floor_db", "sigma_cap_db",
                "fluct_scale", "drift_scale", "stable_band_frac", "active_range_db",
                "weight_exp", "slow_scale", "excess_scale"}


def _provided(args) -> dict:
    return {k: v for k, v in vars(args).items() if v is not None}


def _detector_from_args(args):
    """Build a detector from config + CLI overrides, rejecting mismatched flags."""
    base = load_detector(args.config)
    cfg = base.feature_config.to_dict()
    for key in cfg:
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    provided = _provided(args)
    if isinstance(base, NormalModelDetector):
        clash = sorted(set(provided) & ((LEGACY_ONLY_ARGS - {"threshold", "merge_gap"}) | V3_ONLY_ARGS))
        if clash:
            raise ValueError(f"参数 {clash} 不适用于正常样本模型")
        base.feature_config.assert_compatible(FeatureConfig.from_dict(cfg))
        settings = base.normal.to_dict()
        for arg, key in (("min_dur", "min_duration"), ("merge_gap", "merge_gap"), ("padding", "padding")):
            if getattr(args, arg, None) is not None:
                settings[key] = getattr(args, arg)
        return NormalModelDetector(base.scales, base.threshold if args.threshold is None else args.threshold,
                                   base.feature_config, NormalConfig(**settings), base.training)
    if isinstance(base, SpectralChangeDetector):
        clash = sorted(set(provided) & V3_ONLY_ARGS)
        if clash:
            raise ValueError(f"参数 {clash} 只适用于 spectral_stability_v3；当前配置是旧版 spectral_event_v2")
        pp = dict(base.postprocess)
        for arg, key in (("min_dur", "min_duration"), ("merge_gap", "merge_gap"),
                         ("low_ratio", "low_ratio"), ("padding", "padding"),
                         ("reference_s", "reference_s"), ("release_db", "release_db"),
                         ("quiet_s", "quiet_s"), ("settle_s", "settle_s"),
                         ("stability_db", "stability_db"), ("min_prominence", "min_prominence"),
                         ("relative_floor_db", "relative_floor_db")):
            value = getattr(args, arg, None)
            if value is not None:
                pp[key] = value
        threshold = base.threshold if args.threshold is None else args.threshold
        return SpectralChangeDetector(FeatureConfig.from_dict(cfg), threshold, pp)
    clash = sorted(set(provided) & LEGACY_ONLY_ARGS)
    if clash:
        raise ValueError(f"参数 {clash} 只适用于旧版 spectral_event_v2；"
                         "新版请使用 --trigger-change-z / --confirm-s 等参数")
    stability, events = {}, {}
    for name, target in ((("fast_s", "noise_s", "stable_s", "drift_s", "band_z",
                           "sigma_floor_db", "sigma_cap_db", "fluct_scale", "drift_scale",
                           "stable_band_frac", "active_range_db", "weight_exp",
                           "slow_scale", "excess_scale"), stability),
                         (("trigger_change_z", "trigger_exceed", "trigger_frames",
                           "slow_ratio_trigger", "slow_frames", "slow_min_duration",
                           "confirm_s", "init_s", "backdate_s"), events)):
        for key in name:
            value = getattr(args, key, None)
            if value is not None:
                target[key] = value
    if args.min_dur is not None:
        events["min_duration"] = args.min_dur
    if args.padding is not None:
        events["padding"] = args.padding
    return StabilityDetector(FeatureConfig.from_dict(cfg),
                             StabilityConfig(**{**base.stability.to_dict(), **stability}),
                             EventConfig(**{**base.events.to_dict(), **events}))


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
        "runtime": {"numpy": np.__version__},
        "l1_requires_training": False,
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
            print("\n评估与 L2 数据提示（不阻止 L1 直接推理）：")
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
    print(f"已导入 {len(records)} 个音频，{len(anns)} 条标注")
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


# ---------------------------------------------------------------- evaluate
def cmd_evaluate(args) -> int:
    detector = _detector_from_args(args)
    out = Path(args.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("评估输出目录非空，请使用新目录，避免覆盖结果或标注")
    if args.audio_dir:
        if not args.labels or args.annotations:
            raise ValueError("--audio-dir 必须搭配 --labels，不能同时传 --annotations")
        audio_dir, labels_path = Path(args.audio_dir), Path(args.labels)
        if not audio_dir.is_dir():
            raise ValueError(f"音频目录不存在或不是目录（--audio-dir）: {audio_dir}")
        if args.labels_are_stable and args.assume_unlabeled_normal:
            raise ValueError("--labels-are-stable 与 --assume-unlabeled-normal 语义相反，不能同时使用")
        if args.labels_are_stable and not labels_path.is_dir():
            raise ValueError("--labels-are-stable 目前只支持 Audition 标注目录（--labels 传目录）")
        if not labels_path.exists():
            raise ValueError(f"标注路径不存在（--labels）: {labels_path}；请检查目录名称和层级顺序")
        if not (labels_path.is_dir() or labels_path.is_file()):
            raise ValueError(f"标注路径必须是文件或目录（--labels）: {labels_path}")
        manifest = out / "manifest.csv"
        if labels_path.is_dir():
            from .markers import import_markers
            records, anns, warnings = import_markers(
                args.audio_dir, args.labels, manifest, out / "annotations.csv",
                pairing=args.pairing, assume_unlabeled_normal=args.assume_unlabeled_normal,
                stable_markers=args.labels_are_stable)
            if not args.labels_are_stable:
                names = [a.fault_type for a in anns if a.fault_type]
                calm = sum(1 for n in names if "正常" in n or "normal" in n.lower())
                if names and calm * 2 >= len(names):
                    warnings.append(
                        f"标注中有 {calm}/{len(names)} 个名称含「正常」——若这批标的是平稳区间，"
                        "请加 --labels-are-stable，否则平稳段会被当成异常，指标会整体反过来")
        else:
            if args.pairing != "stem":
                raise ValueError("--pairing 仅用于标注目录")
            records, anns, warnings = import_simple(
                args.audio_dir, args.labels, manifest, out / "annotations.csv",
                assume_unlabeled_normal=args.assume_unlabeled_normal)
    else:
        if not args.annotations or args.labels or args.assume_unlabeled_normal or args.pairing != "stem":
            raise ValueError("--manifest 必须搭配 --annotations；导入选项仅用于 --audio-dir")
        manifest = Path(args.manifest)
        records, anns, errors, warnings = _load_dataset(args)
        _require_clean(errors)
    if args.splits:
        split = load_splits(args.splits)
        assert_no_leakage(split)
        audio_ids = split.ids(args.split)
    else:
        audio_ids = [r.audio_id for r in records]
    from .eval_plots import ComparisonReport, natural_key
    audio_ids = sorted(audio_ids, key=natural_key)
    report_context = (nullcontext(None) if args.no_plots else
                      ComparisonReport(out, cluster_gap=args.plot_merge_gap, font_path=args.plot_font))
    with report_context as report:
        result = evaluate_detector(
            detector, records, anns, audio_ids, manifest, detector.feature_config,
            iou_threshold=args.match_iou, on_file=report.add if report else None,
        )
        if result["file_accuracy"] is None and result["time_accuracy"] is None:
            raise ValueError("没有可评估标签；请提供文件标签、精确区间，或确认完整审核后设置 --assume-unlabeled-normal")
    if report is not None:
        result["visual_report"] = report.metadata()
    result["detector_config"] = detector.to_dict()
    result["mode"] = "l1-test"
    result["selection"] = args.split if args.splits else "all_manifest"
    result["input"] = {"manifest": str(manifest.resolve()), "labels": args.labels,
                       "annotations": args.annotations, "pairing": args.pairing if args.audio_dir else None,
                       "assume_unlabeled_normal": args.assume_unlabeled_normal,
                       "labels_are_stable": bool(getattr(args, "labels_are_stable", False))}
    result["warnings"] = warnings
    metrics, brief = write_eval_outputs(result, args.out)
    from .evaluate import _percent
    print(f"文件准确率={_percent(result['file_accuracy'])}；时长准确率={_percent(result['time_accuracy'])}")
    print(f"L1 窗口评估：GT={result['n_gt_events']}，预测={result['n_pred_events']}")
    print(f"召回={result['event_recall']}；精确率={result['event_precision']}；"
          f"FA/h={result['fa_per_hour']}；覆盖={result['coverage']}")
    if not result["n_gt_events"]:
        print("没有精确异常区间，不能计算召回；请勿把整段粗标当精确窗口。")
    for warning in warnings:
        print(f"提示: {warning}")
    print(f"输出: {metrics}, {brief}, {out / 'per_file.csv'}, {out / 'events.csv'}")
    if report is not None:
        print(f"对照图: {out / 'plots'}；汇总 PDF: {report.final_pdf}（{len(report.pages)} 页）")
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
        "trigger_start_s", "trigger_end_s", "window_path",
        "boundary_reason", "right_censored", "peak_activity_db", "peak_prominence",
        "peak_level_dbfs", "level_floor_dbfs",
        "peak_change_z", "mean_change_z", "peak_exceed", "peak_slow_ratio",
        "stable_confirmed_s", "trigger_change_z", "trigger_exceed",
        "clip_path", "clip_start_s", "clip_end_s",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for ev in events:
            clip = ev.get("clip") or {}
            row = {k: ev.get(k) for k in fields}
            row["window_path"] = (ev.get("window") or {}).get("path")
            row["clip_path"] = clip.get("path")
            row["clip_start_s"] = clip.get("start_s")
            row["clip_end_s"] = clip.get("end_s")
            writer.writerow(row)


def settings_text(detector) -> list[tuple[str, str]]:
    """Human-readable effective settings for the HTML report."""
    if isinstance(detector, NormalModelDetector):
        cfg = detector.normal
        return [("学习方式", "只学习已标正常区间的频谱时间波动"),
                ("正常偏离阈值", f"{detector.threshold:.3g}（不是概率）"),
                ("轻微噪音过滤", f"连续 {cfg.enter_s:g} s，最短 {cfg.min_duration:g} s"),
                ("正常恢复 / 合并间隔", f"{cfg.release_s:g} s / {cfg.merge_gap:g} s"),
                ("特征尺度 dB", ", ".join(f"{v:.2f}" for v in detector.scales))]
    if isinstance(detector, SpectralChangeDetector):
        pp = detector.postprocess
        return [("触发阈值", f"{detector.threshold:g} dB"),
                ("持续活动释放", f"{pp['release_db']:g} dB"),
                ("背景恢复确认", f"{pp['quiet_s']:g} s"),
                ("新工况确认", f"{pp['settle_s']:g} s"),
                ("显著性/响度过滤", f"{pp['min_prominence']:g} / {pp['relative_floor_db']:g} dB")]
    st, ev = detector.stability, detector.events
    return [("快速变化触发", f"{ev.trigger_change_z:g} σ 且超容差频带 ≥ {ev.trigger_exceed:.0%}"),
            ("慢变化触发", f"{ev.slow_ratio_trigger:g} 倍容差，持续 {ev.slow_frames} 帧"),
            ("局部噪声尺度", f"{st.noise_s:g} s 窗，{st.sigma_floor_db:g}–{st.sigma_cap_db:g} dB"),
            ("稳定判据", f"{st.stable_band_frac:.0%} 频带在容差内，观察窗 {st.stable_s:g} s"),
            ("稳定确认", f"{ev.confirm_s:g} s"),
            ("最短事件 / 边界补充", f"{ev.min_duration:g} s / {ev.padding:g} s")]


def _run_one(
    path: Path,
    detector,
    cfg: FeatureConfig,
    out_dir: Path,
    *,
    threshold: float,
    threshold_source: str,
    threshold_unit: str,
    postprocess: dict,
    postprocess_source: str,
    fmax_plot: float,
    clip_context: float,
) -> dict:
    """Production inference accepts waveform paths and configuration only."""
    if clip_context < 0 or not np.isfinite(clip_context):
        raise ValueError("试听上下文必须为有限非负数")
    x, sr = load_wav(path, target_sr=cfg.sr)
    if len(x) == 0:
        raise ValueError(f"空音频: {path}")
    duration = len(x) / sr
    audio_id = path.stem
    bundle = extract_features(x, sr, cfg, audio_id)
    events, traces = detector.analyze(bundle, threshold=threshold, postprocess=postprocess)
    from .l2.common import digest, file_hash
    audio_hash = file_hash(path)
    for ev in events:
        model_identity = (detector.to_dict() if isinstance(detector, NormalModelDetector)
                          else detector.version)
        ev["event_id"] = audio_id + "_" + digest(
            [audio_hash, ev["start_s"], ev["end_s"], model_identity,
             cfg.to_dict(), threshold, postprocess])[:20]
    # Keep the existing report schema, but never load labels in detection mode.
    segments = traces.get("stable_segments") or []
    regions = [{"start_s": s["start_s"], "end_s": s["end_s"], "label": "normal"}
               for s in segments]
    gt = []

    if (out_dir / "review.csv").exists():
        raise ValueError("输出目录已有审核记录，请使用新目录，避免覆盖人工标注")
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_rel = _copy_audio(path, out_dir, audio_id)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    windows_dir = out_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    for ev in events:
        # Long recording names must never truncate the distinguishing event ID.
        wav_name = f"event_{digest(ev['event_id'])[:20]}.wav"
        window = _make_clip(x, sr, ev["start_s"], ev["end_s"],
                            windows_dir / wav_name, 0.0, duration)
        window["path"] = f"windows/{window['path']}"
        ev["window"] = window
        clip = _make_clip(x, sr, ev["start_s"], ev["end_s"], clips_dir / wav_name, clip_context, duration)
        clip["path"] = f"clips/{clip['path']}"
        ev["clip"] = clip

    waveform_png(x, duration, events, regions, out_dir / "waveform.png")
    spectrogram_png(
        bundle.spec_db, bundle.freqs, bundle.mel_times, duration, events, regions,
        out_dir / "spectrogram.png", f_max=min(fmax_plot, cfg.fmax, sr / 2),
    )
    quality = audio_quality(x, sr)
    curves = traces.get("curves") or []
    chart_blocks = []
    for curve in curves:
        image = f"{curve['key']}.png"
        score_png(bundle.times, curve["values"], curve["threshold"], events, regions,
                  duration, out_dir / image)
        chart_blocks.append({"image": image, "title": curve["title"],
                             "caption": curve.get("note", "")})
    extra = [("event_active", traces["active"].astype(int))]
    for key in ("stable", "state", "level_dbfs", "sigma_db"):
        if key in traces:
            values = traces[key]
            extra.append((key, values.astype(int) if key == "stable" else values))
    columns = ["time_s"]
    for curve in curves:
        columns += [curve["key"], curve["key"] + "_threshold"]
    columns += [key for key, _ in extra]
    with (out_dir / "scores.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        series = [bundle.times]
        for curve in curves:
            series += [curve["values"], np.full(len(bundle.times), curve["threshold"])]
        series += [values for _, values in extra]
        writer.writerows(zip(*series))

    effective_config = detector.to_dict()
    if isinstance(detector, SpectralChangeDetector):
        effective_config = {**effective_config, "threshold": threshold, "postprocess": postprocess}
    _write_json(out_dir / "l1_config.json", effective_config)

    diagnostics = {"quality": quality,
                   "output_mode": traces.get("output_mode", getattr(getattr(detector, "events", None),
                                                                      "output_mode", "anomaly")),
                   "anomaly_source": traces.get("anomaly_source",
                                                "complement_of_stable" if segments else "detected_windows")}
    for key in ("rejected_triggers", "near_miss", "context_insufficient",
                "context_required_s", "available_s"):
        if key in traces:
            diagnostics["rejected_candidates" if key == "rejected_triggers" else key] = traces[key]
    if "level_floor_dbfs" in traces:
        diagnostics["level_floor_dbfs"] = traces["level_floor_dbfs"]

    payload = {
        "schema_version": "4.0",
        "mode": "l1-detect",
        "audio": {
            "id": audio_id,
            "file": path.name,
            "path": str(path.resolve()),
            "sha256": audio_hash,
            "duration_s": round(duration, 6),
            "sr": sr,
            "analyzed_fmax_hz": cfg.fmax,
        },
        "model": {**{k: v for k, v in effective_config.items() if k != "schema_version"},
                  "threshold_source": threshold_source,
                  "postprocess_source": postprocess_source,
                  "requires_training": bool(getattr(detector, "requires_training", False))},
        "events": events,
        "stable_segments": segments,
        "diagnostics": diagnostics,
        "annotations": {"regions": [], "gt_events": [], "source": None},
    }
    _write_json(out_dir / "events.json", payload)
    _events_csv(events, out_dir / "events.csv")
    if segments:
        with (out_dir / "stable_segments.csv").open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["index", "start_s", "end_s", "duration_s", "coverage"])
            writer.writeheader()
            for index, seg in enumerate(segments, 1):
                writer.writerow({**seg, "index": index})

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
        threshold_unit=threshold_unit,
        feature_config=cfg.to_dict(),
        postprocess=postprocess,
        settings=settings_text(detector),
        events=events,
        regions=regions,
        gt_events=gt,
        audio_relpath=audio_rel,
        review_relpath="review.csv",
        charts=chart_blocks,
        quality=quality,
        stable_summary=({"segments": len(segments),
                         "stable_s": round(sum(s["duration_s"] for s in segments), 2),
                         "coverage": round(sum(s["duration_s"] for s in segments) / duration, 3)}
                        if segments else None),
    )
    return {"audio_id": audio_id, "file": path.name, "source": str(path.resolve()),
            "audio_sha256": audio_hash, "duration_s": duration,
            "events": len(events), "alarm_s": sum(e["duration_s"] for e in events),
            "windows": [{"start_s": e["start_s"], "end_s": e["end_s"],
                         "right_censored": e["right_censored"]} for e in events],
            "filtered_candidates": len(traces.get("rejected_triggers", [])),
            "near_miss": len(traces.get("near_miss", [])),
            "clip_ratio": quality["clip_ratio"],
            "stable_segments": len(segments),
            "stable_s": round(sum(s["duration_s"] for s in segments), 2),
            "report": str(report), "out": str(out_dir)}


def cmd_run(args) -> int:
    input_path = Path(args.input)
    if input_path.is_dir():
        files = sorted(p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")
    elif input_path.is_file():
        files = [input_path]
    else:
        print(f"输入不存在: {input_path}", file=sys.stderr)
        return 2
    if not files:
        print(f"没有 wav: {input_path}", file=sys.stderr)
        return 2

    detector = _detector_from_args(args)
    cfg = detector.feature_config

    if isinstance(detector, NormalModelDetector):
        threshold, pp = detector.threshold, {}
        threshold_unit = "正常偏离倍数"
        threshold_source = "cli" if args.threshold is not None else "normal_calibration"
    elif isinstance(detector, SpectralChangeDetector):
        threshold = detector.threshold
        pp = detector.postprocess
        threshold_unit = "dB change"
        threshold_source = "cli" if args.threshold is not None else ("config" if args.config else "default")
    else:
        threshold = detector.events.trigger_change_z
        pp = {}
        threshold_unit = "噪声σ"
        threshold_source = "cli" if args.trigger_change_z is not None else ("config" if args.config else "default")
    pp_source = "effective_config"

    out_base = Path(args.out) if args.out else Path("runs")
    results, failures = [], []
    for path in files:
        det = detector
        thr = threshold
        if len(files) == 1 and args.out:
            out_dir = out_base
        else:
            suffix = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:6]
            out_dir = out_base / f"{_safe_name(path.stem)}_{suffix}"
        try:
            result = _run_one(
                path, det, cfg, out_dir,
                threshold=thr, threshold_source=threshold_source,
                threshold_unit=threshold_unit,
                postprocess=pp, postprocess_source=pp_source,
                fmax_plot=args.fmax_plot, clip_context=args.clip_context,
            )
        except Exception as exc:
            print(f"{path.name}: 跳过（{exc}）", file=sys.stderr)
            failures.append({"file": path.name, "source": str(path.resolve()), "error": str(exc)})
            continue
        results.append(result)
        print(f"{path.name}: {result['events']} 事件 -> {result['report']}")
    if input_path.is_dir():
        out_base.mkdir(parents=True, exist_ok=True)
        summary = {"mode": "l1-detect", "files_requested": len(files), "files_succeeded": len(results), "failures": failures,
                   "detector_config": detector.to_dict(), "files": results,
                   "total_events": sum(r["events"] for r in results),
                   "total_duration_s": sum(r["duration_s"] for r in results),
                   "total_alarm_s": sum(r["alarm_s"] for r in results),
                   "note": "检测输出，不是准确率验证；无精确标注不能计算召回或误报率。"}
        _write_json(out_base / "summary.json", summary)
        print(f"批量总览: {write_batch_report(out_base, summary)}")
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


# ------------------------------------------------------------ normal learning
def cmd_train_normal(args) -> int:
    from .normal_fit import fit_normal
    cfg = NormalConfig(enter_s=args.enter_s, release_s=args.normal_release_s,
                       min_duration=args.min_dur, merge_gap=args.merge_gap)
    detector, report = fit_normal(
        args.audio_dir, args.labels, args.out, pairing=args.pairing,
        fully_reviewed=args.fully_reviewed_normal, normal=cfg, alpha=args.alpha,
        calibration_fraction=args.calibration_fraction, test_fraction=args.test_fraction)
    out = Path(args.out)
    _write_json(out / "model.json", detector.to_dict())
    _write_json(out / "training_report.json", report)
    print(f"正常模型已保存 -> {out / 'model.json'}")
    print("只使用训练/校准集的正常区间；留出检查未用于拟合或选择阈值。")
    for group, metrics in report["metrics"].items():
        tm = metrics["time_metrics"]
        print(f"  {group}: 时长准确率={metrics['time_accuracy']} "
              f"异常时长召回={tm['recall']} 精确率={tm['precision']}")
    return 0


# -------------------------------------------------------------------- probe
def cmd_fit(args) -> int:
    from .fit import fit
    base = load_detector(args.config) if args.config else StabilityDetector()
    if not isinstance(base, StabilityDetector):
        raise ValueError("l1-fit 只适用于 spectral_stability_v3 配置")
    detector, report = fit(args.audio_dir, args.labels, base=base,
                           val_fraction=args.val_fraction, alpha=args.alpha,
                           alpha_stable=args.alpha_stable)
    payload = detector.to_dict()
    payload["fit"] = report.to_dict()
    out = _write_json(args.out, payload)
    if args.report:
        _write_json(args.report, report.to_dict())
    v = report.validation
    print(f"拟合完成 -> {out}")
    print(f"  样本: 拟合 {len(report.files['fit'])} 条 / 验证 {len(report.files['validation'])} 条")
    print(f"  阈值: " + ", ".join(f"{k}={v:.3g}" for k, v in report.best["thresholds"].items()))
    print(f"  验证集: 时长 F1={v['time_f1']} acc={v['time_accuracy']} "
          f"R={v['time_recall']} P={v['time_precision']} 覆盖={v['coverage']} | "
          f"事件 F1={v['event_f1']} 预测 {v['n_pred']} 个")
    print(f"  拟合集(样本内): 事件 F1={report.train['event_f1']} 时长准确率={report.train['time_accuracy']}")
    return 0


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
def _add_detector_args(p):
    p.add_argument("--config", help="L1 JSON 配置或正常样本 model.json；不传仍使用 spectral_stability_v3")
    p.add_argument("--threshold", type=float, help="旧版的变化 dB 阈值 / 正常模型的偏离倍数（默认加载模型阈值）")
    p.add_argument("--min-dur", type=float, help="跟踪事件最短时长，默认 0.06 秒")
    p.add_argument("--merge-gap", type=float, help="相邻完整事件合并间隔，默认 0.35 秒")
    p.add_argument("--low-ratio", type=float, help="边缘种子退出/进入比值，非完整事件结束阈值，默认 0.6")
    p.add_argument("--padding", type=float, help="检测窗口两端补充，默认各 0.05 秒")
    p.add_argument("--reference-s", type=float, help="触发前局部背景长度，默认 0.6 秒")
    p.add_argument("--release-db", type=float, help="持续活动释放阈值，默认 3 dB；越低尾部越长")
    p.add_argument("--quiet-s", type=float, help="背景恢复需连续维持时长，默认 0.25 秒")
    p.add_argument("--settle-s", type=float, help="新稳定工况确认时长，默认 0.6 秒")
    p.add_argument("--stability-db", type=float, help="新工况的谱形/能量稳定容差，默认 1.5 dB")
    p.add_argument("--min-prominence", type=float, help="事件与背景的能量差异比例下限，默认 0.3；越大过滤越强")
    p.add_argument("--relative-floor-db", type=float, help="比录音强声参考低多少 dB 仍保留，默认 15；越小过滤越强")
    # --- spectral_stability_v3 -------------------------------------------------
    p.add_argument("--trigger-change-z", type=float,
                   help="变化强度触发阈值，单位=局部噪声σ，默认 3.0；越大越不敏感")
    p.add_argument("--trigger-exceed", type=float,
                   help="触发所需的最小超容差频带比例（0–1），默认 0.15")
    p.add_argument("--trigger-frames", type=int, help="快速变化需连续帧数，默认 3")
    p.add_argument("--slow-ratio-trigger", type=float, help="慢变化触发阈值（容差倍数），默认 2.5")
    p.add_argument("--slow-frames", type=int, help="慢变化需连续帧数，默认 10")
    p.add_argument("--confirm-s", type=float, help="结束事件所需的稳定确认时长，默认 0.8 秒")
    p.add_argument("--stable-s", type=float, help="判断稳定的观察窗，默认 0.6 秒")
    p.add_argument("--noise-s", type=float, help="局部噪声尺度的估计窗，默认 2 秒")
    p.add_argument("--band-z", type=float, help="单频带超容差阈值（噪声σ），默认 3")
    p.add_argument("--fast-s", type=float, help="快速变化比较跨度，默认 0.1 秒")
    p.add_argument("--drift-s", type=float, help="局部均值漂移比较半步长，默认 0.4 秒")
    p.add_argument("--sigma-floor-db", type=float, help="噪声尺度下限，默认 0.35 dB")
    p.add_argument("--sigma-cap-db", type=float, help="噪声尺度上限，默认 6 dB")
    p.add_argument("--fluct-scale", type=float, help="允许波动 = 该倍数 x 局部σ，默认 1.2")
    p.add_argument("--drift-scale", type=float, help="允许漂移 = 该倍数 x 局部σ，默认 1.2")
    p.add_argument("--stable-band-frac", type=float, help="稳定所需频带比例，默认 0.75")
    p.add_argument("--active-range-db", type=float, help="比最强频带低多少 dB 以内才参与判定，默认 30")
    p.add_argument("--slow-min-duration", type=float,
                   help="仅由慢变化触发的事件最短时长，默认 0.5 秒")
    p.add_argument("--weight-exp", type=float,
                   help="频带权重指数（0=按频带计数），默认 0.25")
    p.add_argument("--slow-scale", type=float,
                   help="慢变化容差 = 该倍数 x 该频带自身慢波动，默认 2.0")
    p.add_argument("--excess-scale", type=float,
                   help="电平偏离容差 = 该倍数 x 残差自身波动，默认 3.0")
    p.add_argument("--init-s", type=float, help="开头用于建立噪声基准的时长，默认 0.5 秒")
    p.add_argument("--backdate-s", type=float, help="触发后向前回溯起点上限，默认 0.6 秒")
    for name in ("sr", "n-fft", "hop", "n-mels"):
        p.add_argument("--" + name, type=int)
    for name in ("fmin", "fmax", "target-step", "compare-s", "smooth-s", "dynamic-range-db"):
        p.add_argument("--" + name, type=float)


def _add_data_args(p: argparse.ArgumentParser, required: bool = True) -> None:
    p.add_argument("--manifest", required=required)
    p.add_argument("--annotations", required=required)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wind_audio", description="风机音频两级检测与人工审核增量学习")
    sub = p.add_subparsers(dest="cmd", required=True)

    normal = sub.add_parser("l1-train-normal", help="只从标注正常片段学习波动容差，输出可复用的正常模型")
    normal.add_argument("--audio-dir", required=True)
    normal.add_argument("--labels", required=True, help="正常区间标注目录，名称为 正常/normal（可带编号）")
    normal.add_argument("--out", required=True, help="新的训练输出目录，保存模型/报告/配对表")
    normal.add_argument("--pairing", choices=["stem", "sorted"], default="stem")
    normal.add_argument("--fully-reviewed-normal", action="store_true", help="确认所有正常区间均已标出，其余为异常")
    normal.add_argument("--alpha", type=float, default=0.01, help="正常校准帧的经验尾部比例，不是事件误报率保证")
    normal.add_argument("--calibration-fraction", type=float, default=0.2)
    normal.add_argument("--test-fraction", type=float, default=0.2)
    normal.add_argument("--enter-s", type=float, default=0.16, help="普通偏离需连续保持的秒数")
    normal.add_argument("--normal-release-s", type=float, default=0.6, help="正常恢复确认秒数，输出终点回退到恢复起点")
    normal.add_argument("--min-dur", type=float, default=0.2, help="普通偏离最短时长；强短事件保留")
    normal.add_argument("--merge-gap", type=float, default=0.35)
    normal.set_defaults(func=cmd_train_normal)

    f = sub.add_parser("l1-fit", help="用平稳区域标注重新标定 L1 v3 阈值并选出结构参数")
    f.add_argument("--audio-dir", required=True)
    f.add_argument("--labels", required=True, help="平稳区域标注目录（Name/Start/Duration）")
    f.add_argument("--config", help="结构参数模板，默认使用内置 v3 默认值")
    f.add_argument("--out", required=True, help="输出的 L1 配置文件")
    f.add_argument("--report", help="可选：拟合报告 JSON")
    f.add_argument("--val-fraction", type=float, default=0.3, help="按时间顺序留出的验证比例，默认 0.3")
    f.add_argument("--alpha", type=float, default=1e-3, help="平稳帧上的目标误触发率，默认 1e-3")
    f.add_argument("--alpha-stable", type=float, default=1e-2,
                   help="平稳帧被误判为不稳定的目标比例，默认 1e-2")
    f.set_defaults(func=cmd_fit)

    q = sub.add_parser("probe")
    q.add_argument("--input", required=True)
    q.add_argument("--json", action="store_true")
    q.set_defaults(func=cmd_probe)

    c = sub.add_parser("check", help="数据清单/标注检查")
    _add_data_args(c)
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_check)

    imp = sub.add_parser("import-simple", help="音频+整段/区间标签：自动生成 manifest")
    imp.add_argument("--audio-dir", required=True)
    imp.add_argument("--labels", required=True, help="CSV/JSON：file,label[,start_s,end_s,precision,group]")
    imp.add_argument("--manifest", default="data/manifest.csv")
    imp.add_argument("--annotations", default="data/annotations.csv")
    imp.add_argument("--assume-unlabeled-normal", action="store_true",
                     help="声明所有文件都已完整审核；未标注区域推导为 normal（默认 unknown）")
    imp.add_argument("--reviewed-list", default=None,
                     help="每行一个文件名/相对路径，表示该文件整段已审核（未标注部分为 normal）")
    imp.set_defaults(func=cmd_import_simple)

    s = sub.add_parser("split", help="按录音/批次/设备分组划分")
    _add_data_args(s)
    s.add_argument("--out", default="data/splits.json")
    s.add_argument("--group-by", choices=["source", "device", "time", "audio"], default="source")
    s.add_argument("--ratios", default="0.6,0.2,0.2")
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_split)

    ev = sub.add_parser("l1-test", aliases=["evaluate"],
                        help="L1 测试版：音频 + 标签 → 检验器 → 准确率/召回率/逐条对比",
                        description="L1 测试版：必须提供音频和标签；标签仅用于检验，同检测版共用 L1 算法。")
    source = ev.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="已有标准音频清单，搭配 --annotations")
    source.add_argument("--audio-dir", help="WAV 目录，搭配 --labels")
    truth = ev.add_mutually_exclusive_group(required=True)
    truth.add_argument("--annotations", help="标准 annotations.csv，搭配 --manifest")
    truth.add_argument("--labels", help="简易 CSV/JSON 文件，或 Name/Start/Duration 标注目录，搭配 --audio-dir")
    ev.add_argument("--pairing", choices=["stem", "sorted"], default="stem",
                    help="标注目录配对：同名 stem；用户确认编号与音频排序一致才选 sorted")
    ev.add_argument("--assume-unlabeled-normal", action="store_true",
                    help="声明所有音频已完整审核，未标注部分及空标注均为正常")
    ev.add_argument("--labels-are-stable", action="store_true",
                    help="标注标的是平稳区间：区间内为正常，录音内其余时间按异常评估")
    _add_detector_args(ev)
    ev.add_argument("--splits", help="可选固定划分；不传则评估整个清单")
    ev.add_argument("--split", choices=["val", "test"], default="test")
    ev.add_argument("--out", default="reports/l1_test")
    ev.add_argument("--match-iou", type=float, default=0.1)
    ev.add_argument("--plot-merge-gap", type=float, default=.3,
                    help="辅助事件簇对比图的合并间隔，默认 0.3 秒；不改变标签和正式指标")
    ev.add_argument("--plot-font", help="可选中文 TTF/TTC 字体路径；默认自动寻找系统中文字体")
    ev.add_argument("--no-plots", action="store_true", help="仅输出数值评估，跳过逐音频图片和 PDF")
    ev.set_defaults(func=cmd_evaluate)

    r = sub.add_parser("l1-detect", aliases=["run"],
                       help="L1 检测版：仅原始音频 → 候选事件、切片、可试听报告",
                       description="L1 检测版：仅输入原始 WAV 或音频目录，不接收标签、清单或已审核记录。")
    r.add_argument("--input", required=True, help="原始 WAV 文件路径或音频目录（递归扫描 WAV）")
    _add_detector_args(r)
    r.add_argument("--out", help="检测结果目录，包含事件、音频切片和 HTML 报告")
    r.add_argument("--fmax-plot", type=float, default=20000)
    r.add_argument("--clip-context", type=float, default=1.0,
                   help="仅试听切片额外上下文，不改变检测窗口")
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
    try:
        return int(args.func(args) or 0)
    except (ValueError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
