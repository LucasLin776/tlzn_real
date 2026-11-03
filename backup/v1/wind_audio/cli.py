"""Command line entry points: train / run / probe."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .audio import load_wav, log_mel, stft
from .events import event_features, extract_events
from .l1 import NormalModel
from .render import score_png, spectrogram_png
from .report import write_report

DEFAULTS = dict(
    sr=44100,
    n_fft=2048,
    hop=441,
    n_mels=128,
    fmin=20.0,
    fmax=16000.0,
    smooth_frames=15,
)


def _feature_cfg(args: argparse.Namespace, base: dict | None = None) -> dict:
    cfg = dict(base or DEFAULTS)
    for key in ("sr", "n_fft", "hop", "n_mels", "fmin", "fmax", "smooth_frames"):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    return cfg


def _analyze(path: Path, cfg: dict):
    x, sr = load_wav(path, target_sr=cfg["sr"])
    mel_db, freqs, times = log_mel(
        x,
        sr,
        n_fft=cfg["n_fft"],
        hop=cfg["hop"],
        n_mels=cfg["n_mels"],
        fmin=cfg["fmin"],
        fmax=cfg["fmax"],
    )
    power = np.abs(stft(x, cfg["n_fft"], cfg["hop"])) ** 2
    return x, sr, mel_db, power, freqs, times


def _fit_demo(mel_db: np.ndarray, cfg: dict, threshold: float, version: str = "l1_demo") -> NormalModel:
    """Fit on the quietest 80% of frames — pipeline smoke test only."""
    frame_level = mel_db.mean(axis=1)
    cut = float(np.quantile(frame_level, 0.80))
    mask = frame_level <= cut
    cfg = dict(cfg)
    cfg["demo"] = True
    model = NormalModel(threshold_z=threshold, version=version, config=cfg)
    model.fit(mel_db, mask=mask)
    return model


def cmd_train(args: argparse.Namespace) -> int:
    cfg = _feature_cfg(args)
    files = sorted(Path(args.normal_dir).rglob("*.wav"))
    demo = False
    if not files:
        if not args.demo_from:
            print(
                f"没有在 {args.normal_dir} 找到 .wav。\n"
                "请放入正常样本，或先用 --demo-from <异常文件> 跑通链路（演示模型不可用于生产）。",
                file=sys.stderr,
            )
            return 2
        files = [Path(args.demo_from)]
        demo = True

    chunks = []
    for path in files:
        _, _, mel_db, _, _, _ = _analyze(path, cfg)
        chunks.append(mel_db)
        print(f"  {path.name}: {mel_db.shape[0]} frames")
    mel = np.concatenate(chunks, axis=0)

    version = args.version if not demo or args.version.endswith("demo") else f"{args.version}_demo"
    model = NormalModel(threshold_z=args.threshold, version=version, config=cfg)
    if demo:
        model = _fit_demo(mel, cfg, args.threshold, version=version)
    else:
        model.fit(mel)

    out = model.save(
        args.out,
        demo=demo,
        extra={
            "train_files": [p.name for p in files],
            "train_frames": int(mel.shape[0]),
        },
    )
    print(f"\n模型已保存: {out}")
    print(f"  版本={model.version}  训练帧={mel.shape[0]}  阈值={model.threshold_z}σ  demo={demo}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    inputs = (
        sorted(Path(args.input).rglob("*.wav"))
        if Path(args.input).is_dir()
        else [Path(args.input)]
    )
    if not inputs:
        print(f"没有找到 wav: {args.input}", file=sys.stderr)
        return 2

    if args.model:
        model = NormalModel.load(args.model)
        cfg = _feature_cfg(args, model.config)
    elif args.demo:
        model = None
        cfg = _feature_cfg(args)
    else:
        print("需要 --model <目录> 或 --demo（演示拟合）", file=sys.stderr)
        return 2

    out_base = Path(args.out) if args.out else Path("outputs")
    multi = len(inputs) > 1

    for path in inputs:
        out_dir = out_base / path.stem if multi else out_base
        out_dir.mkdir(parents=True, exist_ok=True)
        _, sr, mel_db, power, freqs, times = _analyze(path, cfg)

        used_model = model
        if used_model is None:
            used_model = _fit_demo(mel_db, cfg, args.threshold)
        threshold = float(used_model.threshold_z)
        demo_flag = bool(used_model.config.get("demo", False)) or used_model.version.endswith("demo")

        score = used_model.score(mel_db)
        frame_dt = cfg["hop"] / cfg["sr"]
        events = extract_events(
            score,
            frame_dt,
            thr_high=threshold,
            thr_low=threshold * 0.6,
            min_dur=args.min_dur,
            merge_gap=args.merge_gap,
        )
        feats = [event_features(power, freqs, score, ev, frame_dt) for ev in events]

        spectrogram_png(10.0 * np.log10(power + 1e-10), freqs, events, out_dir / "spectrogram.png", f_max=min(args.fmax_plot, cfg["sr"] / 2))
        score_png(score, events, threshold, out_dir / "score.png")

        payload = {
            "file": path.name,
            "duration_s": round(len(mel_db) * frame_dt, 3),
            "sr": sr,
            "model": used_model.version,
            "demo": demo_flag,
            "threshold_z": threshold,
            "events": feats,
        }
        (out_dir / "events.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        fields = [
            "start", "end", "duration", "peak_z", "mean_z", "centroid_hz", "flatness",
            "peak_freq_hz", "harmonicity_db", "share_0_200", "share_200_1k",
            "share_1k_4k", "share_4k_10k", "share_10k_22k", "onset_slope_z_per_s", "at_edge",
        ]
        with (out_dir / "events.csv").open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for row in feats:
                writer.writerow({k: row[k] for k in fields})

        report = write_report(
            out_dir,
            file_name=path.name,
            duration=len(mel_db) * frame_dt,
            sr=sr,
            model_version=used_model.version,
            threshold=threshold,
            demo=demo_flag,
            train_files=len(used_model.config.get("train_files", []) or []),
            events=feats,
        )

        print(f"\n{path.name}")
        print(f"  事件 {len(feats)} 个 · 阈值 {threshold:.1f}σ · 输出 {out_dir}")
        for i, ev in enumerate(feats, 1):
            print(
                f"  #{i} {ev['start']:6.2f}-{ev['end']:6.2f}s "
                f"peak={ev['peak_z']:5.1f} 质心={ev['centroid_hz']:6.0f}Hz "
                f"平坦度={ev['flatness']:.3f} 4-10k={ev['share_4k_10k']*100:4.1f}%"
            )
        print(f"  报告: {report}")

    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    path = Path(args.input)
    x, sr = load_wav(path)
    print(f"{path.name}")
    print(f"  sr={sr}  dur={len(x)/sr:.3f}s  peak={np.max(np.abs(x)):.4f}  "
          f"rms={20*np.log10(np.sqrt(np.mean(x**2))+1e-12):.1f} dBFS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wind_audio", description="风机音频异常检测 L1")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="用正常样本训练 L1")
    t.add_argument("--normal-dir", default="data/normal")
    t.add_argument("--out", default="models/l1_v0")
    t.add_argument("--version", default="l1_v0")
    t.add_argument("--threshold", type=float, default=5.0)
    t.add_argument("--demo-from", default=None, help="没有正常样本时，用该文件的安静段做演示拟合")
    _add_feature_args(t)
    t.set_defaults(func=cmd_train)

    r = sub.add_parser("run", help="对 wav 跑 L1 并出报告")
    r.add_argument("--input", required=True)
    r.add_argument("--model", default=None)
    r.add_argument("--demo", action="store_true")
    r.add_argument("--out", default=None)
    r.add_argument("--threshold", type=float, default=5.0)
    r.add_argument("--min-dur", type=float, default=0.2)
    r.add_argument("--merge-gap", type=float, default=0.3)
    r.add_argument("--fmax-plot", type=float, default=20000.0)
    _add_feature_args(r)
    r.set_defaults(func=cmd_run)

    q = sub.add_parser("probe", help="打印 wav 基本信息")
    q.add_argument("--input", required=True)
    q.set_defaults(func=cmd_probe)
    return p


def _add_feature_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sr", type=int, default=None)
    p.add_argument("--n-fft", dest="n_fft", type=int, default=None)
    p.add_argument("--hop", type=int, default=None)
    p.add_argument("--n-mels", dest="n_mels", type=int, default=None)
    p.add_argument("--fmin", type=float, default=None)
    p.add_argument("--fmax", type=float, default=None)
    p.add_argument("--smooth-frames", dest="smooth_frames", type=int, default=None)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
