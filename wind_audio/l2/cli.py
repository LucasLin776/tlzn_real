"""L2 workflow: reviewed events -> embeddings -> training -> inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import file_hash, read_json, write_json
from .data import Dataset, prepare
from .encoder import EncoderConfig, Features, PannsEncoder, embed_dataset
from .model import Model, evaluate, train


def cmd_prepare(a):
    previous = Dataset.load(a.previous) if a.previous else None
    data = prepare(a.reviews, a.manifest, a.splits, a.taxonomy, previous=previous)
    data.save(a.out)
    print(f"L2 数据快照: {a.out}，已审核事件 {len(data.samples)}")


def cmd_check(a):
    data = Dataset.load(a.dataset)
    data.check_audio()
    print({part: sum(r.split == part for r in data.samples) for part in ("train", "val", "test")})
    print("类别:", ", ".join(data.ids))


def cmd_embed(a):
    data = Dataset.load(a.dataset)
    config = EncoderConfig(window_seconds=a.window, stride_seconds=a.stride)
    encoder = PannsEncoder(a.checkpoint, config=config, device=a.device)
    features = embed_dataset(data, encoder)
    features.save(a.out)
    print(f"PANNs 特征: {a.out}，shape={features.X.shape}")


def cmd_train(a):
    if Path(a.out).exists():
        raise ValueError("请为模型选择尚不存在的新目录")
    data, features = Dataset.load(a.dataset), Features.load(a.features)
    data.check_audio()
    parent = Model.load(a.resume) if a.resume else None
    model = train(data, features, version=a.version, parent=parent,
                  epochs=a.epochs, lr=a.lr, memory_per_class=a.memory_per_class,
                  replay=not a.no_replay, distill=a.distill, full_retrain=a.full_retrain)
    model.save(a.out)
    write_json(Path(a.out) / "validation.json", evaluate(model, data, features, split="val"))
    active = [c["id"] for j, c in enumerate(data.classes, 1)
              if model.meta["calibration"][j]["active"]]
    print(f"模型: {a.out}；通过样本量门槛的类型: {active}")
    print("主干被冻结；更新分类头和回放库，不会自动删除 L1 告警。")


def cmd_evaluate(a):
    data = Dataset.load(a.dataset)
    data.check_audio()
    result = evaluate(Model.load(a.model), data, Features.load(a.features), a.split)
    write_json(a.out, result)
    print(f"L2 条件评估: {a.out}；macro F1={result['macro_f1']}")


def run_events(events_path, model_path, checkpoint, out, device="cpu", audio=None):
    from ..audio import load_wav
    payload = read_json(events_path)
    model = Model.load(model_path)
    source = Path(audio or payload["audio"]["path"]).resolve()
    expected = payload["audio"].get("sha256")
    if expected and file_hash(source) != expected:
        raise ValueError("输入音频与 L1 事件不匹配")
    events = payload.get("events", [])
    predictions = []
    if events:
        encoder = PannsEncoder(checkpoint, config=EncoderConfig(**model.meta["encoder"]), device=device)
        waveform, _ = load_wav(source, encoder.config.sample_rate)
        X = np.stack([encoder.encode(waveform, e["start_s"], e["end_s"]) for e in events])
        predictions = model.predict(X)
    for event, result in zip(events, predictions):
        event["l2"] = result
    payload["l2_model"] = {"version": model.meta["version"], "encoder": model.meta["encoder"]}
    write_json(out, payload)
    return payload


def cmd_run(a):
    payload = run_events(a.events, a.model, a.checkpoint, a.out, a.device, a.audio)
    print(f"L2 结果: {a.out}，{len(payload['events'])} 个候选事件")


def cmd_pipeline(a):
    from ..cli import main
    source = Path(a.input)
    if not source.is_file():
        raise ValueError("pipeline 接受单个 WAV；批量先使用 run，再逐份 l2-run")
    out = Path(a.out)
    result = main(["run", "--input", str(source), "--model", a.l1_model, "--out", str(out / "l1")])
    if result:
        return result
    run_events(out / "l1/events.json", a.l2_model, a.checkpoint, out / "events.json", a.device)
    print(f"两级结果: {out / 'events.json'}；审核表: {out / 'l1/review.csv'}")


def register(sub):
    p = sub.add_parser("l2-prepare", help="从人工审核生成 L2 累计数据快照")
    for key in ("reviews", "manifest", "splits", "taxonomy", "out"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--previous", help="续训时必须传上一版数据快照，保留旧类别的审核范围")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("l2-check", help="检查已审核事件、标签、来源划分")
    p.add_argument("--dataset", required=True)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("l2-embed", help="本地 CNN14 权重提取冻结特征")
    for key in ("dataset", "checkpoint", "out"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--window", type=float, default=4.0)
    p.add_argument("--stride", type=float, default=2.0)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser("l2-train", help="训练/回放续训多标签分类头")
    for key in ("dataset", "features", "out", "version"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--resume")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--memory-per-class", type=int, default=32)
    p.add_argument("--distill", type=float, default=0.5)
    p.add_argument("--no-replay", action="store_true", help="对照实验：只用新增/修订样本续训")
    p.add_argument("--full-retrain", action="store_true", help="对照实验：累计训练集从头训练分类头")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("l2-evaluate", help="冻结模型在 val/test 上做条件评估")
    for key in ("model", "dataset", "features", "out"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("l2-run", help="对 L1 events.json 进行多类型识别与未知拒识")
    for key in ("events", "model", "checkpoint", "out"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--audio", help="原始 WAV 移动后显式指定，仍需匹配 L1 校验和")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("pipeline", help="单 WAV 完整两级推理")
    for key in ("input", "l1-model", "l2-model", "checkpoint", "out"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_pipeline)
