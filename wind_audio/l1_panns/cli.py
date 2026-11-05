"""Opt-in commands; importing the CLI does not import torch or load weights."""
from pathlib import Path
import json

from . import MODEL_TYPE
from .config import EventConfig, NetworkConfig
from .data import digest, load_dataset, prepare, read_json, write_json


def model_module():
    try:
        from . import model
    except (ImportError, OSError) as exc:
        raise ValueError("L1 PANNs 需要可用的本地 PyTorch；本命令不安装依赖") from exc
    return model


def cmd_prepare(a):
    obj = prepare(a.pairing, a.semantics, a.out)
    print(f"显式 L1 数据快照: {a.out}，{len(obj['records'])} 条录音；未标注区域保持 unknown")


def cmd_check(a):
    obj = load_dataset(a.dataset)
    print({s: sum(r["split"] == s for r in obj["records"]) for s in ("train", "calibration", "val", "test")})
    print("路径/音频哈希/来源划分已检查；未提取特征、未训练、未拟合阈值。")


def cmd_train(a):
    from .workflow import preflight, run_training, training_data, validate_command
    validate_command(a)
    cfg = NetworkConfig(**read_json(a.network_config)) if a.network_config else NetworkConfig()
    event_cfg = EventConfig(**read_json(a.event_config)) if a.event_config else EventConfig()
    data = training_data(a)
    validation = preflight(data, cfg, reports=not a.train_only, min_frames=a.min_frames, quantiles=a.quantiles)
    if not a.train_only:
        from .reporting import report_dependencies
        report_dependencies(a.plot_font)
    if a.check_only:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        print("仅输入校验完成：未提特征、未训练、未校准、未评估、未创建输出目录。")
        return
    run_training(data, a.checkpoint, a.out, cfg=cfg, event_cfg=event_cfg, validation=validation,
                 epochs=a.epochs, lr=a.lr, noise_std=a.noise_std, augmentation_reviewed=a.augmentation_reviewed,
                 seed=a.seed, device=a.device, train_only=a.train_only,
                 min_frames=a.min_frames, quantiles=a.quantiles, font_path=a.plot_font)
    if a.train_only:
        print(f"未校准训练产物: {a.out}/trained；未执行校准或评估。")
    else:
        print(f"完整流程完成: {a.out}；验证/测试 PDF: val/comparison.pdf、test/comparison.pdf")


def cmd_calibrate(a):
    model_module().calibrate(a.model, a.dataset, a.out, checkpoint=a.checkpoint, device=a.device,
                             quantiles=tuple(a.quantiles), min_frames=a.min_frames)
    print(f"独立正常校准模型: {a.out}；推理固定模型、标准化和阈值。")


def write_detection(model, audio, directory):
    if Path(directory).exists():
        raise ValueError("检测输出目录已存在，请指定新目录")
    payload = model.predict(audio)
    out = model_module().new_directory(directory)
    write_json(out / "events.json", payload)
    # Shared review schema is kept for L2; this only creates a blank review template.
    from ..review import export_template
    export_template(payload, out / "review.csv", out / "review.json")
    return payload


def detect_inputs(model, source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError("检测输出必须为新目录")
    if source.is_file():
        result = write_detection(model, source, output)
        print(f"{source.name}: {len(result['events'])} 候选，{len(result['unknown_regions'])} 未知区间 -> {output}")
        return
    if not source.is_dir():
        raise ValueError(f"音频不存在: {source}")
    files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")
    if not files:
        raise ValueError("输入目录没有 WAV")
    model_module().new_directory(output)
    results = []
    for path in files:
        folder = output / digest(str(path.resolve()))[:20]
        result = write_detection(model, path, folder)
        results.append({"audio": result["audio"], "events_path": str(folder / "events.json"),
                        "event_count": len(result["events"]), "unknown_regions": result["unknown_regions"]})
    write_json(output / "summary.json", {"model_type": MODEL_TYPE, "files": results})


def cmd_detect(a):
    model = model_module().Model.load(a.model, checkpoint=a.checkpoint, device=a.device)
    detect_inputs(model, a.input, a.out)


def cmd_evaluate(a):
    if Path(a.out).exists():
        raise ValueError("评估输出已存在，请使用新路径")
    model = model_module().Model.load(a.model, checkpoint=a.checkpoint, device=a.device)
    data = load_dataset(a.dataset)
    if a.json_only:
        write_json(a.out, model_module().evaluate(model, data, a.split))
    else:
        from .reporting import write_evaluation_report
        write_evaluation_report(model, data, a.split, a.out, font_path=a.plot_font)
    print(f"独立评估: {a.out}；预测路径与检测相同，unknown 单独统计。")


def maybe_legacy_detect(a):
    """Compatibility for existing l1-detect / L2 pipeline, BEFORE old feature extraction."""
    if not a.config:
        return False
    meta = read_json(a.config)
    if meta.get("model_type") != MODEL_TYPE and meta.get("schema_version") != 4:
        return False
    from ..cli import LEGACY_ONLY_ARGS, V3_ONLY_ARGS
    from ..features import FeatureConfig
    forbidden = LEGACY_ONLY_ARGS | V3_ONLY_ARGS | set(FeatureConfig().to_dict()) | {"min_dur", "padding"}
    provided = sorted(k for k in forbidden if getattr(a, k, None) is not None)
    if provided:
        raise ValueError(f"PANNs 推理禁止覆盖模型/阈值/事件参数: {provided}")
    if not a.out:
        raise ValueError("PANNs 检测必须明确 --out 新目录")
    model = model_module().Model.load(a.config)
    detect_inputs(model, a.input, a.out)
    return True


def register(sub):
    p = sub.add_parser("l1-panns-prepare", help="显式配对表 + 标注语义 -> 独立 L1 数据快照")
    for k in ("pairing", "semantics", "out"):
        p.add_argument("--" + k, required=True)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("l1-panns-check", help="只检查数据身份、审核区间及来源划分，不提特征")
    p.add_argument("--dataset", required=True)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("l1-panns-train", help="校验音频/标签路径，正常训练及独立校准，生成 val/test PDF")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio-dir", help="显式选择的 WAV 目录；不会默认加入任何批次")
    source.add_argument("--dataset", help="兼容已审核的显式数据快照")
    p.add_argument("--labels", "--labels-dir", dest="labels", help="Audition CSV/TSV 标注目录")
    p.add_argument("--pairing", choices=("stem", "sorted", "table"), help="必须明确选择；sorted 表示用户确认顺序对应")
    p.add_argument("--pairing-file", help="table 模式：audio_id,audio_path,label_path；路径相对各输入目录")
    p.add_argument("--splits", help="CSV: audio_id,source_id,split；来源不能跨 train/calibration/val/test")
    semantics = p.add_mutually_exclusive_group()
    semantics.add_argument("--marker-label", choices=("normal", "anomaly", "unknown"), help="显式声明全部标记的语义")
    semantics.add_argument("--semantics", help="显式 all_markers/name_map JSON；不能按名字推断")
    for k in ("checkpoint", "out"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--check-only", action="store_true", help="只验证输入，不创建输出，不提特征/训练/校准/评估")
    p.add_argument("--train-only", action="store_true", help="仅训练至 out/trained，不校准，不生成 val/test PDF")
    p.add_argument("--quantiles", type=float, nargs=3, default=(.95, .995, .9995))
    p.add_argument("--min-frames", type=int, default=1000, help="独立正常校准最少有效帧")
    p.add_argument("--plot-font", help="本地中文字体路径，训练前校验可用性")
    p.add_argument("--network-config")
    p.add_argument("--event-config")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--augmentation-reviewed", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("l1-panns-calibrate", help="独立正常校准集拟合残差阈值，写入新模型版本")
    for k in ("model", "dataset", "out"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--checkpoint", help="迁移后的本地 PANNs 路径，仍须匹配保存的哈希")
    p.add_argument("--device", default="cpu")
    p.add_argument("--quantiles", type=float, nargs=3, default=(0.95, 0.995, 0.9995))
    p.add_argument("--min-frames", type=int, default=1000)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("l1-panns-detect", help="完整音频直接推理；不接收标签、阈值覆盖或旧 L1 候选")
    for k in ("model", "input", "out"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("l1-panns-evaluate", help="同一 predict 路径，输出 PDF/CSV/JSON；--out 为新目录")
    for k in ("model", "dataset", "out"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--device", default="cpu")
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--plot-font")
    p.add_argument("--json-only", action="store_true", help="只输出指标 JSON；此时 --out 为文件路径")
    p.set_defaults(func=cmd_evaluate)
