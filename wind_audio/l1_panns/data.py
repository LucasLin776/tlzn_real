"""Explicit pairing, label semantics and source splits; no batch-name conventions."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from ..audio import load_wav, wav_info

SPLITS = {"train", "calibration", "val", "test"}


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def native_audio(path):
    info = wav_info(path)
    if info.n_frames <= 0:
        raise ValueError(f"空音频: {path}")
    x, _ = load_wav(path, info.sample_rate)  # decode only; never global FFT resampling
    if len(x) != info.n_frames or not np.isfinite(x).all():
        raise ValueError(f"WAV 数据截断或包含非有限样本: {path}")
    return x, info


def pcm_hash(x, sr):
    h = hashlib.sha256(str(sr).encode())
    h.update(np.asarray(x, dtype="<f4").tobytes())
    return h.hexdigest()


def load_audio(path, sr=32000):
    """Local windowed-sinc resampling: finite 16-native-sample support per side.

    Unlike whole-file FFT resampling, an impulse in unknown audio cannot leak
    into every normal sample. Anti-alias cutoff follows the downsampling ratio.
    """
    x, info = native_audio(path)
    if info.sample_rate == sr:
        return x, info, 0.0
    count = int(math.floor(len(x) * sr / info.sample_rate))
    if count < 1:
        raise ValueError("音频短于一个目标采样点")
    y = np.empty(count, np.float32)
    cutoff = min(1.0, sr / info.sample_rate) * 0.94
    for start in range(0, count, 65536):
        pos = np.arange(start, min(count, start + 65536)) * info.sample_rate / sr
        idx = np.floor(pos).astype(np.int64)[:, None] + np.arange(-15, 17)[None]
        delta = pos[:, None] - idx
        weights = cutoff * np.sinc(cutoff * delta) * np.where(
            np.abs(delta) < 16, 0.5 + 0.5 * np.cos(np.pi * delta / 16), 0)
        weights /= weights.sum(axis=1, keepdims=True)
        y[start:start + len(pos)] = (x[np.clip(idx, 0, len(x) - 1)] * weights).sum(axis=1)
    return y, info, 16 / info.sample_rate


def seconds(value):
    fields = str(value).strip().split(":")
    values = [float(x) for x in fields]
    if not 1 <= len(values) <= 3 or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError(f"非法时间: {value}")
    if any(x >= 60 for x in values[1:]):
        raise ValueError(f"非法分秒: {value}")
    return sum(x * 60 ** i for i, x in enumerate(reversed(values)))


def read_markers(path, semantics):
    """Semantics is either explicit all_markers or an exact name->label map."""
    if not isinstance(semantics, dict) or set(semantics) not in ({"all_markers"}, {"name_map"}):
        raise ValueError("语义必须明确指定 all_markers 或 name_map，不能推断")
    mapping = semantics.get("name_map", {})
    if not isinstance(mapping, dict) or any(not isinstance(k, str) for k in mapping):
        raise ValueError("name_map 必须是标记名字到语义的 JSON 对象")
    values = [semantics["all_markers"]] if "all_markers" in semantics else list(mapping.values())
    if not values or any(not isinstance(v, str) or v not in {"normal", "anomaly", "unknown"} for v in values):
        raise ValueError("标注语义只允许 normal/anomaly/unknown")
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        header = f.readline()
        f.seek(0)
        reader = csv.DictReader(f, delimiter="\t" if "\t" in header else ",")
        if not {"Name", "Start", "Duration"} <= set(reader.fieldnames or []):
            raise ValueError(f"{path}: 需要 Name/Start/Duration (Audition CSV/TSV)")
        rows = []
        for row in reader:
            if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("Name", "Start", "Duration")):
                raise ValueError("标注 Name/Start/Duration 缺失或为空")
            if row.get("Time Format", "decimal") != "decimal":
                raise ValueError("只支持 decimal 时间")
            name = row["Name"].strip()
            if "all_markers" not in semantics and name not in mapping:
                raise ValueError(f"标记 {name!r} 没有显式语义映射")
            a, d = seconds(row["Start"]), seconds(row["Duration"])
            if d <= 0:
                raise ValueError("点标记不是审核区间")
            rows.append({"start_s": a, "end_s": a + d,
                         "label": semantics.get("all_markers", mapping.get(name)), "name": name})
    return rows


def canonical_intervals(intervals, duration):
    result = []
    for r in sorted(intervals, key=lambda v: (v["start_s"], v["end_s"])):
        a, b, label = r["start_s"], r["end_s"], r["label"]
        if not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= duration + 1e-6):
            raise ValueError("标注越界/非法：不自动裁剪，请审核源标注")
        if label not in {"normal", "anomaly", "unknown"}:
            raise ValueError("未知标注语义")
        if result and a < result[-1]["end_s"] - 1e-9 and label != result[-1]["label"]:
            raise ValueError("不同语义审核区间重叠")
        if result and a <= result[-1]["end_s"] + 1e-9 and label == result[-1]["label"]:
            result[-1]["end_s"] = max(b, result[-1]["end_s"])
        else:
            result.append({"start_s": a, "end_s": b, "label": label})
    return result


def natural_key(value):
    return [int(x) if x.isdigit() else x.casefold() for x in re.split(r"(\d+)", str(value))]


def _record(audio, labels, audio_id, source_id, split, semantics):
    try:
        x, info = native_audio(audio)
        return {"audio_id": audio_id, "audio_path": str(audio), "source_id": source_id, "split": split,
                "audio_sha256": file_hash(audio), "pcm_sha256": pcm_hash(x, info.sample_rate),
                "duration_s": info.duration_s, "sample_rate": info.sample_rate,
                "label_path": str(labels), "label_sha256": file_hash(labels),
                "intervals": canonical_intervals(read_markers(labels, semantics), info.duration_s)}
    except (ValueError, OSError, EOFError) as exc:
        raise ValueError(f"{audio_id}: 音频 {audio} / 标注 {labels}: {exc}") from exc


def _csv_rows(path, required):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"CSV 文件不存在: {path}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not set(required) <= set(reader.fieldnames or []):
            raise ValueError(f"{path}: 需要列 {', '.join(required)}")
        rows = list(reader)
    if not rows or any(not isinstance(r.get(k), str) or not r[k].strip() for r in rows for k in required):
        raise ValueError(f"{path}: 不能为空，必需列不能缺失或为空")
    return [{k: v.strip() if isinstance(v, str) else v for k, v in row.items()} for row in rows]


def from_directories(audio_dir, labels_dir, *, pairing, semantics, splits_path, pairing_file=None):
    """Validate user-selected directories and explicit source/split assignments.

    No writes, network calls, PANNs features, inferred labels, or automatic split.
    `sorted` is accepted only as an explicit user assertion about correspondence.
    """
    roots = [Path(p).expanduser().resolve() for p in (audio_dir, labels_dir)]
    inventories = []
    for root, name, suffixes in zip(roots, ("音频", "标注"), ({".wav"}, {".csv", ".tsv"})):
        if not root.is_dir():
            raise ValueError(f"{name}目录不存在或不是目录: {root}")
        files = sorted((p.resolve() for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes),
                       key=lambda p: natural_key(p.relative_to(root)))
        if not files:
            raise ValueError(f"{name}目录为空或没有受支持文件: {root}")
        if len(files) != len(set(files)):
            raise ValueError(f"{name}目录包含指向同一文件的重复路径")
        inventories.append(files)
    audio, labels = inventories
    if not pairing or pairing not in {"stem", "sorted", "table"}:
        raise ValueError("请显式指定 --pairing stem/sorted/table；不能根据目录猜配对")
    if pairing != "table" and pairing_file:
        raise ValueError("--pairing-file 只能与 --pairing table 一起使用")
    if len(audio) != len(labels):
        raise ValueError(f"音频与标注数量不同: {len(audio)} / {len(labels)}")
    if pairing == "table":
        if not pairing_file:
            raise ValueError("--pairing table 需要 --pairing-file")
        rows = _csv_rows(pairing_file, ("audio_id", "audio_path", "label_path"))
        pairs = [(r["audio_id"], (roots[0] / r["audio_path"]).resolve(),
                  (roots[1] / r["label_path"]).resolve()) for r in rows]
        if ({a for _, a, _ in pairs} != set(audio) or {b for _, _, b in pairs} != set(labels)
                or len(pairs) != len(audio)):
            raise ValueError("配对表必须一一覆盖指定目录，不允许目录外路径、缺失或重复配对")
    else:
        if len({p.stem for p in labels}) != len(labels) or len({p.stem for p in audio}) != len(audio):
            raise ValueError("文件 stem 重复，请显式使用 --pairing table 及唯一 audio_id")
        if pairing == "stem":
            by_stem = {p.stem: p for p in audio}
            if set(by_stem) != {p.stem for p in labels}:
                raise ValueError("音频/标注无法同名配对；确认顺序后才可指定 sorted，或提供 table")
            pairs = [(p.stem, by_stem[p.stem], p) for p in labels]
        else:
            pairs = [(label.stem, wave, label) for wave, label in zip(audio, labels)]
    if not splits_path:
        raise ValueError("必须通过 --splits 指定 audio_id,source_id,split；不自动划分来源")
    assignments = _csv_rows(splits_path, ("audio_id", "source_id", "split"))
    by_id = {r["audio_id"]: r for r in assignments}
    if len(by_id) != len(assignments) or set(by_id) != {aid for aid, _, _ in pairs}:
        raise ValueError("划分表 audio_id 重复或与当前配对不完全一致")
    if semantics is None:
        raise ValueError("必须显式指定 --marker-label 或 --semantics，不能从标记名字猜语义")
    records = [_record(a, b, aid, by_id[aid]["source_id"], by_id[aid]["split"], semantics)
               for aid, a, b in pairs]
    obj = {"schema_version": 1, "kind": "l1_panns_explicit_dataset", "semantics": semantics,
           "unmarked": "unknown", "records": records,
           "input": {"audio_dir": str(roots[0]), "labels_dir": str(roots[1]), "pairing": pairing,
                     "splits_path": str(Path(splits_path).resolve()), "splits_sha256": file_hash(splits_path),
                     "pairing_sha256": file_hash(pairing_file) if pairing_file else None}}
    validate(obj)
    return obj


def prepare(pairing_path, semantics_path, output):
    """CSV columns explicitly assign every path, source_id and split.

    Requiring a reviewed pairing table handles both same-stem and ordinal
    exports without silently treating sorted filenames as corresponding.
    """
    if Path(output).exists():
        raise FileExistsError("数据快照已存在，请指定新路径")
    path = Path(pairing_path).resolve()
    semantics = read_json(semantics_path)
    records = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"audio_id", "audio_path", "label_path", "source_id", "split"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"配对表需要 {sorted(required)}")
        for row in reader:
            audio, labels = ((path.parent / row[k]).resolve() for k in ("audio_path", "label_path"))
            if not row["audio_path"].strip() or not row["label_path"].strip():
                raise ValueError("音频和标注路径必须明确指定；空标注文件仍保持 unknown")
            records.append(_record(audio, labels, row["audio_id"].strip(), row["source_id"].strip(),
                                   row["split"].strip(), semantics))
    obj = {"schema_version": 1, "kind": "l1_panns_explicit_dataset", "semantics": semantics,
           "pairing_sha256": file_hash(path), "unmarked": "unknown", "records": records}
    validate(obj)
    write_json(output, obj)
    return obj


def validate(obj):
    if (obj.get("schema_version") != 1 or obj.get("kind") != "l1_panns_explicit_dataset"
            or obj.get("unmarked") != "unknown" or not obj.get("records")):
        raise ValueError("需要显式配对的 L1 PANNs 数据快照")
    ids, paths, pcm, raw, sources = set(), set(), set(), set(), {}
    for r in obj["records"]:
        if (not isinstance(r["audio_id"], str) or not r["audio_id"].strip()
                or not isinstance(r["source_id"], str) or not r["source_id"].strip()
                or r["split"] not in SPLITS):
            raise ValueError("每条录音必须明确指定 audio_id/source_id/split")
        if not math.isfinite(r["duration_s"]) or r["duration_s"] <= 0:
            raise ValueError("录音时长必须为有限正数")
        for key in ("audio_sha256", "pcm_sha256"):
            value = r[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("数据快照缺少有效音频哈希")
        for seen, value in ((ids, r["audio_id"]), (paths, str(Path(r["audio_path"]).resolve())),
                            (pcm, r["pcm_sha256"]), (raw, r["audio_sha256"])):
            if value in seen:
                raise ValueError("重复音频/ID：请先合并审核，不能跨集合或重复采样")
            seen.add(value)
        if sources.setdefault(r["source_id"], r["split"]) != r["split"]:
            raise ValueError("同一录音来源跨数据划分（重叠窗口泄漏）")
        r["intervals"] = canonical_intervals(r["intervals"], r["duration_s"])


def load_dataset(path, check_audio=True):
    obj = read_json(path)
    for r in obj["records"]:
        r["audio_path"] = str((Path(path).resolve().parent / r["audio_path"]).resolve())
    validate(obj)
    if check_audio:
        for r in obj["records"]:
            x, info = native_audio(r["audio_path"])
            if (file_hash(r["audio_path"]) != r["audio_sha256"]
                    or pcm_hash(x, info.sample_rate) != r["pcm_sha256"]
                    or abs(info.duration_s - r["duration_s"]) > 1e-6):
                raise ValueError("数据快照音频变更，请重新导入并审核")
    return obj


def partition(data, split):
    rows = [r for r in data["records"] if r["split"] == split]
    if not rows:
        raise ValueError(f"没有显式 {split} 录音")
    return rows


def provenance(rows):
    return [{k: r[k] for k in ("source_id", "audio_sha256", "pcm_sha256")} for r in rows]


def assert_disjoint(rows, history):
    for key in ("source_id", "audio_sha256", "pcm_sha256"):
        if {r[key] for r in rows} & {r[key] for r in history}:
            raise ValueError(f"训练/校准/评估泄漏: {key} 重叠")


def contained(times, intervals, radius, label="normal"):
    mask = np.zeros(len(times), bool)
    for r in canonical_intervals(intervals, max((x["end_s"] for x in intervals), default=0)):
        if r["label"] == label:
            mask |= (times - radius >= r["start_s"]) & (times + radius <= r["end_s"])
    return mask


def normal_crops(row, cfg, resampler_margin):
    """Non-overlapping raw crops. Even discarded halo audio is explicitly normal."""
    for r in canonical_intervals(row["intervals"], row["duration_s"]):
        if r["label"] != "normal":
            continue
        start = math.ceil((r["start_s"] + resampler_margin) * cfg.sample_rate / cfg.stride_samples)
        stop = math.floor((r["end_s"] - resampler_margin) * cfg.sample_rate)
        while True:
            available = (stop - start * cfg.stride_samples - cfg.n_fft + cfg.hop) // cfg.stride_samples
            frames = min(cfg.core_frames + 2 * cfg.halo, available)
            if frames <= 2 * cfg.halo:
                break
            a = start * cfg.stride_samples
            b = a + cfg.samples_for_frames(frames)
            yield a, b
            start = math.ceil(b / cfg.stride_samples)
