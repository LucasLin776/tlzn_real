"""Dataset manifest, interval annotations, validation and label masks.

Design rules implemented here:

* Only explicitly reviewed regions get a label.  Everything else is unknown.
* A file-level coarse label never expands into frame/interval truth.
* A file is allowed to derive "normal outside the anomalies" only when
  ``fully_reviewed`` is true AND the anomalies are interval annotations.
* ``uncertain`` always stays unknown, even inside a fully reviewed file.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .audio import wav_info

LABELS = ("normal", "anomaly", "uncertain")
PRECISIONS = ("file", "interval")
TOL = 1e-6

MANIFEST_FIELDS = [
    "audio_id", "path", "source_id", "device_id", "recorded_at",
    "duration_s", "sample_rate", "channels", "fully_reviewed", "notes",
]
ANNOTATION_FIELDS = [
    "annotation_id", "audio_id", "start_s", "end_s", "label", "precision",
    "fault_type", "note", "reviewed_at", "annotator", "version",
]


@dataclass
class AudioRecord:
    audio_id: str
    path: str
    source_id: str = ""
    device_id: str = ""
    recorded_at: str = ""
    duration_s: float = 0.0
    sample_rate: int = 0
    channels: int = 1
    fully_reviewed: bool = False
    notes: str = ""

    def to_row(self) -> dict:
        row = asdict(self)
        row["fully_reviewed"] = "true" if self.fully_reviewed else "false"
        return row


@dataclass
class Annotation:
    annotation_id: str
    audio_id: str
    start_s: float
    end_s: float
    label: str
    precision: str = "interval"
    fault_type: str = ""
    note: str = ""
    reviewed_at: str = ""
    annotator: str = ""
    version: str = "1"

    def to_row(self) -> dict:
        return asdict(self)

    def as_interval(self) -> dict:
        return {"start_s": self.start_s, "end_s": self.end_s, "label": self.label}


@dataclass
class LabelMask:
    times: np.ndarray
    labels: np.ndarray          # 1 anomaly, 0 normal, -1 unknown
    reviewed: np.ndarray        # bool
    duration_s: float
    step_s: float
    coarse: bool = False

    @property
    def normal_s(self) -> float:
        return float(np.sum(self.labels == 0) * self.step_s)

    @property
    def anomaly_s(self) -> float:
        return float(np.sum(self.labels == 1) * self.step_s)

    @property
    def unknown_s(self) -> float:
        return float(np.sum(self.labels == -1) * self.step_s)


def _bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是"}


def _float(value, default=0.0) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def read_manifest(path: str | Path) -> list[AudioRecord]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        records = []
        for row in reader:
            records.append(
                AudioRecord(
                    audio_id=(row.get("audio_id") or "").strip(),
                    path=(row.get("path") or "").strip(),
                    source_id=(row.get("source_id") or "").strip(),
                    device_id=(row.get("device_id") or "").strip(),
                    recorded_at=(row.get("recorded_at") or "").strip(),
                    duration_s=_float(row.get("duration_s")),
                    sample_rate=int(_float(row.get("sample_rate"))),
                    channels=int(_float(row.get("channels"), 1)),
                    fully_reviewed=_bool(row.get("fully_reviewed", "")),
                    notes=(row.get("notes") or "").strip(),
                )
            )
    return records


def write_manifest(records: list[AudioRecord], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec.to_row())
    return path


def read_annotations(path: str | Path) -> list[Annotation]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"标注文件不存在: {path}；不能将缺失标注当成空标注")
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        anns = []
        for row in reader:
            anns.append(
                Annotation(
                    annotation_id=(row.get("annotation_id") or "").strip(),
                    audio_id=(row.get("audio_id") or "").strip(),
                    start_s=_float(row.get("start_s")),
                    end_s=_float(row.get("end_s")),
                    label=(row.get("label") or "uncertain").strip(),
                    precision=(row.get("precision") or "interval").strip(),
                    fault_type=(row.get("fault_type") or "").strip(),
                    note=(row.get("note") or "").strip(),
                    reviewed_at=(row.get("reviewed_at") or "").strip(),
                    annotator=(row.get("annotator") or "").strip(),
                    version=(row.get("version") or "1").strip(),
                )
            )
    return anns


def write_annotations(anns: list[Annotation], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        for ann in anns:
            writer.writerow(ann.to_row())
    return path


def resolve_audio_path(record: AudioRecord, manifest_path: str | Path) -> Path:
    p = Path(record.path)
    if p.is_absolute():
        return p
    return (Path(manifest_path).parent / p).resolve()


def validate_dataset(
    records: list[AudioRecord],
    anns: list[Annotation],
    manifest_path: str | Path,
    check_audio: bool = True,
) -> tuple[list[str], list[str]]:
    """Return (errors, warnings).  Fills missing duration/sample_rate when the wav exists."""
    errors: list[str] = []
    warnings: list[str] = []
    manifest_path = Path(manifest_path)
    ids = [r.audio_id for r in records]
    if len(ids) != len(set(ids)):
        errors.append("manifest 中存在重复 audio_id")
    ann_ids = [a.annotation_id for a in anns]
    if len(ann_ids) != len(set(ann_ids)):
        errors.append("annotations 中存在重复 annotation_id")
    by_id = {r.audio_id: r for r in records}

    for rec in records:
        if not rec.audio_id:
            errors.append("manifest 有空 audio_id")
            continue
        if not rec.path:
            errors.append(f"{rec.audio_id}: path 为空")
            continue
        p = resolve_audio_path(rec, manifest_path)
        if check_audio:
            if not p.exists():
                errors.append(f"{rec.audio_id}: 音频不存在 {p}")
            else:
                try:
                    info = wav_info(p)
                except ValueError as exc:
                    errors.append(f"{rec.audio_id}: {exc}")
                else:
                    if rec.duration_s <= 0:
                        rec.duration_s = info.duration_s
                    if rec.sample_rate <= 0:
                        rec.sample_rate = info.sample_rate
                    if abs(rec.duration_s - info.duration_s) > 0.05:
                        warnings.append(
                            f"{rec.audio_id}: manifest 时长 {rec.duration_s:.3f}s "
                            f"与 wav 实际 {info.duration_s:.3f}s 不一致，以 wav 为准"
                        )
                        rec.duration_s = info.duration_s
        if rec.duration_s <= 0:
            errors.append(f"{rec.audio_id}: duration_s 必须 > 0")

    for ann in anns:
        rec = by_id.get(ann.audio_id)
        if rec is None:
            errors.append(f"{ann.annotation_id}: 未知 audio_id={ann.audio_id}")
            continue
        if ann.label not in LABELS:
            errors.append(f"{ann.annotation_id}: 非法 label={ann.label}")
        if ann.precision not in PRECISIONS:
            errors.append(f"{ann.annotation_id}: 非法 precision={ann.precision}")
        if not (math.isfinite(ann.start_s) and math.isfinite(ann.end_s)):
            errors.append(f"{ann.annotation_id}: 起止时间不是有限数值")
            continue
        if ann.start_s < -TOL:
            errors.append(f"{ann.annotation_id}: start_s < 0")
        if ann.end_s <= ann.start_s + TOL:
            errors.append(f"{ann.annotation_id}: end_s 必须大于 start_s")
        if rec.duration_s > 0 and ann.end_s > rec.duration_s + 0.05:
            errors.append(
                f"{ann.annotation_id}: end_s={ann.end_s:.3f} 超出音频时长 {rec.duration_s:.3f}"
            )
        if ann.precision == "file":
            if ann.start_s > 0.05 or (rec.duration_s > 0 and abs(ann.end_s - rec.duration_s) > 0.1):
                errors.append(
                    f"{ann.annotation_id}: file 级标注必须覆盖 0..duration，"
                    f"当前 {ann.start_s:.3f}..{ann.end_s:.3f}"
                )
        if ann.precision == "file" and ann.label == "anomaly" and rec.fully_reviewed:
            errors.append(
                f"{ann.annotation_id}: fully_reviewed=true 但只有 file 级 anomaly，"
                "无法推导区间真值；请补充区间标注或把 fully_reviewed 设为 false"
            )

    # conflicts on overlapping interval annotations
    for audio_id in {a.audio_id for a in anns}:
        intervals = [a for a in anns if a.audio_id == audio_id and a.precision == "interval"]
        intervals.sort(key=lambda a: a.start_s)
        for i, left in enumerate(intervals):
            for right in intervals[i + 1 :]:
                if right.start_s >= left.end_s - TOL:
                    break
                overlap = min(left.end_s, right.end_s) - max(left.start_s, right.start_s)
                if overlap <= TOL:
                    continue
                if left.label != right.label:
                    errors.append(
                        f"{audio_id}: 区间冲突 {left.annotation_id}({left.label} "
                        f"{left.start_s:.2f}-{left.end_s:.2f}) 与 "
                        f"{right.annotation_id}({right.label} "
                        f"{right.start_s:.2f}-{right.end_s:.2f})"
                    )
    return errors, warnings


def label_mask(
    record: AudioRecord,
    anns: list[Annotation],
    step_s: float = 0.1,
) -> LabelMask:
    """Build a target-grid mask.  Unknown is -1 and must never be treated as normal."""
    duration = float(record.duration_s)
    if duration <= 0:
        raise ValueError(f"{record.audio_id}: duration_s 必须 > 0")
    n = max(1, int(math.ceil(duration / step_s)))
    times = (np.arange(n, dtype=np.float64) + 0.5) * step_s
    times = times[times < duration]
    labels = np.full(len(times), -1, dtype=np.int8)
    reviewed = np.zeros(len(times), dtype=bool)
    painted = np.zeros(len(times), dtype=bool)
    uncertain = np.zeros(len(times), dtype=bool)
    coarse = False
    coarse_anomaly = False

    mine = [a for a in anns if a.audio_id == record.audio_id]
    for ann in mine:
        if ann.precision == "file":
            coarse = True
            if ann.label == "uncertain":
                uncertain[:] = True
            elif ann.label == "anomaly":
                coarse_anomaly = True
            continue
        m = (times >= ann.start_s) & (times < ann.end_s)
        if not m.any():
            continue
        painted |= m
        if ann.label == "uncertain":
            labels[m] = -1
            reviewed[m] = False
            uncertain |= m
        else:
            labels[m] = 1 if ann.label == "anomaly" else 0
            reviewed[m] = True

    if record.fully_reviewed and not coarse_anomaly:
        fill = (~painted) & (~uncertain)
        labels[fill] = 0
        reviewed[fill] = True
    return LabelMask(times, labels, reviewed, duration, step_s, coarse=coarse)


def regions_from_mask(mask: LabelMask) -> list[dict]:
    """Contiguous 0/1/-1 runs with real start/end seconds."""
    out: list[dict] = []
    if len(mask.labels) == 0:
        return out
    start = 0
    for i in range(1, len(mask.labels) + 1):
        if i == len(mask.labels) or mask.labels[i] != mask.labels[start]:
            a = max(0.0, mask.times[start] - mask.step_s / 2)
            b = min(mask.duration_s, mask.times[i - 1] + mask.step_s / 2)
            out.append(
                {
                    "start_s": round(a, 4),
                    "end_s": round(b, 4),
                    "label": {-1: "uncertain", 0: "normal", 1: "anomaly"}[int(mask.labels[start])],
                    "reviewed": bool(mask.reviewed[start]),
                }
            )
            start = i
    return out


def anomaly_events(record: AudioRecord, anns: list[Annotation]) -> list[dict]:
    """Exact interval ground-truth events (interval precision only), merged."""
    intervals = [
        (a.start_s, a.end_s)
        for a in anns
        if a.audio_id == record.audio_id and a.precision == "interval" and a.label == "anomaly"
    ]
    intervals.sort()
    merged: list[list[float]] = []
    for a, b in intervals:
        if merged and a <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [
        {"start_s": float(max(0.0, a)), "end_s": float(min(record.duration_s, b))}
        for a, b in merged
        if b > a
    ]


def dataset_summary(
    records: list[AudioRecord],
    anns: list[Annotation],
    step_s: float = 0.1,
) -> dict:
    normal_s = anomaly_s = unknown_s = 0.0
    events = 0
    coarse_files = 0
    full_reviewed = 0
    for rec in records:
        mask = label_mask(rec, anns, step_s)
        normal_s += mask.normal_s
        anomaly_s += mask.anomaly_s
        unknown_s += mask.unknown_s
        events += len(anomaly_events(rec, anns))
        if rec.fully_reviewed:
            full_reviewed += 1
        if any(a.audio_id == rec.audio_id and a.precision == "file" for a in anns):
            coarse_files += 1

    reasons: list[str] = []
    sources = {r.source_id or r.audio_id for r in records}
    devices = {r.device_id for r in records if r.device_id}
    if normal_s < 60:
        reasons.append(f"已审核正常音频 {normal_s:.1f}s < 60s，不足以训练基线/有监督模型")
    if len(sources) < 3:
        reasons.append("至少需要 3 个独立录音/批次，当前不足以做 train/val/test 分组划分")
    if anomaly_s < 10:
        reasons.append(f"已审核异常音频 {anomaly_s:.1f}s < 10s，不足以训练有监督模型")
    if events < 3:
        reasons.append(f"独立异常事件 {events} < 3，不足以做事件级评估")

    return {
        "files": len(records),
        "sources": len(sources),
        "devices": len(devices),
        "normal_s": round(normal_s, 3),
        "anomaly_s": round(anomaly_s, 3),
        "unknown_s": round(unknown_s, 3),
        "anomaly_events": events,
        "fully_reviewed_files": full_reviewed,
        "coarse_label_files": coarse_files,
        "reasons": reasons,
    }


def migrate_legacy_labels(
    labels_csv: str | Path,
    audio_dir: str | Path,
    manifest_out: str | Path,
    annotations_out: str | Path,
    base_dir: str | Path | None = None,
) -> tuple[list[AudioRecord], list[Annotation], list[str]]:
    """Migrate the old flat labels.csv without inventing interval truth."""
    labels_csv = Path(labels_csv)
    audio_dir = Path(audio_dir)
    manifest_out = Path(manifest_out)
    annotations_out = Path(annotations_out)
    base_dir = Path(base_dir) if base_dir else manifest_out.parent.parent
    warnings: list[str] = []
    records: list[AudioRecord] = []
    anns: list[Annotation] = []

    with labels_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    seen: set[str] = set()
    for i, row in enumerate(rows, 1):
        fname = (row.get("file") or row.get("filename") or "").strip()
        if not fname:
            warnings.append(f"第 {i} 行缺少文件名，跳过")
            continue
        audio_path = (audio_dir / fname).resolve()
        if not audio_path.exists():
            warnings.append(f"找不到音频 {audio_path}，仍写入 manifest 供人工修复")
            duration = 0.0
            sr = 0
            channels = 1
        else:
            info = wav_info(audio_path)
            duration = info.duration_s
            sr = info.sample_rate
            channels = info.channels
        audio_id = audio_path.stem
        base_id = audio_id
        k = 2
        while audio_id in seen:
            audio_id = f"{base_id}_{k}"
            k += 1
        seen.add(audio_id)
        try:
            rel = audio_path.relative_to(manifest_out.parent.resolve())
        except ValueError:
            rel = audio_path
        records.append(
            AudioRecord(
                audio_id=audio_id,
                path=str(rel),
                source_id=(row.get("source_id") or base_id).strip(),
                device_id=(row.get("device_id") or "").strip(),
                recorded_at=(row.get("recorded_at") or "").strip(),
                duration_s=duration,
                sample_rate=sr,
                channels=channels,
                fully_reviewed=False,
                notes="迁移自旧 labels.csv；未声明整段完整审核",
            )
        )
        label = (row.get("label") or "uncertain").strip().lower()
        if label not in LABELS:
            label = "uncertain"
        note = (row.get("note") or "").strip()
        start = _float(row.get("start_s"))
        end = _float(row.get("end_s"), duration)
        file_level = ("无秒级" in note) or ("无区间" in note) or (abs(start) < TOL and duration > 0 and abs(end - duration) < 0.1)
        if file_level:
            start, end = 0.0, duration
        anns.append(
            Annotation(
                annotation_id=f"legacy_{i:04d}",
                audio_id=audio_id,
                start_s=start,
                end_s=end,
                label=label,
                precision="file" if file_level else "interval",
                fault_type=(row.get("fault_type") or "").strip(),
                note=note or "旧标注迁移",
                version="legacy",
            )
        )
        if not file_level and duration > 0 and end > duration + 0.05:
            warnings.append(f"{fname}: 标注 end={end} 超出时长 {duration}")

    write_manifest(records, manifest_out)
    write_annotations(anns, annotations_out)
    migration_note = annotations_out.parent / "MIGRATION.md"
    migration_note.write_text(
        "# 旧标签迁移说明\n\n"
        f"- 原文件：`{labels_csv}`（未修改）\n"
        f"- 新 manifest：`{manifest_out}`\n"
        f"- 新区间标注：`{annotations_out}`\n\n"
        "## 规则\n\n"
        "1. 旧 `labels.csv` 中 0–30s、备注“无秒级标注”的记录迁移为 **file 级粗标**，"
        "`fully_reviewed=false`，不生成逐帧/逐区间异常真值。\n"
        "2. 只有明确写出的区间标注才保留为 interval；file 级标签只用于审计和追溯。\n"
        "3. 未审核区域一律 unknown，不参与训练、阈值校准和评估。\n"
        "4. 原 `data/labels.csv` 保持原样，作为历史记录。\n",
        encoding="utf-8",
    )
    return records, anns, warnings


def import_simple(
    audio_dir: str | Path,
    labels_path: str | Path,
    manifest_out: str | Path,
    annotations_out: str | Path,
    *,
    assume_unlabeled_normal: bool = False,
    reviewed_files: set[str] | None = None,
) -> tuple[list[AudioRecord], list[Annotation], list[str]]:
    """Import "audio folder + interval labels" without requiring metadata.

    Labels may be CSV with ``file,start_s,end_s,label[,group,fault_type,note]``
    or JSON with the same keys.  Duration/sample rate/channels are read from
    the WAV header.  ``device_id``/``recorded_at`` stay empty and are never
    model features.

    ``assume_unlabeled_normal=False`` is the safe default: only explicitly
    labelled intervals are used, everything else stays unknown.  Set it to
    True only when the user declares that every file was fully reviewed and
    all anomalies were marked.
    """
    import json

    audio_dir = Path(audio_dir).resolve()
    labels_path = Path(labels_path)
    manifest_out = Path(manifest_out)
    annotations_out = Path(annotations_out)
    warnings: list[str] = []
    reviewed_set = {str(x).strip() for x in (reviewed_files or set()) if str(x).strip()}

    files = sorted(audio_dir.rglob("*.wav"))
    if not files:
        raise ValueError(f"目录里没有 WAV: {audio_dir}")

    lookup: dict[str, list[Path]] = {}
    for path in files:
        rel = path.relative_to(audio_dir)
        keys = {str(rel), rel.name, path.stem, str(path)}
        for key in keys:
            lookup.setdefault(key, []).append(path)
            lookup.setdefault(key.lower(), []).append(path)

    def resolve_file(value: str) -> Path:
        key = str(value).strip()
        hits = lookup.get(key) or lookup.get(key.lower())
        if not hits:
            raise ValueError(f"标注引用了不存在的音频: {value}")
        unique = sorted({p.resolve() for p in hits})
        if len(unique) > 1:
            raise ValueError(
                f"标注文件名有歧义: {value}，请写相对路径（如 sub/a.wav）"
            )
        return Path(unique[0])

    if labels_path.suffix.lower() == ".json":
        raw = json.loads(labels_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("annotations") or raw.get("labels") or []
        if not isinstance(raw, list):
            raise ValueError("JSON 标注需要是数组，或包含 annotations/labels 数组")
        rows = raw
    else:
        with labels_path.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))

    parsed: list[dict] = []
    for i, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"第 {i} 条标注不是对象")
        fname = row.get("file") or row.get("audio") or row.get("audio_id") or row.get("path") or ""
        if not fname:
            raise ValueError(f"第 {i} 条标注缺少 file/audio_id")
        label = str(row.get("label") or "uncertain").strip().lower()
        if label in {"正常", "n", "0"}:
            label = "normal"
        elif label in {"异常", "abnormal", "1"}:
            label = "anomaly"
        elif label in {"不确定", "unknown", "unlabeled", ""}:
            label = "uncertain"
        if label not in LABELS:
            raise ValueError(f"第 {i} 条标注 label 非法: {label}")
        try:
            start = float(row.get("start_s", row.get("start", "")))
            end = float(row.get("end_s", row.get("end", "")))
        except (TypeError, ValueError):
            raise ValueError(f"第 {i} 条标注 start_s/end_s 不是数字") from None
        path = resolve_file(str(fname))
        parsed.append(
            {
                "path": path,
                "start_s": start,
                "end_s": end,
                "label": label,
                "group": str(row.get("group") or row.get("source_id") or "").strip(),
                "fault_type": str(row.get("fault_type") or "").strip(),
                "note": str(row.get("note") or "").strip(),
            }
        )

    by_path: dict[Path, list[dict]] = {}
    for row in parsed:
        by_path.setdefault(row["path"], []).append(row)

    # unique audio_id per file; duplicate stems get a short hash suffix
    used_ids: set[str] = set()
    records: list[AudioRecord] = []
    audio_id_by_path: dict[Path, str] = {}
    for path in files:
        info = wav_info(path)
        base = path.stem
        audio_id = base
        if audio_id in used_ids:
            audio_id = f"{base}_{short_hash(str(path.resolve()))}"
        used_ids.add(audio_id)
        audio_id_by_path[path] = audio_id
        rel_for_review = str(path.relative_to(audio_dir))
        full_reviewed = bool(
            assume_unlabeled_normal
            or path.name in reviewed_set
            or path.stem in reviewed_set
            or rel_for_review in reviewed_set
        )
        rows_for_file = by_path.get(path, [])
        groups = {r["group"] for r in rows_for_file if r["group"]}
        if len(groups) > 1:
            warnings.append(f"{path.name}: 标注意义上的 group 不一致 {sorted(groups)}，取第一个")
        group = sorted(groups)[0] if groups else base
        try:
            rel = path.resolve().relative_to(manifest_out.parent.resolve())
        except ValueError:
            rel = path.resolve()
        records.append(
            AudioRecord(
                audio_id=audio_id,
                path=str(rel),
                source_id=group,
                device_id="",
                recorded_at="",
                duration_s=info.duration_s,
                sample_rate=info.sample_rate,
                channels=info.channels,
                fully_reviewed=full_reviewed,
                notes="simple import; device/time not provided",
            )
        )

    anns: list[Annotation] = []
    for i, row in enumerate(parsed, 1):
        audio_id = audio_id_by_path[row["path"]]
        info = wav_info(row["path"])
        if not (math.isfinite(row["start_s"]) and math.isfinite(row["end_s"])):
            raise ValueError(f"第 {i} 条标注起止不是有限数字")
        if row["end_s"] <= row["start_s"]:
            raise ValueError(f"第 {i} 条标注 end_s 必须大于 start_s")
        if row["start_s"] < 0 or row["end_s"] > info.duration_s + 0.05:
            raise ValueError(
                f"第 {i} 条标注越界: {row['start_s']}..{row['end_s']}，"
                f"音频 {info.duration_s:.3f}s"
            )
        anns.append(
            Annotation(
                annotation_id=f"simple_{i:05d}",
                audio_id=audio_id,
                start_s=row["start_s"],
                end_s=min(row["end_s"], info.duration_s),
                label=row["label"],
                precision="interval",
                fault_type=row["fault_type"],
                note=row["note"],
                version="simple",
            )
        )

    if not parsed and not assume_unlabeled_normal:
        warnings.append(
            "没有读到任何区间标注，且未开启 --assume-unlabeled-normal："
            "所有音频都会是 unknown，不能训练。"
        )
    labelled_ids = {audio_id_by_path[row["path"]] for row in parsed}
    records_missing = [
        rec.audio_id for rec in records
        if (not rec.fully_reviewed) and rec.audio_id not in labelled_ids
    ]
    if records_missing and not assume_unlabeled_normal:
        warnings.append(f"{len(records_missing)} 个音频没有标注，将保持 unknown")

    write_manifest(records, manifest_out)
    write_annotations(anns, annotations_out)
    errors, more_warnings = validate_dataset(records, anns, manifest_out, check_audio=True)
    warnings.extend(more_warnings)
    if errors:
        raise ValueError("导入后校验失败：\n" + "\n".join(errors))
    return records, anns, warnings


def write_dataset_json(records: list[AudioRecord], anns: list[Annotation], path: str | Path) -> Path:
    path = Path(path)
    path.write_text(
        json.dumps(
            {"records": [asdict(r) for r in records], "annotations": [asdict(a) for a in anns]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def short_hash(text: str, n: int = 8) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]
