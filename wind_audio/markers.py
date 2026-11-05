"""Read Audition-style marker exports: Name / Start / Duration (CSV or TSV).

Two input shapes are supported:

* ``import_markers`` -- one flat pair of directories (the historical entry).
* ``import_batches``  -- one dataset root whose children are batches, each with
  its own audio and label directory (``B05/音频`` + ``B05/标注``).  Batches are
  merged into one manifest so a model can be trained on several recording
  sessions at once; the batch name is kept in the manifest notes.

Label semantics are never guessed from the directory name: the caller selects
"every range is an anomaly" (default) or "every range is a stable/normal
region, the complement is an anomaly" (``stable_markers``).
"""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path

from .audio import wav_info
from .data import Annotation, AudioRecord, validate_dataset, write_annotations, write_manifest

# Accepted directory names inside one batch.  The first entry is the nominal
# one; the rest keep already exported folders readable.
AUDIO_DIR_NAMES = ("音频", "audio", "wav", "audios", "录音")
LABEL_DIR_NAMES = ("标注", "labels", "label", "annotations", "标签")
AUDIO_SUFFIXES = (".wav",)
LABEL_SUFFIXES = (".csv", ".tsv")


def _seconds(value):
    parts = str(value).strip().split(":")
    try:
        numbers = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"无效时间: {value!r}") from None
    if not 1 <= len(numbers) <= 3 or any(not math.isfinite(n) or n < 0 for n in numbers):
        raise ValueError(f"无效时间: {value!r}")
    if any(n >= 60 for n in numbers[1:]):
        raise ValueError(f"分/秒必须小于 60: {value!r}")
    return sum(n * 60 ** i for i, n in enumerate(reversed(numbers)))


def _natural_key(path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(path))]


def _files(directory, suffixes):
    directory = Path(directory)
    return sorted((p for p in directory.rglob("*")
                   if p.is_file() and p.suffix.lower() in suffixes), key=_natural_key)


def _pair_files(audio_dir, labels_dir, pairing):
    """Pair WAV and label exports; ``sorted`` is an explicit user assertion."""
    audio_dir, labels_dir = Path(audio_dir).resolve(), Path(labels_dir).resolve()
    audio, labels = _files(audio_dir, AUDIO_SUFFIXES), _files(labels_dir, LABEL_SUFFIXES)
    if not audio or not labels:
        raise ValueError("音频目录或标注目录为空（需要 WAV 和 CSV/TSV）")
    if len(audio) != len(labels):
        raise ValueError(f"音频和标注数量不同: {len(audio)} / {len(labels)}；每条音频需一份标注")
    if pairing == "stem":
        by_stem = {p.stem: p for p in audio}
        if len(by_stem) != len(audio) or {p.stem for p in labels} != set(by_stem):
            raise ValueError("音频和标注文件名无法一一对应；仅确认编号顺序一致后使用 --pairing sorted")
        pairs = [(by_stem[p.stem], p) for p in labels]
    elif pairing == "sorted":
        pairs = list(zip(audio, labels))
    else:
        raise ValueError("pairing 只支持 stem / sorted")
    if len({p.stem for p in labels}) != len(labels):
        raise ValueError("标注文件名重复，无法生成唯一 audio_id")
    return pairs


def _read_ranges(label_path, audio_id, label):
    """Return the named ranges of one export file as Annotation objects."""
    out = []
    with label_path.open(encoding="utf-8-sig", newline="") as fh:
        header = fh.readline()
        fh.seek(0)
        reader = csv.DictReader(fh, delimiter="\t" if "\t" in header else ",")
        if not {"Name", "Start", "Duration"} <= set(reader.fieldnames or []):
            raise ValueError(f"{label_path.name}: 缺少 Name / Start / Duration 表头")
        for line, row in enumerate(reader, 2):
            try:
                if row.get("Time Format", "decimal") != "decimal":
                    raise ValueError("仅支持 decimal 时间，不支持帧号或采样点时间")
                if not row["Name"].strip():
                    raise ValueError("事件名称为空")
                start, duration = _seconds(row["Start"]), _seconds(row["Duration"])
                if duration <= 0:
                    raise ValueError("点标记缺少持续时间，不能作为区间真值")
            except (ValueError, AttributeError) as exc:
                raise ValueError(f"{label_path.name}:{line}: {exc}") from exc
            out.append(Annotation(annotation_id=f"{audio_id}_{line - 1:03d}", audio_id=audio_id,
                                  start_s=start, end_s=start + duration, label=label,
                                  fault_type=row["Name"].strip(),
                                  note=row.get("Description") or ""))
    return out


def _batch_records(pairs, *, batch, pairing, assume_unlabeled_normal, stable_markers):
    """Build records/annotations for one batch.  Labels are not yet inverted."""
    label = "normal" if stable_markers else "anomaly"
    records, annotations = [], []
    for audio_path, label_path in pairs:
        info = wav_info(audio_path)
        audio_id = label_path.stem
        notes = f"marker export: {label_path}; pairing={pairing}"
        source_id = audio_path.stem
        if batch:
            notes = f"batch={batch}; " + notes
            source_id = f"{batch}/{audio_path.stem}"
        records.append(AudioRecord(audio_id=audio_id, path=str(audio_path), source_id=source_id,
                                   duration_s=info.duration_s, sample_rate=info.sample_rate,
                                   channels=info.channels, fully_reviewed=assume_unlabeled_normal,
                                   notes=notes))
        annotations.extend(_read_ranges(label_path, audio_id, label))
    return records, annotations


def _finalize(records, annotations, manifest_out, annotations_out, mapping, warnings, *,
              stable_markers, assume_unlabeled_normal):
    """Validate, then write manifest / annotations / pairing in one place."""
    errors, extra = validate_dataset(records, annotations, manifest_out)
    warnings = warnings + extra
    if errors:
        raise ValueError("标注校验失败：\n" + "\n".join(errors))
    write_manifest(records, manifest_out)
    write_annotations(annotations, annotations_out)
    with (Path(manifest_out).parent / "pairing.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["audio_id", "audio_path", "label_path"])
        writer.writeheader()
        writer.writerows(mapping)
    if stable_markers:
        warnings.append("按平稳区域标注导入：标注区间为 normal，录音内其余时间按异常处理")
    elif not assume_unlabeled_normal:
        warnings.append("未声明完整审核：空标注和未标注部分保持 unknown，不视为正常")
    return records, annotations, warnings


def import_markers(audio_dir, labels_dir, manifest_out, annotations_out, *,
                   pairing="stem", assume_unlabeled_normal=False,
                   stable_markers=False, min_gap_s=0.10):
    """Read one audio/label directory pair.

    * default: every named range is an **anomaly** (the historical behaviour);
      unmarked audio is normal only with ``assume_unlabeled_normal``.
    * ``stable_markers=True``: every named range is a **stable / normal**
      region.  The complement inside each fully reviewed recording becomes the
      anomaly truth, so the same evaluation code can score "find everything
      that is not stable".  Gaps shorter than ``min_gap_s`` are treated as
      annotation tolerance and stay stable.
    """
    pairs = _pair_files(audio_dir, labels_dir, pairing)
    records, annotations = _batch_records(
        pairs, batch=None, pairing=pairing,
        assume_unlabeled_normal=assume_unlabeled_normal or stable_markers,
        stable_markers=stable_markers)
    mapping = [{"audio_id": r.audio_id, "audio_path": r.path,
                "label_path": pairs[i][1].as_posix()} for i, r in enumerate(records)]
    extra_warnings: list[str] = []
    if not stable_markers:
        names = [a.fault_type for a in annotations]
        calm = sum("正常" in n or "normal" in n.lower() for n in names)
        if calm:
            raise ValueError(f"标注中有 {calm}/{len(names)} 个正常标记，不能按异常区间导入。"
                             "请使用 l1-test-normal + 正常模型，或显式 --labels-are-stable；"
                             "不能用 --assume-unlabeled-normal 解释正常标注。")
    if stable_markers:
        annotations = _invert_stable(records, annotations, min_gap_s, extra_warnings)
    return _finalize(records, annotations, manifest_out, annotations_out, mapping,
                     extra_warnings, stable_markers=stable_markers,
                     assume_unlabeled_normal=assume_unlabeled_normal)


def _has_files(directory, suffixes):
    return any(p.is_file() and p.suffix.lower() in suffixes for p in directory.iterdir())


def _find_subdir(parent, names, suffixes):
    """Locate the single child directory that holds files with ``suffixes``."""
    candidates = []
    for child in sorted(Path(parent).iterdir(), key=lambda p: _natural_key(p.name)):
        if not child.is_dir() or child.name.startswith(".") or child.name == "__pycache__":
            continue
        if _has_files(child, suffixes):
            candidates.append(child)
    if not candidates:
        return None
    named = [p for p in candidates if p.name.lower() in names]
    if len(named) == 1:
        return named[0]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"{parent}: 无法确定目录，候选 {[p.name for p in candidates]}；"
                     f"请命名为 {' / '.join(names[:2])}")


def _layout(parent):
    """Return ``(audio_dir, labels_dir)`` when ``parent`` is one batch."""
    audio = _find_subdir(parent, AUDIO_DIR_NAMES, AUDIO_SUFFIXES)
    labels = _find_subdir(parent, LABEL_DIR_NAMES, LABEL_SUFFIXES)
    if audio is None or labels is None or audio == labels:
        return None
    return audio, labels


def _batch_hint(directory):
    """True when the folder is named like a batch but is missing one side."""
    names = {p.name.lower() for p in Path(directory).iterdir() if p.is_dir()}
    known = {n.lower() for n in AUDIO_DIR_NAMES + LABEL_DIR_NAMES}
    return bool(names & known)


def find_batches(dataset_dir):
    """Discover ``[(batch_name, audio_dir, labels_dir)]`` under a dataset root.

    Accepts both shapes:

    * ``root/B05/音频`` + ``root/B05/标注`` (one entry per batch), and
    * ``root/音频`` + ``root/标注`` (the root itself is a single batch).

    Directory names are only used to locate the folders; they never decide the
    label semantics.
    """
    root = Path(dataset_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"数据集目录不存在: {root}")
    single = _layout(root)
    if single is not None:
        return [(root.name, *single)]
    batches = []
    for child in sorted(root.iterdir(), key=lambda p: _natural_key(p.name)):
        if not child.is_dir() or child.name.startswith(".") or child.name == "__pycache__":
            continue
        found = _layout(child)
        if found is None:
            # A folder that holds a 音频/ or 标注/ style subdirectory was meant
            # to be a batch and is incomplete; unrelated folders are skipped so
            # reports/models can live beside the data without breaking import.
            if _batch_hint(child):
                raise ValueError(f"{child}: 缺少音频目录（{AUDIO_DIR_NAMES[0]}/）"
                                 f"或标注目录（{LABEL_DIR_NAMES[0]}/）")
            continue
        batches.append((child.name, *found))
    if not batches:
        raise ValueError(f"{root} 下没有可用的批次目录")
    return batches


def import_batches(dataset_dir, manifest_out, annotations_out, *,
                   pairing="sorted", assume_unlabeled_normal=False,
                   stable_markers=False, min_gap_s=0.10):
    """Merge every batch under ``dataset_dir`` into one manifest.

    Returns ``(records, annotations, warnings, batch_of)`` where ``batch_of``
    maps ``audio_id -> batch name``.  Audio IDs must stay unique across
    batches; a collision is an error instead of a silent overwrite.
    """
    batches = find_batches(dataset_dir)
    records, annotations, mapping, warnings, batch_of = [], [], [], [], {}
    for name, audio_dir, labels_dir in batches:
        pairs = _pair_files(audio_dir, labels_dir, pairing)
        batch_records, batch_annotations = _batch_records(
            pairs, batch=name, pairing=pairing,
            assume_unlabeled_normal=assume_unlabeled_normal or stable_markers,
            stable_markers=stable_markers)
        records.extend(batch_records)
        annotations.extend(batch_annotations)
        batch_of.update({r.audio_id: name for r in batch_records})
        for (audio, label), record in zip(pairs, batch_records):
            mapping.append({"audio_id": record.audio_id, "audio_path": audio.as_posix(),
                            "label_path": label.as_posix()})
        warnings.append(f"批次 {name}: {len(batch_records)} 条音频（配对 {pairing}，"
                        f"示例 {pairs[0][1].name} ↔ {pairs[0][0].name}）")
    ids = [r.audio_id for r in records]
    duplicates = sorted({aid for aid in ids if ids.count(aid) > 1})
    if duplicates:
        raise ValueError(f"跨批次 audio_id 重名: {duplicates[:5]}；标注编号需在全部批次中唯一")
    if not stable_markers:
        names = [a.fault_type for a in annotations]
        calm = sum("正常" in n or "normal" in n.lower() for n in names)
        if calm:
            raise ValueError(f"标注中有 {calm}/{len(names)} 个正常标记，不能按异常区间导入；"
                             "这些批次标的是正常/平稳区间，请加 --labels-are-stable")
    extra_warnings: list[str] = []
    if stable_markers:
        annotations = _invert_stable(records, annotations, min_gap_s, extra_warnings)
    return _finalize(records, annotations, manifest_out, annotations_out, mapping,
                     warnings + extra_warnings, stable_markers=stable_markers,
                     assume_unlabeled_normal=assume_unlabeled_normal) + (batch_of,)


def _stable_complement(duration: float, intervals: list[tuple[float, float]],
                       min_gap_s: float) -> list[tuple[float, float]]:
    """Everything outside the stable intervals, ignoring slivers < min_gap_s."""
    merged: list[list[float]] = []
    for a, b in sorted(intervals):
        a, b = max(0.0, float(a)), min(float(duration), float(b))
        if b <= a:
            continue
        if merged and a <= merged[-1][1] + min_gap_s:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    gaps: list[tuple[float, float]] = []
    prev = 0.0
    for a, b in merged:
        if a - prev > min_gap_s + 1e-9:
            gaps.append((prev, a))
        prev = max(prev, b)
    if duration - prev > min_gap_s + 1e-9:
        gaps.append((prev, duration))
    return gaps


def _invert_stable(records, stable_annotations, min_gap_s: float, warnings: list[str]):
    """Turn stable ranges into normal + complement anomalies for evaluation."""
    by_id: dict[str, list] = {}
    for ann in stable_annotations:
        by_id.setdefault(ann.audio_id, []).append(ann)
    duration = {r.audio_id: r.duration_s for r in records}
    out = []
    # Empty marker files still describe a reviewed recording. In this mode no
    # normal ranges means the WHOLE recording is anomalous, not silently normal.
    for record in records:
        audio_id = record.audio_id
        items = by_id.get(audio_id, [])
        intervals = [(a.start_s, a.end_s) for a in items]
        out.extend(items)
        gaps = _stable_complement(duration.get(audio_id, 0.0), intervals, min_gap_s)
        for index, (a, b) in enumerate(gaps, 1):
            out.append(Annotation(annotation_id=f"{audio_id}_nonstable_{index:03d}",
                                  audio_id=audio_id, start_s=a, end_s=b,
                                  label="anomaly", precision="interval",
                                  fault_type="non_stable", note="complement of stable labels"))
        if not gaps:
            warnings.append(f"{audio_id}: 全段平稳，无异常区间")
        elif not intervals:
            warnings.append(f"{audio_id}: 无平稳标注，整段按非平稳处理")
    return out
