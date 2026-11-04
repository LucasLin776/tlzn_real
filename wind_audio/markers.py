"""Read Audition-style marker exports: Name / Start / Duration (CSV or TSV)."""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path

from .audio import wav_info
from .data import Annotation, AudioRecord, validate_dataset, write_annotations, write_manifest


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


def import_markers(audio_dir, labels_dir, manifest_out, annotations_out, *,
                   pairing="stem", assume_unlabeled_normal=False,
                   stable_markers=False, min_gap_s=0.10):
    """Read named ranges exported from an audio editor.

    Two conventions are supported:

    * default: every named range is an **anomaly** (the historical behaviour);
      unmarked audio is normal only with ``assume_unlabeled_normal``.
    * ``stable_markers=True``: every named range is a **stable / normal**
      region.  The complement inside each fully reviewed recording becomes the
      anomaly truth, so the same evaluation code can score "find everything
      that is not stable".  Gaps shorter than ``min_gap_s`` are treated as
      annotation tolerance and stay stable.

    ``sorted`` is an explicit user assertion about file correspondence; never
    infer it just because two folders contain the same number of files.
    """
    audio_dir, labels_dir = Path(audio_dir).resolve(), Path(labels_dir).resolve()
    audio = sorted((p for p in audio_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"),
                   key=_natural_key)
    labels = sorted((p for p in labels_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".tsv"}),
                    key=_natural_key)
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

    records, annotations, mapping = [], [], []
    for audio_path, label_path in pairs:
        info = wav_info(audio_path)
        audio_id = label_path.stem
        records.append(AudioRecord(audio_id=audio_id, path=str(audio_path), source_id=audio_path.stem,
                                   duration_s=info.duration_s, sample_rate=info.sample_rate,
                                   channels=info.channels,
                                   fully_reviewed=assume_unlabeled_normal or stable_markers,
                                   notes=f"marker export: {label_path}; pairing={pairing}"))
        mapping.append({"audio_id": audio_id, "audio_path": str(audio_path), "label_path": str(label_path)})
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
                annotations.append(Annotation(annotation_id=f"{audio_id}_{line - 1:03d}", audio_id=audio_id,
                                              start_s=start, end_s=start + duration,
                                              label="normal" if stable_markers else "anomaly",
                                              fault_type=row["Name"].strip(), note=row.get("Description") or ""))
    extra_warnings: list[str] = []
    if stable_markers:
        annotations = _invert_stable(records, annotations, min_gap_s, extra_warnings)
    errors, warnings = validate_dataset(records, annotations, manifest_out)
    warnings = extra_warnings + warnings
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
