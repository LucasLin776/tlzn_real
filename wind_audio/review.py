"""Human-review records for L1/L2 feedback.

The L1 and L2 fields are deliberately independent:

* ``l1_label=anomaly`` + ``l2_label=nonfault`` is still an L1 **positive**
  (an anomaly, just not a mechanical fault).
* Only ``l1_label=normal`` is a confirmed L1 false alarm and may be used as a
  normal example.
* Blank / ``uncertain`` never becomes training truth.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .data import Annotation

L1_LABELS = {"normal", "anomaly", "uncertain"}
L2_LABELS = {"fault", "nonfault", "uncertain", "not_trained"}
REVIEW_FIELDS = [
    "event_id", "audio_id", "source_audio_path", "predicted_start_s", "predicted_end_s",
    "corrected_start_s", "corrected_end_s", "l1_label", "l2_label", "fault_type",
    "note", "reviewed_at", "source_model_version", "clip_path",
]


@dataclass
class ReviewRecord:
    event_id: str
    audio_id: str
    source_audio_path: str = ""
    predicted_start_s: float = 0.0
    predicted_end_s: float = 0.0
    corrected_start_s: float | None = None
    corrected_end_s: float | None = None
    l1_label: str = ""
    l2_label: str = "not_trained"
    fault_type: str = ""
    note: str = ""
    reviewed_at: str = ""
    source_model_version: str = ""
    clip_path: str = ""

    def to_row(self) -> dict:
        row = asdict(self)
        row["corrected_start_s"] = "" if self.corrected_start_s is None else self.corrected_start_s
        row["corrected_end_s"] = "" if self.corrected_end_s is None else self.corrected_end_s
        return row

    def effective_interval(self) -> tuple[float, float]:
        a = self.corrected_start_s if self.corrected_start_s not in (None, "") else self.predicted_start_s
        b = self.corrected_end_s if self.corrected_end_s not in (None, "") else self.predicted_end_s
        return float(a), float(b)

    def l1_feedback(self) -> int | None:
        """1 = anomaly positive, 0 = confirmed normal, None = unknown."""
        if self.l1_label == "anomaly":
            return 1
        if self.l1_label == "normal":
            return 0
        return None


def read_reviews(path: str | Path) -> list[ReviewRecord]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(_row_to_review(row))
    return out


def _num(value, default=None):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_to_review(row: dict) -> ReviewRecord:
    return ReviewRecord(
        event_id=(row.get("event_id") or "").strip(),
        audio_id=(row.get("audio_id") or "").strip(),
        source_audio_path=(row.get("source_audio_path") or "").strip(),
        predicted_start_s=_num(row.get("predicted_start_s"), 0.0),
        predicted_end_s=_num(row.get("predicted_end_s"), 0.0),
        corrected_start_s=_num(row.get("corrected_start_s"), None),
        corrected_end_s=_num(row.get("corrected_end_s"), None),
        l1_label=(row.get("l1_label") or "").strip(),
        l2_label=(row.get("l2_label") or "not_trained").strip(),
        fault_type=(row.get("fault_type") or "").strip(),
        note=(row.get("note") or "").strip(),
        reviewed_at=(row.get("reviewed_at") or "").strip(),
        source_model_version=(row.get("source_model_version") or "").strip(),
        clip_path=(row.get("clip_path") or "").strip(),
    )


def write_reviews(reviews: list[ReviewRecord], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for r in reviews:
            writer.writerow(r.to_row())
    return path


def validate_reviews(reviews: list[ReviewRecord]) -> list[str]:
    errors = []
    ids = [r.event_id for r in reviews]
    if len(ids) != len(set(ids)):
        errors.append("review 中存在重复 event_id")
    for r in reviews:
        if not r.event_id:
            errors.append("存在空 event_id")
        if not r.audio_id:
            errors.append(f"{r.event_id}: audio_id 为空")
        if r.predicted_end_s <= r.predicted_start_s:
            errors.append(f"{r.event_id}: predicted 起止顺序错误")
        a, b = r.effective_interval()
        if b <= a:
            errors.append(f"{r.event_id}: corrected 起止顺序错误")
        if r.l1_label and r.l1_label not in L1_LABELS:
            errors.append(f"{r.event_id}: 非法 l1_label={r.l1_label}")
        if r.l2_label and r.l2_label not in L2_LABELS:
            errors.append(f"{r.event_id}: 非法 l2_label={r.l2_label}")
        if r.l2_label == "fault" and not r.fault_type:
            errors.append(f"{r.event_id}: l2_label=fault 但 fault_type 为空，请填写具体类型")
    return errors


def export_template(events_payload: dict, out_csv: str | Path, out_json: str | Path | None = None) -> list[ReviewRecord]:
    audio = events_payload.get("audio", {})
    model = events_payload.get("model", {})
    audio_id = audio.get("id") or Path(audio.get("file", "audio")).stem
    rows = []
    for ev in events_payload.get("events", []):
        rows.append(
            ReviewRecord(
                event_id=ev.get("event_id", ""),
                audio_id=audio_id,
                source_audio_path=audio.get("path", ""),
                predicted_start_s=float(ev.get("start_s", 0.0)),
                predicted_end_s=float(ev.get("end_s", 0.0)),
                l1_label="",
                l2_label="not_trained",
                source_model_version=str(model.get("version", "")),
                clip_path=(ev.get("clip") or {}).get("path", ""),
            )
        )
    write_reviews(rows, out_csv)
    if out_json:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps([r.to_row() for r in rows], ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return rows


def import_reviews(path: str | Path, out_csv: str | Path, out_json: str | Path | None = None) -> list[ReviewRecord]:
    incoming = read_reviews(path)
    errors = validate_reviews(incoming)
    if errors:
        raise ValueError("审核文件校验失败：\n" + "\n".join(errors))
    out_csv = Path(out_csv)
    existing = read_reviews(out_csv) if out_csv.exists() else []
    merged = {r.event_id: r for r in existing}
    for r in incoming:
        if not r.reviewed_at:
            r.reviewed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        merged[r.event_id] = r
    rows = sorted(merged.values(), key=lambda r: (r.audio_id, r.predicted_start_s, r.event_id))
    write_reviews(rows, out_csv)
    if out_json:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps([r.to_row() for r in rows], ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return rows


def reviews_to_annotations(reviews: list[ReviewRecord], version: str = "review") -> list[Annotation]:
    """Only confirmed L1 labels become interval annotations.

    anomaly+nonfault stays an anomaly; normal becomes normal; uncertain is skipped.
    """
    out = []
    for r in reviews:
        y = r.l1_feedback()
        if y is None:
            continue
        a, b = r.effective_interval()
        out.append(
            Annotation(
                annotation_id=f"review_{r.event_id}",
                audio_id=r.audio_id,
                start_s=a,
                end_s=b,
                label="anomaly" if y == 1 else "normal",
                precision="interval",
                fault_type=r.fault_type,
                note=f"l2={r.l2_label}; {r.note}".strip(),
                reviewed_at=r.reviewed_at,
                annotator="human_review",
                version=version,
            )
        )
    return out


def review_summary(reviews: list[ReviewRecord]) -> dict:
    from collections import Counter

    return {
        "total": len(reviews),
        "l1": dict(Counter(r.l1_label or "pending" for r in reviews)),
        "l2": dict(Counter(r.l2_label or "pending" for r in reviews)),
        "l1_positive": sum(1 for r in reviews if r.l1_feedback() == 1),
        "l1_negative": sum(1 for r in reviews if r.l1_feedback() == 0),
        "unknown": sum(1 for r in reviews if r.l1_feedback() is None),
    }
