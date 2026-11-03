"""Grouped train/validation/test splits.

The default group is the original recording/batch (``source_id``).  A group can
never appear in two splits, so adjacent windows and augmented copies cannot
leak.  ``--group-by device`` is the right option when measuring cross-turbine
generalisation.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .data import AudioRecord, Annotation


@dataclass
class SplitSet:
    train: list[str]
    val: list[str]
    test: list[str]
    group_by: str = "source"
    seed: int = 42
    ratios: list[float] = field(default_factory=lambda: [0.6, 0.2, 0.2])
    groups: dict[str, list[str]] = field(default_factory=dict)
    created_at: str = ""

    def ids(self, name: str) -> list[str]:
        return list(getattr(self, name))

    def summary(self) -> dict:
        return {
            "train": len(self.train), "val": len(self.val), "test": len(self.test),
            "group_by": self.group_by, "seed": self.seed, "ratios": self.ratios,
            "groups": len(self.groups),
        }


def group_key(record: AudioRecord, group_by: str = "source") -> str:
    if group_by == "device":
        return record.device_id or record.source_id or record.audio_id
    if group_by == "time":
        return (record.recorded_at or "")[:10] or record.source_id or record.audio_id
    if group_by == "audio":
        return record.audio_id
    return record.source_id or record.audio_id


def _has_anomaly(audio_ids: list[str], anns: list[Annotation]) -> bool:
    ids = set(audio_ids)
    return any(a.audio_id in ids and a.label == "anomaly" for a in anns)


def make_splits(
    records: list[AudioRecord],
    anns: list[Annotation],
    *,
    group_by: str = "source",
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> SplitSet:
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("划分比例之和必须为 1")
    groups: dict[str, list[str]] = {}
    for rec in records:
        groups.setdefault(group_key(rec, group_by), []).append(rec.audio_id)
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError(
            f"只有 {len(keys)} 个独立分组（group_by={group_by}），至少需要 3 个才能划分 train/val/test"
        )
    rng = random.Random(seed)
    rng.shuffle(keys)
    n = len(keys)
    n_train = max(1, int(round(n * ratios[0])))
    n_val = max(1, int(round(n * ratios[1])))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    train_keys = keys[:n_train]
    val_keys = keys[n_train : n_train + n_val]
    test_keys = keys[n_train + n_val :]
    if not test_keys:
        test_keys = [train_keys.pop()]
    # Try to put at least one anomaly group in val and test.
    anomaly_keys = [k for k in keys if _has_anomaly(groups[k], anns)]
    for target in (val_keys, test_keys):
        if anomaly_keys and not any(k in anomaly_keys for k in target):
            donor = next((k for k in anomaly_keys if k in train_keys), None)
            replace = next((k for k in target if k not in anomaly_keys), target[0])
            if donor:
                train_keys.remove(donor)
                target.remove(replace)
                target.append(donor)
                train_keys.append(replace)

    def ids(keys_):
        out = []
        for k in keys_:
            out.extend(groups[k])
        return sorted(out)

    return SplitSet(
        train=ids(train_keys), val=ids(val_keys), test=ids(test_keys),
        group_by=group_by, seed=seed, ratios=list(ratios),
        groups={k: sorted(groups[k]) for k in keys},
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def assert_no_leakage(split: SplitSet) -> None:
    tr, va, te = set(split.train), set(split.val), set(split.test)
    if tr & va or tr & te or va & te:
        raise AssertionError("split 之间存在 audio_id 泄漏")
    for name, ids in (("train", tr), ("val", va), ("test", te)):
        if not ids:
            raise AssertionError(f"{name} 为空")
    for g, members in split.groups.items():
        present = {name for name, s in (("train", tr), ("val", va), ("test", te)) if set(members) & s}
        if len(present) > 1:
            raise AssertionError(f"分组 {g} 跨集合: {present}")


def save_splits(split: SplitSet, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(split), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_splits(path: str | Path) -> SplitSet:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return SplitSet(**data)
