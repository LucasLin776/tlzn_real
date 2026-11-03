"""Versioned event datasets. Partial labels are explicitly masked."""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..audio import wav_info
from ..data import read_manifest, resolve_audio_path
from ..review import read_reviews, validate_reviews
from ..splits import assert_no_leakage, group_key, load_splits
from .common import digest, file_hash, read_json, write_json


def taxonomy(path) -> list[dict]:
    rows = read_json(path)["classes"]
    ids = [r["id"] for r in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("类别表必须非空，且 ID 不重复")
    for row in rows:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", row["id"]) or not row.get("name"):
            raise ValueError("类别 ID 使用小写英文/数字/下划线，且必须有 name")
    return [{"id": r["id"], "name": r["name"]} for r in rows]


def type_ids(value: str) -> list[str]:
    return sorted(set(x.strip() for x in (value or "").split(";") if x.strip()))


@dataclass
class Sample:
    sample_id: str
    audio_path: str
    audio_sha256: str
    start_s: float
    end_s: float
    group_id: str
    split: str
    anomaly: int
    positive_types: list[str]
    negative_types: list[str]
    review_revision: int = 1
    event_id: str = ""

    def content_key(self) -> str:
        return digest([self.audio_sha256, self.start_s, self.end_s])

    def signature(self) -> str:
        return digest({k: v for k, v in asdict(self).items()
                       if k not in {"audio_path", "event_id"}})


class Dataset:
    def __init__(self, classes: list[dict], samples: list[Sample]):
        self.classes, self.samples = classes, samples
        self.validate()

    @property
    def ids(self):
        return [c["id"] for c in self.classes]

    def validate(self):
        ids = self.ids
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("类别 ID 为空或重复")
        if not self.samples:
            raise ValueError("没有已审核样本")
        seen, content, groups, recordings = set(), set(), {}, {}
        for r in self.samples:
            if r.sample_id in seen or r.content_key() in content:
                raise ValueError("样本 ID 或音频区间重复；请先合并标签")
            seen.add(r.sample_id)
            content.add(r.content_key())
            if (not r.sample_id or not r.group_id or r.split not in {"train", "val", "test"}
                    or r.anomaly not in (0, 1) or r.review_revision < 1):
                raise ValueError("样本必须有稳定 ID、group、split、审核标签和正整数版本")
            if not re.fullmatch(r"[a-f0-9]{64}", r.audio_sha256):
                raise ValueError("无效音频校验和")
            if not (math.isfinite(r.start_s) and math.isfinite(r.end_s)
                    and 0 <= r.start_s < r.end_s):
                raise ValueError("无效事件时间")
            pos, neg = set(r.positive_types), set(r.negative_types)
            if pos & neg or (pos | neg) - set(ids) or (r.anomaly == 0 and pos):
                raise ValueError("类别标签冲突/未知，或正常事件包含异常类型")
            if groups.setdefault(r.group_id, r.split) != r.split:
                raise ValueError("同一来源分组跨 train/val/test")
            identity = (r.group_id, r.split)
            if recordings.setdefault(r.audio_sha256, identity) != identity:
                raise ValueError("同一原始音频被重复命名或跨分组/集合")

    def check_audio(self):
        checked = {}
        for r in self.samples:
            if r.audio_path not in checked:
                checked[r.audio_path] = (file_hash(r.audio_path), wav_info(r.audio_path).duration_s)
            sha, duration = checked[r.audio_path]
            if sha != r.audio_sha256 or r.end_s > duration + 1e-6:
                raise ValueError(f"音频已变更或标注越界: {r.sample_id}")

    @property
    def fingerprint(self):
        return digest({"classes": self.classes,
                       "samples": sorted((r.sample_id, r.signature()) for r in self.samples)})

    @property
    def partitions(self):
        return {r.sample_id: {"group": r.group_id, "split": r.split,
                             "audio": r.audio_sha256, "content": r.content_key(),
                             "revision": r.review_revision, "signature": r.signature()}
                for r in self.samples}

    def targets(self, samples=None):
        rows = self.samples if samples is None else samples
        y = np.zeros((len(rows), len(self.ids) + 1), dtype=np.float64)
        mask = np.zeros_like(y)
        for i, r in enumerate(rows):
            y[i, 0], mask[i, 0] = r.anomaly, 1
            for j, c in enumerate(self.ids, 1):
                if c in r.positive_types:
                    y[i, j], mask[i, j] = 1, 1
                elif c in r.negative_types:
                    mask[i, j] = 1
        return y, mask

    def save(self, path):
        path = Path(path).resolve()
        rows = []
        for r in self.samples:
            row = asdict(r)
            # Relative paths are resolved from the dataset file, never cwd.
            import os
            row["audio_path"] = os.path.relpath(r.audio_path, path.parent)
            rows.append(row)
        write_json(path, {"schema_version": 1, "classes": self.classes, "samples": rows})

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        obj = read_json(path)
        if obj.get("schema_version") != 1:
            raise ValueError("不支持的 L2 数据版本")
        rows = []
        for item in obj["samples"]:
            row = dict(item)
            row["audio_path"] = str((path.parent / row["audio_path"]).resolve())
            rows.append(Sample(**row))
        return cls(obj["classes"], rows)


def prepare(reviews_path, manifest_path, splits_path, taxonomy_path, previous=None) -> Dataset:
    classes = taxonomy(taxonomy_path)
    ids = {c["id"] for c in classes}
    records = {r.audio_id: r for r in read_manifest(manifest_path)}
    splits = load_splits(splits_path)
    assert_no_leakage(splits)
    assignments = {aid: part for part in ("train", "val", "test") for aid in splits.ids(part)}
    reviews = read_reviews(reviews_path)
    errors = validate_reviews(reviews)
    if errors:
        raise ValueError("\n".join(errors))
    rows, hashes, source_splits = {}, {}, {}
    old = {r.sample_id: r for r in previous.samples} if previous else {}
    if previous and [c["id"] for c in classes][:len(previous.ids)] != previous.ids:
        raise ValueError("类别表只能末尾追加")
    for review in reviews:
        if review.l1_label not in {"normal", "anomaly"}:
            continue
        if not review.reviewed_at:
            raise ValueError("L2 只接受带 reviewed_at 的人工审核；请先 review-import")
        if review.audio_id not in records or review.audio_id not in assignments:
            raise ValueError(f"审核引用未知/未划分音频: {review.audio_id}")
        rec = records[review.audio_id]
        if not rec.source_id:
            raise ValueError("L2 要求 manifest 显式提供 source_id，不能随机划分裁剪片段")
        part = assignments[review.audio_id]
        if source_splits.setdefault(rec.source_id, part) != part:
            raise ValueError("同一原始来源不能跨集合")
        path = resolve_audio_path(rec, manifest_path).resolve()
        if path not in hashes:
            hashes[path] = file_hash(path)
        a, b = review.effective_interval()
        sha = hashes[path]
        sid = "sample_" + digest([sha, a, b])[:24]
        pos, neg = type_ids(review.positive_types), type_ids(review.negative_types)
        if (set(pos) | set(neg)) - ids:
            raise ValueError(f"类别未注册: {review.event_id}")
        if review.types_complete or review.l1_label == "normal":
            # Materialise negatives NOW. Future classes remain unknown.
            neg = sorted(set(neg) | (ids - set(pos)))
        if sid in old and review.review_revision == old[sid].review_revision:
            # A complete review applies only to the taxonomy at review time.
            before = old[sid]
            if pos != before.positive_types or set(neg) & set(previous.ids) != set(before.negative_types):
                raise ValueError("标签修改后请递增 review_revision")
            neg = before.negative_types
        row = Sample(sid, str(path), sha, a, b, group_key(rec, splits.group_by),
                     assignments[review.audio_id], int(review.l1_label == "anomaly"),
                     pos, neg, review.review_revision, review.event_id)
        if sid in rows and rows[sid].signature() != row.signature():
            raise ValueError("同一音频区间有冲突审核，请先合并")
        rows[sid] = row
    dataset = Dataset(classes, list(rows.values()))
    dataset.check_audio()
    return dataset


def assert_continuation(parent: dict, dataset: Dataset):
    old_ids = [c["id"] for c in parent["classes"]]
    if dataset.ids[:len(old_ids)] != old_ids:
        raise ValueError("增量类别表只能末尾追加，禁止删除/重排旧 ID")
    current = dataset.partitions
    old = parent["partitions"]
    if set(old) - set(current):
        raise ValueError("增量更新需要累计数据快照，不能丢弃历史样本")
    group_splits, audio_splits = {}, {}
    for entry in list(old.values()) + list(current.values()):
        if group_splits.setdefault(entry["group"], entry["split"]) != entry["split"]:
            raise ValueError("历史分组不能移入/移出保留集")
        if audio_splits.setdefault(entry["audio"], entry["split"]) != entry["split"]:
            raise ValueError("历史录音不能跨保留集")
    for sid, before in old.items():
        after = current[sid]
        if any(before[k] != after[k] for k in ("group", "split", "audio", "content")):
            raise ValueError("历史样本的来源、区间和划分不可变")
        if before["signature"] != after["signature"]:
            if after["revision"] <= before["revision"]:
                raise ValueError("修订标签必须递增 review_revision")
            if before["split"] != "train":
                raise ValueError("保留集修订需重新建立基准，不支持静默续训")
