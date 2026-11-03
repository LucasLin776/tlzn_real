"""Frozen embeddings + masked multi-label head + bounded exemplar replay."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .common import read_json, write_json
from .data import assert_continuation


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def scores(y, pred):
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    tp, fp, fn = int((y & pred).sum()), int((~y & pred).sum()), int((y & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else None
    return {"positive_support": int(y.sum()), "negative_support": int((~y).sum()),
            "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def select_memory(rows, X, classes, per_class):
    """Nearest-to-centroid exemplars, round-robin across recording groups."""
    selected = set()
    for label in [None] + [c["id"] for c in classes]:
        candidates = [i for i, r in enumerate(rows)
                      if (r.anomaly == 0 if label is None else label in r.positive_types)]
        if not candidates:
            continue
        centroid = unit(X[candidates].mean(axis=0))
        ordered = sorted(candidates, key=lambda i: (-float(X[i] @ centroid), rows[i].sample_id))
        groups = {}
        for i in ordered:
            groups.setdefault(rows[i].group_id, []).append(i)
        chosen = []
        while len(chosen) < per_class and any(groups.values()):
            for group in groups.values():
                if group and len(chosen) < per_class:
                    chosen.append(group.pop(0))
        selected.update(chosen)
    unspecified = [i for i, r in enumerate(rows) if r.anomaly and not r.positive_types]
    selected.update(unspecified[:per_class])
    return sorted(selected)


class Model:
    def __init__(self, meta, weights, bias, prototypes, memory_X):
        self.meta, self.W, self.b = meta, weights, bias
        self.prototypes, self.memory_X = prototypes, memory_X

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(directory / "model.npz", W=self.W, b=self.b,
                            prototypes=self.prototypes, memory_X=self.memory_X)
        write_json(directory / "meta.json", self.meta)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = read_json(directory / "meta.json")
        if meta.get("schema_version") != 1:
            raise ValueError("不支持的 L2 模型格式")
        with np.load(directory / "model.npz", allow_pickle=False) as z:
            return cls(meta, z["W"], z["b"], z["prototypes"], z["memory_X"])

    def predict(self, X, top_k=3):
        X = unit(np.asarray(X))
        p, similarities = sigmoid(X @ self.W + self.b), X @ self.prototypes.T
        cfg = self.meta["calibration"]
        output = []
        for i in range(len(X)):
            types = [c["id"] for j, c in enumerate(self.meta["classes"], 1)
                     if cfg[j]["active"] and p[i, j] >= cfg[j]["threshold"]
                     and similarities[i, j] >= cfg[j]["min_similarity"]]
            binary = cfg[0]
            if binary["active"] and p[i, 0] >= binary["threshold"] and types:
                status = "known_anomaly"
            elif (binary["active"] and p[i, 0] < binary["normal_threshold"]
                  and similarities[i, 0] >= binary["min_similarity"]):
                status, types = "likely_normal", []
            else:
                status, types = "unknown", []
            nearest = []
            if len(self.memory_X) and top_k:
                sim = self.memory_X @ X[i]
                for k in np.argsort(-sim, kind="stable")[:top_k]:
                    nearest.append({**self.meta["memory"][int(k)], "cosine": float(sim[k])})
            output.append({"status": status, "types": types,
                           "anomaly_score": float(p[i, 0]),
                           "type_scores": {c["id"]: float(p[i, j + 1])
                                           for j, c in enumerate(self.meta["classes"])},
                           "keep_l1_alarm": True, "needs_review": True, "similar_cases": nearest})
        return output


def train(dataset, features, *, version, parent=None, epochs=400, lr=0.03,
          memory_per_class=32, replay=True, distill=0.5, full_retrain=False):
    if not version or epochs < 1 or lr <= 0 or memory_per_class < 1 or distill < 0:
        raise ValueError("训练参数无效")
    if parent:
        assert_continuation(parent.meta, dataset)
        if parent.meta["encoder"] != features.encoder:
            raise ValueError("续训必须使用相同 PANNs 权重与窗口配置")
        if parent.meta["version"] == version:
            raise ValueError("续训请使用新版本名")
    rows = [r for r in dataset.samples if r.split == "train"]
    val = [r for r in dataset.samples if r.split == "val"]
    if not rows or not val:
        raise ValueError("需要已审核 train 和独立 val 样本")
    Xall, XV = features.matrix(rows), features.matrix(val)
    yall, mall = dataset.targets(rows)
    if not np.any(yall[:, 0] == 0) or not np.any(yall[:, 0] == 1):
        raise ValueError("L2 train 至少需要已审核正常和异常样本")
    selected = list(range(len(rows)))
    changed = {r.sample_id for r in rows}
    if parent and not full_retrain:
        old = parent.meta["partitions"]
        changed = {r.sample_id for r in rows
                   if r.sample_id not in old or r.signature() != old[r.sample_id]["signature"]}
        replay_ids = {r["sample_id"] for r in parent.meta["memory"]} if replay else set()
        selected = [i for i, r in enumerate(rows) if r.sample_id in changed | replay_ids]
        if not changed:
            raise ValueError("没有新增或修订的训练样本；请先人工审核")
    X, y, mask = Xall[selected], yall[selected], mall[selected]
    selected_rows = [rows[i] for i in selected]
    columns = len(dataset.classes) + 1
    W = np.zeros((X.shape[1], columns), dtype=np.float64)
    b = np.zeros(columns, dtype=np.float64)
    old_columns = 0
    if parent and not full_retrain:
        old_columns = parent.W.shape[1]
        W[:, :old_columns], b[:old_columns] = parent.W, parent.b
    # Each recording has comparable influence, each observed +/- label balanced.
    counts = Counter(r.group_id for r in selected_rows)
    group_weight = np.array([1 / counts[r.group_id] for r in selected_rows])
    weight = mask * group_weight[:, None]
    for j in range(columns):
        for value in (0, 1):
            idx = (y[:, j] == value) & (mask[:, j] > 0)
            total = weight[idx, j].sum()
            if total:
                weight[idx, j] /= total
    weight /= np.maximum(weight.sum(axis=0, keepdims=True), 1)
    teacher = sigmoid(X @ parent.W + parent.b) if old_columns else None
    distill_rows = np.array([r.sample_id not in changed for r in selected_rows])
    if old_columns:
        # Do not constrain previously unseen/masked class outputs.
        dmask = np.array([c["active"] for c in parent.meta["calibration"]], float)
        dmask = distill_rows[:, None] * dmask[None, :]
    else:
        dmask = None
    mw, vw, mb, vb = np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b)
    for step in range(1, epochs + 1):
        p = sigmoid(X @ W + b)
        gradient = (p - y) * weight
        if old_columns and distill:
            gradient[:, :old_columns] += (distill * (p[:, :old_columns] - teacher) * dmask
                                         / max(int(distill_rows.sum()), 1))
        gw, gb = X.T @ gradient + 1e-4 * W, gradient.sum(axis=0)
        mw, vw = 0.9 * mw + 0.1 * gw, 0.999 * vw + 0.001 * gw * gw
        mb, vb = 0.9 * mb + 0.1 * gb, 0.999 * vb + 0.001 * gb * gb
        W -= lr * (mw / (1 - 0.9 ** step)) / (np.sqrt(vw / (1 - 0.999 ** step)) + 1e-8)
        b -= lr * (mb / (1 - 0.9 ** step)) / (np.sqrt(vb / (1 - 0.999 ** step)) + 1e-8)
    # Prototypes and replay are train-only, never validation/test exemplars.
    prototypes = np.zeros((columns, X.shape[1]), dtype=np.float32)
    for j in range(columns):
        positive = (yall[:, j] == (0 if j == 0 else 1)) & (mall[:, j] == 1)
        if positive.any():
            prototypes[j] = unit(Xall[positive].mean(axis=0))
    yv, mv = dataset.targets(val)
    pv, cosv = sigmoid(XV @ W + b), XV @ prototypes.T
    calibration = []
    for j in range(columns):
        train_known, val_known = mall[:, j] > 0, mv[:, j] > 0
        support = {"train_pos": int(((yall[:, j] == 1) & train_known).sum()),
                   "train_neg": int(((yall[:, j] == 0) & train_known).sum()),
                   "val_pos": int(((yv[:, j] == 1) & val_known).sum()),
                   "val_neg": int(((yv[:, j] == 0) & val_known).sum())}
        group_support = [len({r.group_id for i, r in enumerate(rr)
                            if mm[i, j] and yy[i, j] == value})
                         for rr, yy, mm in ((rows, yall, mall), (val, yv, mv))
                         for value in (0, 1)]
        active = min(support.values()) >= 2 and min(group_support) >= 2
        ref = (yall[:, j] == (0 if j == 0 else 1)) & train_known
        minimum = float(np.quantile(Xall[ref] @ prototypes[j], 0.05)) if ref.any() else 1.0
        item = {"active": active, "threshold": 0.5, "normal_threshold": 0.0,
                "min_similarity": minimum, "support": support,
                "reason": "ready" if active else "每个 train/val 正负类别需至少两个独立来源"}
        if active:
            known_y, prob, similarity = yv[val_known, j], pv[val_known, j], cosv[val_known, j]
            candidates = np.unique(np.r_[0.0, prob, np.nextafter(1.0, 2.0)])
            choices = []
            for t in candidates:
                pred = prob >= t
                if j:
                    pred &= similarity >= minimum
                metrics = scores(known_y, pred)
                if j == 0 and metrics["recall"] < 0.95:
                    continue
                choices.append((metrics["f1"], metrics["precision"], float(t)))
            if choices:
                item["threshold"] = max(choices)[2]
                if j == 0:
                    item["normal_threshold"] = min(0.5, item["threshold"])
            else:
                item["active"] = False
        calibration.append(item)
    memory_ids = select_memory(rows, Xall, dataset.classes, memory_per_class)
    memory = [{"sample_id": rows[i].sample_id, "event_id": rows[i].event_id,
               "audio_path": rows[i].audio_path, "start_s": rows[i].start_s, "end_s": rows[i].end_s,
               "group_id": rows[i].group_id, "anomaly": rows[i].anomaly,
               "positive_types": rows[i].positive_types} for i in memory_ids]
    meta = {"schema_version": 1, "version": version, "classes": dataset.classes,
            "encoder": features.encoder, "dataset_fingerprint": dataset.fingerprint,
            "partitions": dataset.partitions, "parent_version": parent.meta["version"] if parent else None,
            "calibration": calibration, "memory": memory,
            "training": {"epochs": epochs, "lr": lr, "distill": distill,
                         "replay": replay, "full_retrain": full_retrain,
                         "selected": len(selected), "new_or_revised": len(changed),
                         "memory_per_class": memory_per_class,
                         "sample_ids": [r.sample_id for r in selected_rows]}}
    return Model(meta, W, b, prototypes, Xall[memory_ids])


def evaluate(model, dataset, features, split="test"):
    if split not in {"val", "test"}:
        raise ValueError("评估仅支持 val/test")
    if features.encoder != model.meta["encoder"]:
        raise ValueError("评估特征与模型编码器不一致")
    model_ids = [c["id"] for c in model.meta["classes"]]
    if dataset.ids[:len(model_ids)] != model_ids:
        raise ValueError("评估类别表必须兼容模型")
    # Prevent accidentally evaluating on the model's training sources.
    train_groups = {p["group"] for p in model.meta["partitions"].values() if p["split"] == "train"}
    train_audio = {p["audio"] for p in model.meta["partitions"].values() if p["split"] == "train"}
    rows = [r for r in dataset.samples if r.split == split]
    if not rows:
        raise ValueError("评估集合为空")
    if any(r.group_id in train_groups or r.audio_sha256 in train_audio for r in rows):
        raise ValueError("评估样本与模型训练来源重叠")
    predictions = model.predict(features.matrix(rows))
    per_class = {}
    for c in dataset.ids:
        known = [i for i, r in enumerate(rows) if c in r.positive_types or c in r.negative_types]
        if known:
            per_class[c] = scores([c in rows[i].positive_types for i in known],
                                  [c in predictions[i]["types"] for i in known])
    positives = [i for i, r in enumerate(rows) if r.anomaly]
    normal = [i for i, r in enumerate(rows) if not r.anomaly]
    active = {c["id"] for j, c in enumerate(model.meta["classes"], 1)
              if model.meta["calibration"][j]["active"]}
    unknown = [i for i, r in enumerate(rows) if r.anomaly and r.positive_types
               and not set(r.positive_types) & active]
    ratio = lambda numerator, denom: numerator / len(denom) if denom else None
    return {"scope": "conditional_on_reviewed_L1_candidates_not_end_to_end",
            "version": model.meta["version"], "split": split, "samples": len(rows),
            "per_class": per_class,
            "macro_f1": float(np.mean([m["f1"] for m in per_class.values()
                                      if m["positive_support"]])) if any(
                                          m["positive_support"] for m in per_class.values()) else None,
            "suggested_anomaly_retention": ratio(sum(predictions[i]["status"] != "likely_normal"
                                                     for i in positives), positives),
            "normal_rejection_rate": ratio(sum(predictions[i]["status"] == "likely_normal"
                                                for i in normal), normal),
            "unknown_rejection_recall": ratio(sum(predictions[i]["status"] == "unknown"
                                                  for i in unknown), unknown),
            "unknown_support": len(unknown), "alarms_automatically_removed": 0,
            "predictions": [{"sample_id": r.sample_id, **p} for r, p in zip(rows, predictions)]}
