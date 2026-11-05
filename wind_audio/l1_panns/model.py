"""Training, immutable artifacts, normal-only calibration, and shared prediction."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from . import MODEL_TYPE
from .config import EventConfig, NetworkConfig
from .data import (assert_disjoint, contained, digest, file_hash, load_audio, load_dataset,
                   natural_key, normal_crops, partition, provenance, read_json, write_json)
from .events import segment_scores
from .network import FrozenPanns, TemporalDAE, dynamics, residual_score

PREPROCESSING = {"waveform": "PCM mono arithmetic channel mean",
                 "resampling": "local_windowed_sinc_radius16_cutoff0.94_v1",
                 "stft": "official CNN14 kernels, center=False; no reflected waveform",
                 "logmel": "official melW, 10log10 floor1e-10, per-frame mel mean removed",
                 "features": "CNN14 B1/B2/B3 frequency mean; align 4/2/1 avg pooling; channel L2",
                 "dynamics": "backward short/long differences, centered local mean residual",
                 "normalization": "normal-train-only per-channel RMS, no centering or clipping",
                 "score": "pointwise RMS reconstruction residual, no smoothing"}


def new_directory(path):
    path = Path(path)
    if path.exists():
        raise ValueError("输出路径已存在；请使用新目录，禁止覆盖历史模型/结果")
    path.mkdir(parents=True)
    return path


class Model:
    def __init__(self, encoder, head, cfg, scales, metadata, device="cpu"):
        self.encoder = encoder.to(device).eval()
        self.head = head.to(device).eval()
        self.cfg, self.meta, self.device = cfg, metadata, device
        scales = np.asarray(scales, np.float32)
        if scales.shape != (cfg.input_dim,) or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("训练标准化缺失或无效")
        self.scales = torch.as_tensor(scales, device=device)[None, :, None]
        self.events = EventConfig(**metadata["event_config"])

    def require_ready(self, calibrated=True):
        if self.meta.get("trained") is not True or self.meta.get("training", {}).get("optimizer_steps", 0) <= 0:
            raise ValueError("L1 PANNs 模型尚未训练；禁止随机推理/回退旧 L1")
        if calibrated and not self.meta.get("calibration"):
            raise ValueError("L1 PANNs 模型尚未正常校准；请先显式运行 l1-panns-calibrate")
        if calibrated:
            t = self.meta["calibration"]["thresholds"]
            if not (all(math.isfinite(t[k]) for k in ("low", "high", "impulse"))
                    and 0 < t["low"] <= t["high"] <= t["impulse"]):
                raise ValueError("无效校准阈值")

    @classmethod
    def load(cls, path, *, checkpoint=None, device="cpu", calibrated=True):
        path = Path(path)
        path = path / "model.json" if path.is_dir() else path
        meta = read_json(path)
        if meta.get("model_type") != MODEL_TYPE or meta.get("schema_version") != 4:
            raise ValueError("不是 panns_temporal_dae_v1 模型；不自动回退")
        cfg = NetworkConfig(**meta["network_config"])
        if meta.get("preprocessing") != PREPROCESSING or meta.get("timing") != cfg.timing():
            raise ValueError("模型预处理/时间支持版本不兼容")
        # Fail before even opening the external PANNs checkpoint.
        if meta.get("trained") is not True or meta.get("training", {}).get("optimizer_steps", 0) <= 0:
            raise ValueError("L1 PANNs 模型尚未训练")
        if calibrated and not meta.get("calibration"):
            raise ValueError("L1 PANNs 模型尚未正常校准")
        weights = path.parent / "temporal.pt"
        if file_hash(weights) != meta["temporal_sha256"]:
            raise ValueError("时序权重哈希不匹配")
        encoder = FrozenPanns(checkpoint or meta["panns_checkpoint"], cfg, meta["panns_sha256"])
        head = TemporalDAE(cfg)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        if any(not torch.isfinite(v).all() for v in state.values()):
            raise ValueError("网络权重包含非有限值")
        head.load_state_dict(state, strict=True)
        result = cls(encoder, head, cfg, meta["normalization_scale"], meta, device)
        result.require_ready(calibrated)
        return result

    def save(self, directory):
        self.require_ready(calibrated=False)
        if any(not torch.isfinite(v).all() for v in self.head.state_dict().values()):
            raise ValueError("网络权重非有限，拒绝保存")
        out = new_directory(directory)
        with (out / "temporal.pt").open("xb") as f:
            torch.save({k: v.detach().cpu() for k, v in self.head.state_dict().items()}, f)
        self.meta = {**self.meta, "temporal_sha256": file_hash(out / "temporal.pt")}
        write_json(out / "model.json", self.meta)

    @torch.inference_mode()
    def score_wave(self, waveform, *, resampler_margin=0.0):
        """Full audio, bounded feature chunks with exact global pool alignment.

        No labels, old detector, online normalization, or threshold adaptation.
        Scores at padded/incomplete context are NaN and explicitly invalid.
        """
        self.require_ready(calibrated=False)
        x = np.asarray(waveform, np.float32)
        if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
            raise ValueError("需要非空有限单声道波形")
        cfg = self.cfg
        n = max(0, ((len(x) - cfg.n_fft) // cfg.hop + 1) // 8)
        times = cfg.offset_s + np.arange(n) * cfg.step_s
        radius = cfg.context_radius_s + resampler_margin
        valid = (times - radius >= 0) & (times + radius <= len(x) / cfg.sample_rate)
        scores = np.full(n, np.nan, np.float64)
        self.encoder.eval()
        self.head.eval()
        for a in range(0, n, cfg.core_frames):
            b = min(n, a + cfg.core_frames)
            if not valid[a:b].any():
                continue
            left, right = max(0, a - cfg.halo), min(n, b + cfg.halo)
            start = left * cfg.stride_samples
            end = start + cfg.samples_for_frames(right - left)
            wave = torch.as_tensor(x[start:end], device=self.device)[None]
            target = dynamics(self.encoder(wave), cfg) / self.scales
            prediction = self.head(target)
            score = residual_score(prediction, target)[0].cpu().numpy()
            scores[a:b] = score[a - left:b - left]
        scores[~valid] = np.nan
        if not np.isfinite(scores[valid]).all():
            raise ValueError("网络产生非有限残差；不能静默判正常")
        return times, scores, valid

    def predict(self, audio):
        """Sole detection/evaluation prediction entry. Accepts an audio path only."""
        self.require_ready()
        waveform, info, margin = load_audio(audio, self.cfg.sample_rate)
        times, scores, valid = self.score_wave(waveform, resampler_margin=margin)
        result = segment_scores(times, scores, valid, info.duration_s,
                                self.meta["calibration"]["thresholds"], self.events, self.cfg.step_s)
        states = result.pop("states")
        sha, model_id = file_hash(audio), digest(self.meta)
        for event in result["events"]:
            event["event_id"] = Path(audio).stem + "_" + digest(
                [sha, model_id, event["start_s"], event["end_s"]])[:20]
        return {"schema_version": "4.1", "mode": "l1-detect",
                "audio": {"id": Path(audio).stem, "file": Path(audio).name,
                          "path": str(Path(audio).resolve()), "sha256": sha,
                          "duration_s": info.duration_s, "sr": self.cfg.sample_rate,
                          "source_sr": info.sample_rate},
                "model": {"model_type": MODEL_TYPE, "version": MODEL_TYPE, "fingerprint": model_id,
                          "panns_sha256": self.meta["panns_sha256"],
                          "thresholds": self.meta["calibration"]["thresholds"],
                          "event_config": self.events.to_dict(), "requires_training": True},
                **result,
                "label": {"intervals": [[e["start_s"], e["end_s"]] for e in result["events"]],
                          "count": len(result["events"]), "rule": "network_residual_no_collapse",
                          "unknown_intervals": [[e["start_s"], e["end_s"]] for e in result["unknown_regions"]]},
                "scores": {"time_s": times.tolist(), "values": [float(s) if v else None
                                                                   for s, v in zip(scores, valid)],
                           "valid": valid.tolist(), "state": states.tolist()},
                "diagnostics": {**self.cfg.timing(), "context_radius_s": self.cfg.context_radius_s + margin,
                                "context_insufficient": not bool(valid.any()),
                                "unknown_is_normal": False, "anomaly_source": "network_residual_only"},
                "annotations": {"regions": [], "gt_events": [], "source": None}}


def _normal_features(rows, cfg, encoder, device):
    for row in rows:
        waveform, _, margin = load_audio(row["audio_path"], cfg.sample_rate)
        for a, b in normal_crops(row, cfg, margin):
            with torch.no_grad():
                wave = torch.as_tensor(waveform[a:b], device=device)[None]
                target = dynamics(encoder(wave), cfg).detach()
            yield target, {"audio_id": row["audio_id"], "start_s": a / cfg.sample_rate,
                           "end_s": b / cfg.sample_rate}


def train(dataset, checkpoint, output, *, cfg=None, event_cfg=None, epochs=20,
          lr=0.001, noise_std=0.0, augmentation_reviewed=False, seed=0, device="cpu"):
    """Explicitly invoked only; never called by import/check/detection/tests."""
    if Path(output).exists():
        raise ValueError("训练输出必须为新路径")
    if type(epochs) is not int or epochs <= 0 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("epochs/lr 必须为正数")
    if not math.isfinite(noise_std) or not 0 <= noise_std <= 0.05:
        raise ValueError("轻微动态特征噪声 std 需在 [0,0.05] 内")
    if noise_std > 0 and not augmentation_reviewed:
        raise ValueError("噪声增强需显式 --augmentation-reviewed 确认符合正常定义")
    cfg, event_cfg = cfg or NetworkConfig(), event_cfg or EventConfig()
    data = load_dataset(dataset)
    rows = partition(data, "train")
    torch.manual_seed(seed)
    encoder = FrozenPanns(checkpoint, cfg).to(device).eval()
    square_sum, count, crops = np.zeros(cfg.input_dim, np.float64), 0, []
    for target, crop in _normal_features(rows, cfg, encoder, device):
        core = target[0, :, cfg.halo:-cfg.halo].cpu().numpy().astype(np.float64)
        square_sum += np.sum(core ** 2, axis=1)
        count += core.shape[1]
        crops.append(crop)
    if not count:
        raise ValueError("没有完整特征及网络上下文均落在明确正常区间内的样本")
    scale = np.maximum(np.sqrt(square_sum / count), cfg.scale_floor).astype(np.float32)
    scale_tensor = torch.as_tensor(scale, device=device)[None, :, None]
    head = TemporalDAE(cfg).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    steps, losses = 0, []
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        head.train()
        order = [rows[i] for i in rng.permutation(len(rows))]
        loss_sum, frames = 0.0, 0
        for target, _ in _normal_features(order, cfg, encoder, device):
            target = (target / scale_tensor).detach()  # NEVER train the target or encoder
            noisy = target + torch.randn_like(target) * noise_std if noise_std else target
            reconstruction = head(noisy)
            loss = (reconstruction[:, :, cfg.halo:-cfg.halo] - target[:, :, cfg.halo:-cfg.halo]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("训练损失非有限，未保存模型")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            steps += 1
            n = target.shape[-1] - 2 * cfg.halo
            frames += n
            loss_sum += float(loss.detach()) * n
        losses.append(loss_sum / frames)
        print(f"epoch {epoch + 1}/{epochs}: normal reconstruction MSE={losses[-1]:.6g}", flush=True)
    metadata = {"schema_version": 4, "model_type": MODEL_TYPE, "trained": True,
                "network_config": cfg.to_dict(), "event_config": event_cfg.to_dict(),
                "preprocessing": PREPROCESSING, "timing": cfg.timing(),
                "panns_checkpoint": str(Path(checkpoint).resolve()), "panns_sha256": encoder.sha256,
                "normalization_scale": scale.tolist(), "calibration": None,
                "training": {"dataset_fingerprint": digest(data), "provenance": provenance(rows),
                             "optimizer_steps": steps, "epochs": epochs, "lr": lr, "seed": seed,
                             "noise_std": noise_std, "augmentation_reviewed": augmentation_reviewed,
                             "normal_frames": count, "crops": crops, "loss": losses,
                             "unused_audio_ids": sorted({r["audio_id"] for r in rows}
                                                        - {c["audio_id"] for c in crops}),
                             "panns_frozen": True, "batchnorm_frozen": True}}
    model = Model(encoder, head, cfg, scale, metadata, device)
    model.save(output)
    return model


def calibrate(model_path, dataset, output, *, checkpoint=None, device="cpu",
              quantiles=(0.95, 0.995, 0.9995), min_frames=1000):
    if Path(output).exists():
        raise ValueError("校准输出必须为新路径")
    if (len(quantiles) != 3 or not all(math.isfinite(q) for q in quantiles)
            or not 0 < quantiles[0] < quantiles[1] < quantiles[2] < 1
            or type(min_frames) is not int or min_frames < 2):
        raise ValueError("需要递增的三个 (0,1) 分位数及至少 2 个校准帧")
    model = Model.load(model_path, checkpoint=checkpoint, device=device, calibrated=False)
    if model.meta.get("calibration"):
        raise ValueError("请从未校准的训练产物创建新的校准版本，不修改现有阈值")
    data = load_dataset(dataset)
    rows = partition(data, "calibration")
    assert_disjoint(rows, model.meta["training"]["provenance"])
    values, used = [], []
    for row in rows:
        waveform, _, margin = load_audio(row["audio_path"], model.cfg.sample_rate)
        times, scores, valid = model.score_wave(waveform, resampler_margin=margin)
        mask = valid & contained(times, row["intervals"], model.cfg.context_radius_s + margin)
        if mask.any():
            values.append(scores[mask])
            used.append({"audio_id": row["audio_id"], "normal_frames": int(mask.sum())})
    if not values or sum(len(v) for v in values) < min_frames:
        raise ValueError("完整正常上下文的校准帧不足；未保存阈值")
    thresholds = np.maximum(np.quantile(np.concatenate(values), quantiles, method="higher"), 1e-6)
    model.meta = {**model.meta, "calibration": {
        "thresholds": dict(zip(("low", "high", "impulse"), thresholds.tolist())),
        "quantiles": list(quantiles), "method": "normal_only_empirical_higher",
        "normal_frames": sum(len(v) for v in values), "recordings": used,
        "dataset_fingerprint": digest(data), "provenance": provenance(rows),
        "note": "correlated temporal samples; quantiles are not a guaranteed event false alarm rate"}}
    model.save(output)
    return model


def evaluate(model, data, split="test", *, on_file=None):
    """Truth is applied only AFTER model.predict; no unknown->normal conversion."""
    if split not in {"val", "test"}:
        raise ValueError("独立评估只允许显式 val/test 集")
    model.require_ready()
    rows = sorted(partition(data, split), key=lambda r: natural_key(r["audio_id"]))
    assert_disjoint(rows, model.meta["training"]["provenance"] + model.meta["calibration"]["provenance"])
    totals = {"tp_s": 0.0, "fp_s": 0.0, "tn_s": 0.0, "fn_s": 0.0,
              "reviewed_s": 0.0, "abstained_reviewed_s": 0.0,
              "reviewed_normal_s": 0.0, "reviewed_anomaly_s": 0.0,
              "abstained_normal_s": 0.0, "abstained_anomaly_s": 0.0}
    per_file = []
    for index, row in enumerate(rows, 1):
        prediction = model.predict(row["audio_path"])  # same exact path as detection
        times = np.asarray(prediction["scores"]["time_s"])
        state = np.asarray(prediction["scores"]["state"])
        step = model.cfg.step_s
        truth = np.full(len(times), -1, np.int8)
        truth[contained(times, row["intervals"], step / 2, "normal")] = 0
        truth[contained(times, row["intervals"], step / 2, "anomaly")] = 1
        metrics = {"tp_s": float(np.sum((truth == 1) & (state == 1)) * step),
                   "fp_s": float(np.sum((truth == 0) & (state == 1)) * step),
                   "tn_s": float(np.sum((truth == 0) & (state == 0)) * step),
                   "fn_s": float(np.sum((truth == 1) & (state == 0)) * step),
                   "reviewed_s": sum(r["end_s"] - r["start_s"] for r in row["intervals"]
                                     if r["label"] != "unknown")}
        classified = sum(metrics[k] for k in ("tp_s", "fp_s", "tn_s", "fn_s"))
        metrics["abstained_reviewed_s"] = max(0.0, metrics["reviewed_s"] - classified)
        for label, keys in (("normal", ("tn_s", "fp_s")), ("anomaly", ("tp_s", "fn_s"))):
            reviewed = sum(r["end_s"] - r["start_s"] for r in row["intervals"] if r["label"] == label)
            metrics[f"reviewed_{label}_s"] = reviewed
            metrics[f"abstained_{label}_s"] = max(0.0, reviewed - sum(metrics[k] for k in keys))
        for k in totals:
            totals[k] += metrics[k]
        item = {"audio_id": row["audio_id"], "metrics": metrics, "prediction": prediction}
        if on_file is not None:
            on_file(row, item, index, len(rows))
        per_file.append(item)
    tp, fp, fn = (totals[k] for k in ("tp_s", "fp_s", "fn_s"))
    return {"model_type": MODEL_TYPE, "split": split, "duration_metrics": totals,
            "precision_on_classified": tp / (tp + fp) if tp + fp else None,
            "recall_on_classified": tp / (tp + fn) if tp + fn else None,
            "anomaly_recall_including_abstentions": tp / totals["reviewed_anomaly_s"]
            if totals["reviewed_anomaly_s"] else None,
            "reviewed_coverage": 1 - totals["abstained_reviewed_s"] / totals["reviewed_s"]
            if totals["reviewed_s"] else None,
            "note": "时间格指标；unknown/边缘单独计入 abstention，不当正常，不是事件级准确率",
            "per_file": per_file}
