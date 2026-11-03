"""Explicit local-checkpoint PANNs inference; no network or pickle execution.

CNN14 layer names follow qiuqiangkong/audioset_tagging_cnn. Its frozen STFT
and mel kernels are read directly from the official checkpoint, avoiding a
second, potentially inconsistent mel implementation. PyTorch is imported only when encoding audio.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np

from ..audio import load_wav
from .common import digest, file_hash


@dataclass(frozen=True)
class EncoderConfig:
    name: str = "panns_cnn14"
    revision: int = 1
    sample_rate: int = 32000
    window_seconds: float = 4.0
    stride_seconds: float = 2.0
    embedding_dim: int = 2048
    checkpoint_sha256: str = ""

    def __post_init__(self):
        if (self.name != "panns_cnn14" or self.revision != 1 or self.sample_rate != 32000
                or self.embedding_dim != 2048):
            raise ValueError("仅支持 32 kHz CNN14 2048 维检查点")
        if not (1 <= self.window_seconds <= 30 and
                0 < self.stride_seconds <= self.window_seconds):
            raise ValueError("窗口需在 1–30 秒，步长需大于零且不超过窗口")

    def to_dict(self):
        return asdict(self)


def windows(x, start, end, cfg: EncoderConfig):
    """Keep real context; zero-pad boundaries; never repeat or time-stretch."""
    if not (np.isfinite(start) and np.isfinite(end) and
            0 <= start < end <= len(x) / cfg.sample_rate + 1e-6):
        raise ValueError("事件区间越界")
    width = cfg.window_seconds
    if end - start <= width:
        starts = [(start + end - width) / 2]
    else:
        starts = list(np.arange(start, end - width, cfg.stride_seconds))
        starts.append(end - width)
    n = round(width * cfg.sample_rate)
    for t in starts:
        a = round(t * cfg.sample_rate)
        b = a + n
        out = np.zeros(n, dtype=np.float32)
        lo, hi = max(0, a), min(len(x), b)
        if hi > lo:
            out[lo - a:hi - a] = x[lo:hi]
        yield out


def _network(torch, state):
    nn, F = torch.nn, torch.nn.functional
    keys = ("spectrogram_extractor.stft.conv_real.weight",
            "spectrogram_extractor.stft.conv_imag.weight", "logmel_extractor.melW")
    if any(k not in state for k in keys):
        raise ValueError("不是受支持的官方 CNN14 检查点：缺少冻结前端")
    real, imag, mel = (state[k] for k in keys)
    if tuple(real.shape) != (513, 1, 1024) or imag.shape != real.shape or tuple(mel.shape) != (513, 64):
        raise ValueError("检查点前端不匹配 1024 FFT / 64 mel 配置")

    class Block(nn.Module):
        def __init__(self, a, b):
            super().__init__()
            self.conv1 = nn.Conv2d(a, b, 3, padding=1, bias=False)
            self.conv2 = nn.Conv2d(b, b, 3, padding=1, bias=False)
            self.bn1, self.bn2 = nn.BatchNorm2d(b), nn.BatchNorm2d(b)

        def forward(self, x, pool):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            return F.avg_pool2d(x, pool)

    class CNN14(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("stft_real", real, persistent=False)
            self.register_buffer("stft_imag", imag, persistent=False)
            self.register_buffer("mel_filter", mel, persistent=False)
            self.bn0 = nn.BatchNorm2d(64)
            channels = (1, 64, 128, 256, 512, 1024, 2048)
            for i in range(6):
                setattr(self, f"conv_block{i + 1}", Block(channels[i], channels[i + 1]))
            self.fc1 = nn.Linear(2048, 2048)
            self.fc_audioset = nn.Linear(2048, 527)

        def forward(self, wave):
            x = F.pad(wave.unsqueeze(1), (512, 512), mode="reflect")
            re = F.conv1d(x, self.stft_real, stride=320)
            im = F.conv1d(x, self.stft_imag, stride=320)
            power = (re.square() + im.square()).transpose(1, 2)
            x = (10 * torch.log10(torch.clamp(power @ self.mel_filter, min=1e-10))).unsqueeze(1)
            x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
            for i in range(6):
                x = getattr(self, f"conv_block{i + 1}")(x, (2, 2) if i < 5 else (1, 1))
            x = x.mean(dim=3)
            x = x.amax(dim=2) + x.mean(dim=2)
            return F.relu(self.fc1(x))

    model = CNN14()
    model.load_state_dict({k: v for k, v in state.items() if k not in keys}, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class PannsEncoder:
    def __init__(self, checkpoint, *, config=None, device="cpu"):
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError("请显式提供本地 CNN14 权重；程序不会自动下载")
        sha = file_hash(path)
        config = config or EncoderConfig()
        if config.checkpoint_sha256 and sha != config.checkpoint_sha256:
            raise ValueError("PANNs 检查点校验和与模型不一致")
        self.config = EncoderConfig(**{**config.to_dict(), "checkpoint_sha256": sha})
        try:
            import torch
        except (ImportError, OSError) as exc:
            raise RuntimeError("PyTorch 不可用或动态库损坏；请检查 post_training 中的 PyTorch 安装") from exc
        self.torch, self.device = torch, device
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise RuntimeError("需要支持 weights_only=True 的 PyTorch；禁止退回不安全加载") from exc
        state = obj.get("model", obj)
        self.model = _network(torch, state).to(device)

    def encode(self, waveform, start, end):
        vecs = []
        with self.torch.inference_mode():
            for window in windows(waveform, start, end, self.config):
                tensor = self.torch.from_numpy(window[None]).to(self.device)
                vecs.append(self.model(tensor).cpu().numpy()[0])
        vector = np.mean(vecs, axis=0).astype(np.float32)
        norm = np.linalg.norm(vector)
        if not np.isfinite(vector).all() or norm <= 1e-12:
            raise ValueError("PANNs 输出不是有效非零特征")
        return vector / norm


class Features:
    def __init__(self, ids, content_keys, X, encoder):
        self.ids = [str(x) for x in ids]
        self.keys = [str(x) for x in content_keys]
        self.X = np.asarray(X, dtype=np.float32)
        self.encoder = encoder
        if (len(self.ids) != len(set(self.ids)) or len(self.keys) != len(self.ids)
                or self.X.shape != (len(self.ids), encoder["embedding_dim"])
                or not np.isfinite(self.X).all()):
            raise ValueError("特征缓存 ID、维度或数值不合法")
        norms = np.linalg.norm(self.X, axis=1)
        if np.any(norms < 1e-9):
            raise ValueError("特征不能为零")
        self.X = self.X / norms[:, None]
        self.index = {sid: i for i, sid in enumerate(self.ids)}

    def matrix(self, samples):
        result = []
        for row in samples:
            if row.sample_id not in self.index:
                raise ValueError(f"缺少特征: {row.sample_id}")
            i = self.index[row.sample_id]
            if self.keys[i] != row.content_key():
                raise ValueError("音频或事件区间变更，请重新提取特征")
            result.append(self.X[i])
        return np.stack(result) if result else np.empty((0, self.X.shape[1]), np.float32)

    @property
    def fingerprint(self):
        return digest({"encoder": self.encoder, "ids": self.ids, "keys": self.keys,
                       "matrix": __import__("hashlib").sha256(self.X.tobytes()).hexdigest()})

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as f:
            np.savez_compressed(f, ids=np.asarray(self.ids), keys=np.asarray(self.keys),
                                X=self.X, encoder=np.asarray(json.dumps(self.encoder)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(z["ids"], z["keys"], z["X"], json.loads(str(z["encoder"])))


def embed_dataset(dataset, encoder):
    dataset.check_audio()
    # Group files without changing the dataset's stable order.
    by_file = {}
    for r in dataset.samples:
        by_file.setdefault(r.audio_path, []).append(r)
    output = {}
    for path, rows in by_file.items():
        waveform, _ = load_wav(path, encoder.config.sample_rate)
        for r in rows:
            output[r.sample_id] = encoder.encode(waveform, r.start_s, r.end_s)
    return Features([r.sample_id for r in dataset.samples],
                    [r.content_key() for r in dataset.samples],
                    np.stack([output[r.sample_id] for r in dataset.samples]), encoder.config.to_dict())
