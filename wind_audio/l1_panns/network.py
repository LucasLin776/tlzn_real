"""Independent frozen CNN14 blocks 1--3 and a strict temporal bottleneck.

Only local, explicitly supplied official weights are accepted. The deeper
CNN14 blocks / classifier are not used. No L2 imports or modifications.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .config import NetworkConfig
from .data import file_hash


class ConvBlock(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, outputs, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(outputs, outputs, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outputs)
        self.bn2 = nn.BatchNorm2d(outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, 2)


class FrozenPanns(nn.Module):
    """Frozen official frontend + CNN14 shallow/mid blocks, time preserved.

    Per-frame logmel centering removes absolute level BEFORE the frozen CNN.
    It is deliberately recorded as a preprocessing change from AudioSet.
    Frozen features are L2-normalized per block/time, never over the recording.
    """
    def __init__(self, checkpoint, cfg=None, expected_sha=None):
        super().__init__()
        self.cfg = cfg or NetworkConfig()
        path = Path(checkpoint)
        if not path.is_file():
            raise ValueError("请显式提供本地官方 CNN14 权重；不自动下载或随机初始化")
        self.sha256 = file_hash(path)
        if expected_sha and self.sha256 != expected_sha:
            raise ValueError("PANNs 权重哈希不匹配")
        obj = torch.load(path, map_location="cpu", weights_only=True)
        state = obj.get("model", obj)
        names = ("spectrogram_extractor.stft.conv_real.weight",
                 "spectrogram_extractor.stft.conv_imag.weight", "logmel_extractor.melW")
        shapes = ((513, 1, 1024), (513, 1, 1024), (513, 64))
        for name, shape in zip(names, shapes):
            if name not in state or tuple(state[name].shape) != shape or not torch.isfinite(state[name]).all():
                raise ValueError(f"官方 CNN14 前端缺失/不兼容: {name}")
        self.register_buffer("stft_real", state[names[0]].float())
        self.register_buffer("stft_imag", state[names[1]].float())
        self.register_buffer("mel_filter", state[names[2]].float())
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        wanted = {k for k in self.state_dict() if k not in {"stft_real", "stft_imag", "mel_filter"}}
        missing = wanted - set(state)
        if missing:
            raise ValueError(f"官方 CNN14 缺少参数: {sorted(missing)}")
        selected = {k: state[k] for k in wanted}
        if any(not torch.isfinite(v).all() for v in selected.values()):
            raise ValueError("PANNs 参数存在非有限值")
        selected.update(stft_real=self.stft_real, stft_imag=self.stft_imag, mel_filter=self.mel_filter)
        self.load_state_dict(selected, strict=True)
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        # A parent .train() must NEVER update PANNs BatchNorm running statistics.
        return super().train(False)

    @torch.no_grad()
    def forward(self, wave):
        if wave.ndim != 2 or wave.shape[-1] < self.cfg.samples_for_frames(1):
            raise ValueError("PANNs 时序前向需要至少一个原生时间格")
        x = wave.unsqueeze(1)
        re = F.conv1d(x, self.stft_real, stride=self.cfg.hop)
        im = F.conv1d(x, self.stft_imag, stride=self.cfg.hop)
        power = (re.square() + im.square()).transpose(1, 2)
        mel = 10 * torch.log10((power @ self.mel_filter).clamp_min(1e-10))
        mel = mel - mel.mean(dim=-1, keepdim=True) + self.cfg.level_center_db
        x = self.bn0(mel.unsqueeze(1).transpose(1, 3)).transpose(1, 3)
        scales = []
        for i in range(1, 4):
            x = getattr(self, f"conv_block{i}")(x)
            z = x.mean(dim=-1)  # only frequency pooled; time never globally pooled
            if i < 3:
                z = F.avg_pool1d(z, 2 ** (3 - i))
            scales.append(F.normalize(z, dim=1, eps=1e-8))
        return torch.cat(scales, dim=1)


def dynamics(features, cfg):
    """Fixed target: short/long signed changes plus symmetric local residual.

    There is no absolute feature/level branch and no learned target projection.
    A constant embedding (including a new constant background) maps to zero.
    Padding is invalid context and must be removed by the caller's halo/mask.
    """
    diffs = []
    for lag in (cfg.short_lag, cfg.long_lag):
        shifted = F.pad(features, (lag, 0), mode="replicate")[:, :, :features.shape[-1]]
        diffs.append(features - shifted)
    radius = cfg.local_radius
    mean = F.avg_pool1d(F.pad(features, (radius, radius), mode="replicate"), 2 * radius + 1, stride=1)
    return torch.cat([*diffs, features - mean], dim=1)


class TemporalDAE(nn.Module):
    """1344 -> 96 -> 24 -> 96 -> 1344; dilations 1,2,2,1; RF=13.

    No identity/skip paths. Bias-free + zero-preserving GELU makes a constant
    spectrum's zero dynamics a zero reconstruction independently of training.
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(cfg.input_dim, cfg.hidden, 3, padding=1, bias=False), nn.GELU(),
            nn.Conv1d(cfg.hidden, cfg.bottleneck, 3, dilation=2, padding=2, bias=False), nn.GELU(),
            nn.Conv1d(cfg.bottleneck, cfg.hidden, 3, dilation=2, padding=2, bias=False), nn.GELU(),
            nn.Conv1d(cfg.hidden, cfg.input_dim, 3, padding=1, bias=False))

    def forward(self, x):
        return self.layers(x)


def residual_score(reconstruction, fixed_target):
    # Pointwise RMS, with no median filtering that would erase a one-frame impulse.
    return (reconstruction - fixed_target).square().mean(dim=1).sqrt()
