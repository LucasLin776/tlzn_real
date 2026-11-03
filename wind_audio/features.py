"""Per-file audio features with local context and gain normalisation.

Feature design (all fitted per file, no labels involved):

* 64 log-mel bands, 20 Hz - 16 kHz (configurable).
* 10 frame descriptors: gain-normalised RMS, centroid, flatness, flux,
  energy rate and 5 band-energy shares.
* For every target time (default 0.1 s grid):
  - current log-mel + descriptors
  - mean/std/min/max of descriptors over the context window
  - mean/std of log-mel over the context window
  - right-minus-left descriptor differences at 0.25/0.5/1/2 s

Fixed per-file gain/EQ is removed by subtracting the file's median band
levels and median RMS.  This keeps within-file changes (bursts, switches,
sustained anomalies) while reducing false alarms from recording gain.  It is
a per-file global offset, never a per-window normalisation, so a sustained
anomaly is not absorbed as "new normal".
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from .audio import frame_rms_db, frame_times, mel_filterbank, stft
from .data import AudioRecord, Annotation, LabelMask, label_mask

DESCRIPTOR_NAMES = [
    "rms_db_norm", "centroid_hz", "flatness", "flux", "energy_rate_db_s",
    "share_0_200", "share_200_1k", "share_1k_4k", "share_4k_10k", "share_10k_up",
]
BANDS = [(0, 200), (200, 1000), (1000, 4000), (4000, 10000), (10000, 30000)]


@dataclass(frozen=True)
class FeatureConfig:
    sr: int = 44100
    n_fft: int = 2048
    hop: int = 441
    n_mels: int = 64
    fmin: float = 20.0
    fmax: float = 16000.0
    target_step: float = 0.1
    context_s: float = 1.0
    scales: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    normalize: str = "per_file"

    def __post_init__(self):
        if self.sr < 8000:
            raise ValueError("sr 必须 >= 8000")
        if self.n_fft < 256:
            raise ValueError("n_fft 必须 >= 256")
        if self.hop <= 0 or self.hop > self.n_fft:
            raise ValueError("hop 必须 > 0 且 <= n_fft")
        if not (0 <= self.fmin < self.fmax <= self.sr / 2):
            raise ValueError("要求 0 <= fmin < fmax <= sr/2")
        if self.target_step < self.hop / self.sr:
            raise ValueError("target_step 不能小于帧移")
        if self.target_step > 5:
            raise ValueError("target_step 过大")
        if self.context_s < self.target_step or self.context_s > 10:
            raise ValueError("context_s 必须在 [target_step, 10] 秒")
        if self.normalize not in {"none", "per_file"}:
            raise ValueError("normalize 只能是 none 或 per_file")
        if not self.scales or min(self.scales) <= 0:
            raise ValueError("scales 必须为正数")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["scales"] = list(self.scales)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureConfig":
        allowed = {f.name for f in fields(cls)}
        clean = {k: v for k, v in (data or {}).items() if k in allowed}
        if "scales" in clean:
            clean["scales"] = tuple(float(x) for x in clean["scales"])
        return cls(**clean)

    def fingerprint(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def assert_compatible(self, other: "FeatureConfig") -> None:
        if self.to_dict() != other.to_dict():
            raise ValueError(
                "特征配置与模型不一致，拒绝静默运行：\n"
                f"  模型: {json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)}\n"
                f"  当前: {json.dumps(other.to_dict(), ensure_ascii=False, sort_keys=True)}"
            )


@dataclass
class FeatureBundle:
    audio_id: str
    duration_s: float
    sr: int
    times: np.ndarray
    X: np.ndarray
    feature_names: list[str]
    mel_db: np.ndarray
    mel_norm: np.ndarray
    mel_times: np.ndarray
    freqs: np.ndarray
    spec_db: np.ndarray
    rms_db: np.ndarray
    descriptors: np.ndarray
    descriptor_names: list[str]
    ref_rms_db: float
    ref_mel_db: np.ndarray
    config: FeatureConfig = field(default_factory=FeatureConfig)

    def matrix(self) -> np.ndarray:
        return self.X


def _descriptor_matrix(spec: np.ndarray, freqs: np.ndarray, mel_n: np.ndarray, rms_n: np.ndarray, frame_dt: float) -> np.ndarray:
    total = spec.sum(axis=1) + 1e-12
    centroid = (spec * freqs).sum(axis=1) / total
    flatness = np.exp(np.mean(np.log(spec + 1e-12), axis=1)) / (np.mean(spec, axis=1) + 1e-12)
    flux = np.zeros(spec.shape[0], dtype=np.float64)
    if spec.shape[0] > 1:
        diff = np.diff(mel_n, axis=0)
        flux[1:] = np.sqrt(np.mean(diff**2, axis=1))
    rate = np.gradient(rms_n.astype(np.float64), frame_dt) if len(rms_n) > 1 else np.zeros_like(rms_n)
    cols = [rms_n.astype(np.float64), centroid, flatness, flux, rate]
    for lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        cols.append(spec[:, m].sum(axis=1) / total)
    return np.stack(cols, axis=1).astype(np.float32)


def extract_features(
    x: np.ndarray,
    sr: int,
    cfg: FeatureConfig,
    audio_id: str,
) -> FeatureBundle:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        raise ValueError(f"{audio_id}: 空音频")
    duration = len(x) / sr
    spec = np.abs(stft(x, cfg.n_fft, cfg.hop, center=True)) ** 2
    fb = mel_filterbank(sr, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax)
    mel_db = (10.0 * np.log10(spec @ fb.T + 1e-10)).astype(np.float32)
    freqs = np.fft.rfftfreq(cfg.n_fft, 1.0 / sr)
    mel_times = frame_times(mel_db.shape[0], cfg.hop, sr, center=True, n_fft=cfg.n_fft)
    spec_db = (10.0 * np.log10(spec + 1e-12)).astype(np.float32)
    rms_db = frame_rms_db(x, cfg.n_fft, cfg.hop, center=True)
    n = min(len(mel_db), len(spec_db), len(rms_db))
    mel_db, spec, spec_db, rms_db = mel_db[:n], spec[:n], spec_db[:n], rms_db[:n]
    mel_times = mel_times[:n]

    if cfg.normalize == "per_file":
        ref_rms = float(np.median(rms_db))
        ref_mel = np.median(mel_db, axis=0).astype(np.float32)
        rms_n = rms_db - ref_rms
        mel_n = mel_db - ref_mel
    else:
        ref_rms = 0.0
        ref_mel = np.zeros(cfg.n_mels, dtype=np.float32)
        rms_n = rms_db
        mel_n = mel_db

    frame_dt = cfg.hop / sr
    D = _descriptor_matrix(spec, freqs, mel_n, rms_n, frame_dt)

    step = cfg.target_step
    n_target = max(1, int(np.ceil(duration / step)))
    times = (np.arange(n_target, dtype=np.float64) + 0.5) * step
    times = times[times < duration]
    if len(times) == 0:
        times = np.array([min(duration / 2, step / 2)], dtype=np.float64)
    idx = np.clip(np.rint(times * sr / cfg.hop).astype(int), 0, n - 1)

    rows: list[np.ndarray] = []
    names: list[str] = []
    names += [f"mel_cur_{i:02d}" for i in range(cfg.n_mels)]
    names += [f"desc_cur_{name}" for name in DESCRIPTOR_NAMES]
    names += [f"ctx_mean_{name}" for name in DESCRIPTOR_NAMES]
    names += [f"ctx_std_{name}" for name in DESCRIPTOR_NAMES]
    names += [f"ctx_min_{name}" for name in DESCRIPTOR_NAMES]
    names += [f"ctx_max_{name}" for name in DESCRIPTOR_NAMES]
    names += [f"mel_ctx_mean_{i:02d}" for i in range(cfg.n_mels)]
    names += [f"mel_ctx_std_{i:02d}" for i in range(cfg.n_mels)]
    for scale in cfg.scales:
        names += [f"lr_{scale:g}s_{name}" for name in DESCRIPTOR_NAMES]

    for j, t in enumerate(times):
        lo = int(np.searchsorted(mel_times, t - cfg.context_s, side="left"))
        hi = int(np.searchsorted(mel_times, t + cfg.context_s, side="right"))
        lo = max(0, min(lo, n - 1))
        hi = max(lo + 1, min(hi, n))
        seg_D = D[lo:hi]
        seg_mel = mel_n[lo:hi]
        cur = np.concatenate([mel_n[idx[j]], D[idx[j]]])
        ctx = np.concatenate(
            [
                seg_D.mean(axis=0),
                seg_D.std(axis=0),
                seg_D.min(axis=0),
                seg_D.max(axis=0),
                seg_mel.mean(axis=0),
                seg_mel.std(axis=0),
            ]
        )
        lr = []
        for scale in cfg.scales:
            a0 = int(np.searchsorted(mel_times, t - scale, side="left"))
            a1 = int(np.searchsorted(mel_times, t, side="right"))
            b1 = int(np.searchsorted(mel_times, t + scale, side="right"))
            a0, a1, b1 = max(0, a0), max(1, a1), max(a1 + 1, b1)
            left = D[a0:min(a1, n)].mean(axis=0) if min(a1, n) > a0 else D[idx[j]]
            right = D[min(a1, n - 1) : min(b1, n)].mean(axis=0) if min(b1, n) > min(a1, n - 1) else D[idx[j]]
            lr.append(right - left)
        rows.append(np.concatenate([cur, ctx, np.concatenate(lr)]).astype(np.float32))

    X = np.stack(rows, axis=0)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return FeatureBundle(
        audio_id=audio_id,
        duration_s=duration,
        sr=sr,
        times=times,
        X=X,
        feature_names=names,
        mel_db=mel_db,
        mel_norm=mel_n,
        mel_times=mel_times,
        freqs=freqs,
        spec_db=spec_db,
        rms_db=rms_db,
        descriptors=D,
        descriptor_names=DESCRIPTOR_NAMES,
        ref_rms_db=ref_rms,
        ref_mel_db=ref_mel,
        config=cfg,
    )


def labels_at_times(
    record: AudioRecord,
    anns: list[Annotation],
    times: np.ndarray,
    step_s: float = 0.1,
) -> np.ndarray:
    """Map a target grid onto an existing mask without inventing labels."""
    mask = label_mask(record, anns, step_s)
    if len(mask.times) == 0:
        return np.full(len(times), -1, dtype=np.int8)
    idx = np.clip(np.rint(times / mask.step_s - 0.5).astype(int), 0, len(mask.times) - 1)
    labels = mask.labels[idx].copy()
    labels[(times < 0) | (times >= record.duration_s)] = -1
    return labels


def cache_path(cache_dir: str | Path, audio_id: str, path: Path, cfg: FeatureConfig) -> Path:
    st = path.stat()
    key = hashlib.sha1(
        f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}|{cfg.fingerprint()}".encode("utf-8")
    ).hexdigest()[:16]
    return Path(cache_dir) / f"{audio_id}_{key}.npz"


def load_or_extract(
    path: str | Path,
    audio_id: str,
    cfg: FeatureConfig,
    cache_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, times, feature_names), using a deterministic per-file cache."""
    path = Path(path)
    cache = cache_path(cache_dir, audio_id, path, cfg)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["times"], [str(s) for s in z["feature_names"]]
    from .audio import load_wav

    x, sr = load_wav(path, target_sr=cfg.sr)
    bundle = extract_features(x, sr, cfg, audio_id)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        X=bundle.X,
        times=bundle.times,
        feature_names=np.array(bundle.feature_names, dtype=object),
        duration_s=np.array([bundle.duration_s]),
    )
    return bundle.X, bundle.times, bundle.feature_names


def build_matrix(
    records: list[AudioRecord],
    anns: list[Annotation],
    audio_ids: list[str],
    manifest_path: str | Path,
    cfg: FeatureConfig,
    cache_dir: str | Path,
) -> dict:
    """Assemble training/evaluation matrices from reviewed target cells only."""
    from .data import resolve_audio_path

    by_id = {r.audio_id: r for r in records}
    Xs, ys, groups, file_ids, times_all = [], [], [], [], []
    skipped_unknown = 0
    for audio_id in audio_ids:
        rec = by_id[audio_id]
        p = resolve_audio_path(rec, manifest_path)
        X, times, names = load_or_extract(p, audio_id, cfg, cache_dir)
        y = labels_at_times(rec, anns, times, cfg.target_step)
        valid = y != -1
        skipped_unknown += int((~valid).sum())
        Xs.append(X[valid])
        ys.append(y[valid].astype(np.int8))
        groups.append(np.array([audio_id] * int(valid.sum()), dtype=object))
        file_ids.append(np.array([audio_id] * int(valid.sum()), dtype=object))
        times_all.append(times[valid])
    if not Xs or sum(len(v) for v in Xs) == 0:
        return {
            "X": np.zeros((0, 0), dtype=np.float32),
            "y": np.zeros(0, dtype=np.int8),
            "groups": np.array([], dtype=object),
            "file_ids": np.array([], dtype=object),
            "times": np.zeros(0),
            "feature_names": [],
            "skipped_unknown": skipped_unknown,
        }
    return {
        "X": np.concatenate(Xs, axis=0),
        "y": np.concatenate(ys),
        "groups": np.concatenate(groups),
        "file_ids": np.concatenate(file_ids),
        "times": np.concatenate(times_all),
        "feature_names": names,
        "skipped_unknown": skipped_unknown,
    }
