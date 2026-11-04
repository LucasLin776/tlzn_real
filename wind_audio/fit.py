"""Fit L1 v3 parameters from **stable-region** annotations.

The target is the opposite of the historical one: the labels say *where the
spectrum is stable*, and the detector must find everything that is not.  The
fitting objective is therefore

    learn the distribution of the stability statistics on stable frames,
    put every threshold in its tail,
    then detect the complement and score it against the complement truth.

Two kinds of numbers are produced:

* **trigger / quiet thresholds** -- placed at the ``1 - alpha`` quantile of the
  corresponding statistic on stable frames, i.e. at a target false-trigger rate
  of ``alpha`` per stable frame (:func:`calibrate`).
* **stability tolerances** -- ``fluct_scale`` / ``drift_scale`` at the 99.5th
  percentile of the per-band fluctuation and drift measured *on stable frames*,
  so the stability criterion learns what "stable" means for this material, and
  ``stable_band_frac`` at the 1st percentile of the share of bands still inside
  tolerance, so ~99% of stable frames are called stable.

Nothing uses the non-stable regions while calibrating; the complement is only
used afterwards for scoring.  Files are split by time (the tail of the session
is held out) because adjacent one-minute clips share operating conditions.

    wind_audio l1-fit --audio-dir ... --labels ... --out config/l1_fit.json
"""
from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .audio import load_wav
from .detectors import StabilityDetector
from .evaluate import aggregate_file_metrics, evaluate_file_events
from .features import FeatureConfig, extract_features
from .markers import import_markers
from .stability import analyze
from .state import track_events

STAT_KEYS = ("change_z", "exceed", "slow_ratio", "excess_ratio", "z",
             "fluct_ratio", "drift_ratio", "peak_change", "fluct_frac", "drift_frac")

# Structural coordinates worth searching on a new batch.  Everything else is a
# scale / unit convention and stays at its documented default.
DEFAULT_GRID = [
    {"event": {"segment_mode": "state", "confirm_s": 1.5}},
    {"event": {"segment_mode": "state", "confirm_s": 2.5}},
    {"event": {"segment_mode": "state", "confirm_s": 4.0}},
    {"event": {"segment_mode": "state", "confirm_s": 4.0}, "stability": {"noise_s": 1.0}},
    {"event": {"segment_mode": "trigger", "release_mode": "stable"}},
    {},
]


@dataclass
class Calibration:
    alpha: float
    stable_frames: int
    quantiles: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _quantile(values, q: float, default: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 100:
        return default
    return float(np.quantile(values, q))


def calibrate(stats: dict, band_z: float, alpha: float = 1e-3,
              alpha_stable: float = 1e-2, limits: dict | None = None) -> Calibration:
    """Place every threshold in the tail of the stable-frame distribution."""
    lim = {"change_z": (1.5, 8.0), "exceed": (0.02, 0.60), "slow": (1.5, 8.0),
           "excess": (0.05, 0.80), "stable_change": (0.4, 5.0),
           "fluct_scale": (0.5, 10.0), "drift_scale": (0.5, 10.0)}
    if limits:
        lim.update(limits)
    q = 1.0 - alpha
    change_on = _quantile(stats["change_z"], q, 3.0)
    exceed_on = _quantile(stats["exceed"], q, 0.15)
    slow_on = _quantile(stats["slow_ratio"], q, 2.5)
    excess_quiet = _quantile(stats["excess_ratio"], 0.98, 0.2)
    change_quiet = _quantile(stats["change_z"], 0.98, change_on * 0.5)
    exceed_quiet = _quantile(stats["exceed"], 0.98, exceed_on * 0.5)
    stable_change = _quantile(stats["peak_change"], q, 1.5)

    def clip(value, key):
        lo, hi = lim[key]
        return float(np.clip(value, lo, hi))

    thresholds = {
        "band_z": float(band_z),
        "trigger_change_z": clip(change_on, "change_z"),
        "trigger_exceed": clip(exceed_on, "exceed"),
        "slow_ratio_trigger": clip(slow_on, "slow"),
        "quiet_excess": clip(excess_quiet, "excess"),
        "quiet_change_ratio": float(np.clip(change_quiet / max(change_on, 1e-9), 0.2, 0.9)),
        "quiet_exceed_ratio": float(np.clip(exceed_quiet / max(exceed_on, 1e-9), 0.2, 0.9)),
        # The stability tolerances follow their own, tighter target: a stable
        # frame must be called stable with probability 1 - alpha_stable.
        "stable_change_z": clip(_quantile(stats["peak_change"], 1 - alpha_stable, 1.5), "stable_change"),
        "fluct_scale": clip(_quantile(stats["fluct_ratio"], 1 - alpha_stable, 1.2), "fluct_scale"),
        "drift_scale": clip(_quantile(stats["drift_ratio"], 1 - alpha_stable, 1.2), "drift_scale"),
    }
    quantiles = {"trigger_alpha": alpha, "stable_alpha": alpha_stable,
                 "change_z": q, "exceed": q, "slow_ratio": q,
                 "excess_ratio": 0.98, "peak_change": 1 - alpha_stable,
                 "fluct_ratio": 1 - alpha_stable, "drift_ratio": 1 - alpha_stable}
    return Calibration(alpha=float(alpha), stable_frames=int(np.size(stats["change_z"])),
                       quantiles=quantiles, thresholds=thresholds)


def band_fraction_threshold(stats: dict, alpha: float = 1e-2) -> float:
    """Share of bands that must stay inside tolerance so stable frames pass."""
    share = np.minimum(np.asarray(stats["fluct_frac"]), np.asarray(stats["drift_frac"]))
    return float(np.clip(_quantile(share, alpha, 0.75), 0.50, 0.98))


class _Corpus:
    """Per-file cached mel power so a search does not re-run the STFT."""

    def __init__(self, audio_dir, labels_dir, feature_config: FeatureConfig):
        with tempfile.TemporaryDirectory(prefix="wind_audio_fit_") as tmp:
            self.records, self.anns, self.warnings = import_markers(
                audio_dir, labels_dir, Path(tmp) / "manifest.csv",
                Path(tmp) / "annotations.csv", pairing="sorted", stable_markers=True)
        self.cfg = feature_config
        self.bundles: dict[str, SimpleNamespace] = {}
        self.stable: dict[str, np.ndarray] = {}
        self.duration: dict[str, float] = {}
        self.by_id: dict[str, list] = {}
        for ann in self.anns:
            self.by_id.setdefault(ann.audio_id, []).append(ann)

    def prepare(self) -> None:
        for record in self.records:
            x, sr = load_wav(Path(record.path), target_sr=self.cfg.sr)
            bundle = extract_features(x, sr, self.cfg, record.audio_id)
            self.bundles[record.audio_id] = SimpleNamespace(
                audio_id=record.audio_id, duration_s=bundle.duration_s, times=bundle.times,
                mel_times=bundle.mel_times,
                mel_power=np.asarray(bundle.mel_power, dtype=np.float32), config=bundle.config)
            self.duration[record.audio_id] = bundle.duration_s
            mask = np.zeros(len(bundle.times), bool)
            for ann in self.by_id.get(record.audio_id, []):
                if ann.label == "normal":
                    mask |= (bundle.times >= ann.start_s) & (bundle.times < ann.end_s)
            self.stable[record.audio_id] = mask

    def record(self, audio_id):
        return next(r for r in self.records if r.audio_id == audio_id)


def _stable_stats(corpus: _Corpus, ids, stability_cfg, margin_s: float = 0.3) -> dict:
    """Concatenate statistics of stable frames (boundaries eroded by margin)."""
    out: dict[str, list] = {k: [] for k in STAT_KEYS}
    for audio_id in ids:
        frames = analyze(corpus.bundles[audio_id], stability_cfg)
        mask = corpus.stable[audio_id] & frames.valid
        if margin_s > 0:
            step = float(np.median(np.diff(frames.times))) if len(frames.times) > 1 else 0.02
            k = int(round(margin_s / step))
            core = mask.copy()
            for shift in range(1, k + 1):
                core[shift:] &= mask[:-shift]
                core[:-shift] &= mask[shift:]
            mask = core
        if not mask.any():
            continue
        for key in ("change_z", "exceed", "slow_ratio", "excess_ratio",
                    "peak_change", "fluct_frac", "drift_frac"):
            out[key].append(getattr(frames, key)[mask])
        for key in ("z", "fluct_ratio", "drift_ratio"):
            out[key].append(getattr(frames, key)[mask].ravel())
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}


def _band_z(z_values, alpha: float, limits=(2.0, 6.0)) -> float:
    if np.size(z_values) < 1000:
        return 3.0
    return float(np.clip(np.quantile(z_values, 1 - alpha), *limits))


def _evaluate(corpus: _Corpus, ids, stability_cfg, event_cfg) -> dict:
    rows = []
    for audio_id in ids:
        frames = analyze(corpus.bundles[audio_id], stability_cfg)
        raw, _ = track_events(frames, event_cfg, corpus.duration[audio_id])
        spans = [{"start_s": e["start_s"], "end_s": e["end_s"]} for e in raw]
        rows.append(evaluate_file_events(corpus.record(audio_id), corpus.anns, spans))
    overall = aggregate_file_metrics(rows, with_groups=False)
    tm = overall["time_metrics"]
    p, r = tm["precision"] or 0.0, tm["recall"] or 0.0
    return {"event_recall": overall["event_recall"], "event_precision": overall["event_precision"],
            "event_f1": overall["event_f1"], "time_accuracy": overall["time_accuracy"],
            "time_recall": tm["recall"], "time_precision": tm["precision"],
            "time_f1": (2 * p * r / (p + r)) if (p + r) else None,
            "coverage": overall["coverage"], "trailing_alarm_s": overall["trailing_alarm_s"],
            "n_pred": overall["n_pred_events"], "n_gt": overall["n_gt_events"]}


@dataclass
class FitReport:
    grid: list = field(default_factory=list)
    grid_results: list = field(default_factory=list)
    best: dict = field(default_factory=dict)
    train: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def split_ids(records, val_fraction: float = 0.3) -> tuple[list[str], list[str]]:
    """Temporal split: the last ``val_fraction`` of the session is held out."""
    ordered = [r.audio_id for r in sorted(records, key=lambda r: str(r.path))]
    n_val = max(1, int(round(len(ordered) * val_fraction)))
    return ordered[:-n_val], ordered[-n_val:]


def _select(entry: dict) -> tuple:
    """Rank candidates on the actual target: how much of the non-stable time is found.

    Time F1 is the primary criterion because the target is a *region*, not a
    point event; time accuracy and event F1 break ties.
    """
    v = entry["validation"]
    return (v["time_f1"] or 0.0, v["time_accuracy"] or 0.0, v["event_f1"] or 0.0)


def fit(audio_dir, labels_dir, base: StabilityDetector | None = None,
        grid: list[dict] | None = None, val_fraction: float = 0.3,
        alpha: float = 1e-3, alpha_stable: float = 1e-2,
        verbose: bool = True) -> tuple[StabilityDetector, FitReport]:
    base = base or StabilityDetector()
    corpus = _Corpus(audio_dir, labels_dir, base.feature_config)
    corpus.prepare()
    train_ids, val_ids = split_ids(corpus.records, val_fraction)
    candidates = grid or DEFAULT_GRID
    report = FitReport(grid=candidates, files={"fit": train_ids, "validation": val_ids})
    best = None
    for index, candidate in enumerate(candidates, 1):
        stability_cfg = replace(base.stability, **candidate.get("stability", {}))
        stats = _stable_stats(corpus, train_ids, stability_cfg)
        cal = calibrate(stats, _band_z(stats["z"], alpha), alpha=alpha,
                        alpha_stable=alpha_stable)
        th = cal.thresholds
        stability_cfg = replace(stability_cfg, band_z=th["band_z"],
                                stable_change_z=th["stable_change_z"],
                                fluct_scale=th["fluct_scale"],
                                drift_scale=th["drift_scale"])
        # Second pass: the band-share criterion depends on the new tolerances.
        stats2 = _stable_stats(corpus, train_ids, stability_cfg)
        stable_band_frac = band_fraction_threshold(stats2, alpha=alpha_stable)
        cal.thresholds["stable_band_frac"] = stable_band_frac
        stability_cfg = replace(stability_cfg, stable_band_frac=stable_band_frac)
        event_cfg = replace(base.events, **candidate.get("event", {}),
                            trigger_change_z=th["trigger_change_z"],
                            trigger_exceed=th["trigger_exceed"],
                            slow_ratio_trigger=th["slow_ratio_trigger"],
                            quiet_change_ratio=th["quiet_change_ratio"],
                            quiet_exceed_ratio=th["quiet_exceed_ratio"],
                            quiet_excess=th["quiet_excess"])
        validation = _evaluate(corpus, val_ids, stability_cfg, event_cfg)
        train = _evaluate(corpus, train_ids, stability_cfg, event_cfg)
        entry = {"candidate": candidate, "calibration": cal.to_dict(),
                 "validation": validation, "train": train}
        report.grid_results.append(entry)
        if verbose:
            print(f"[{index}/{len(candidates)}] {candidate or 'base'} -> "
                  f"val timeF1 {validation['time_f1']:.3f} acc {validation['time_accuracy']:.3f} "
                  f"cov {validation['coverage']:.3f} eventF1 {validation['event_f1']:.3f} "
                  f"(train timeF1 {train['time_f1']:.3f})", flush=True)
        score = _select(entry)
        if best is None or score > best[0]:
            best = (score, stability_cfg, event_cfg, cal, entry)
    _, stability_cfg, event_cfg, cal, entry = best
    report.best = {"candidate": entry["candidate"], "thresholds": cal.thresholds,
                   "stability": stability_cfg.to_dict(), "event": event_cfg.to_dict()}
    report.calibration = cal.to_dict()
    report.train = entry["train"]
    report.validation = entry["validation"]
    return StabilityDetector(base.feature_config, stability_cfg, event_cfg), report
