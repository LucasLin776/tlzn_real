"""Stable-region labels, state segmentation and stable-frame calibration."""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from wind_audio import cli
from wind_audio.audio import write_wav
from wind_audio.detectors import StabilityDetector
from wind_audio.features import FeatureConfig, extract_features
from wind_audio.fit import band_fraction_threshold, calibrate
from wind_audio.markers import _stable_complement, import_markers
from wind_audio.state import EventConfig

CFG = FeatureConfig(sr=8000, n_fft=512, hop=80, n_mels=32, fmax=3500)
SR = CFG.sr


def tone(duration=8.0, frequency=500.0, amplitude=0.1, noise=0.001, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * frequency * t)
            + rng.normal(0, noise, len(t))).astype(np.float32)


class StableLabelTests(unittest.TestCase):
    def test_complement_ignores_slivers(self):
        # stable [0.02,4] and [5,9.9] -> the only real gap is 4..5
        self.assertEqual(_stable_complement(10.0, [(0.02, 4.0), (5.0, 9.9)], 0.1),
                         [(4.0, 5.0)])
        # a fully stable file yields no anomaly at all
        self.assertEqual(_stable_complement(10.0, [(0.02, 10.0)], 0.1), [])

    def test_markers_as_stable_become_complement_anomalies(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audio = root / "audio"
            labels = root / "labels"
            audio.mkdir()
            labels.mkdir()
            write_wav(audio / "a.wav", tone(10.0), SR)
            with (labels / "a.csv").open("w", encoding="utf-8-sig", newline="") as fh:
                writer = csv.writer(fh, delimiter="\t")
                writer.writerow(["Name", "Start", "Duration", "Time Format", "Type", "Description"])
                writer.writerow(["正常", "0:00.000", "0:03.000", "decimal", "Cue", ""])
                writer.writerow(["正常", "0:06.000", "0:04.000", "decimal", "Cue", ""])
            records, anns, warnings = import_markers(
                audio, labels, root / "manifest.csv", root / "annotations.csv",
                pairing="stem", stable_markers=True)
            stable = [a for a in anns if a.label == "normal"]
            danger = [a for a in anns if a.label == "anomaly"]
            self.assertEqual(len(stable), 2)
            self.assertEqual(len(danger), 1)
            self.assertAlmostEqual(danger[0].start_s, 3.0, places=6)
            self.assertAlmostEqual(danger[0].end_s, 6.0, places=6)
            self.assertTrue(records[0].fully_reviewed)
            self.assertTrue(any("平稳区域" in w for w in warnings))


class StateSegmentationTests(unittest.TestCase):
    def _unstable_stretch(self, start=2.0, end=5.0):
        """A sustained irregular stretch: bursts every 0.2 s, no clean gap."""
        signal = tone(8.0)
        rng = np.random.default_rng(3)
        for t0 in np.arange(start, end, 0.2):
            i = int(t0 * SR)
            signal[i:i + 800] += rng.normal(0, 0.12, 800)
        return signal, start, end

    def test_state_mode_covers_the_whole_unstable_run(self):
        signal, start, end = self._unstable_stretch()
        detector = StabilityDetector(CFG, events=EventConfig(segment_mode="state", confirm_s=1.0))
        bundle = extract_features(signal, SR, CFG, "x")
        events, _ = detector.analyze(bundle)
        self.assertTrue(events)
        covered = sum(max(0.0, min(e["end_s"], end) - max(e["start_s"], start)) for e in events)
        self.assertGreater(covered, (end - start) * 0.7)
        self.assertLess(events[0]["start_s"], start + 0.8)

    def test_state_mode_is_quiet_on_a_stable_tone(self):
        detector = StabilityDetector(CFG, events=EventConfig(segment_mode="state"))
        events, _ = detector.analyze(extract_features(tone(8.0), SR, CFG, "x"))
        self.assertEqual(events, [])


class CalibrationTests(unittest.TestCase):
    def _stats(self, seed=0, n=20000):
        rng = np.random.default_rng(seed)
        base = np.abs(rng.normal(1.0, 0.25, n))
        return {"change_z": base, "exceed": np.abs(rng.normal(0.1, 0.05, n)),
                "slow_ratio": np.abs(rng.normal(0.8, 0.3, n)),
                "excess_ratio": np.abs(rng.normal(0.05, 0.03, n)),
                "z": np.abs(rng.normal(1.0, 0.35, n * 8)),
                "peak_change": base * 1.5,
                "fluct_ratio": np.abs(rng.normal(1.0, 0.3, n * 8)),
                "drift_ratio": np.abs(rng.normal(1.0, 0.6, n * 8)),
                "fluct_frac": rng.uniform(0.6, 1.0, n),
                "drift_frac": rng.uniform(0.6, 1.0, n)}

    def test_thresholds_sit_in_the_stable_tail(self):
        stats = self._stats()
        cal = calibrate(stats, band_z=3.0, alpha=1e-3, alpha_stable=1e-2)
        th = cal.thresholds
        self.assertGreater(th["trigger_change_z"], np.quantile(stats["change_z"], 0.99))
        self.assertGreater(th["trigger_exceed"], np.quantile(stats["exceed"], 0.99))
        self.assertLess(th["quiet_change_ratio"], 1.0)
        self.assertTrue(2.0 <= th["band_z"] <= 6.0)
        self.assertTrue(0.5 <= th["fluct_scale"] <= 10.0)

    def test_looser_false_alarm_target_lowers_thresholds(self):
        stats = self._stats(seed=1)
        strict = calibrate(stats, 3.0, alpha=1e-4, alpha_stable=1e-3).thresholds
        loose = calibrate(stats, 3.0, alpha=1e-2, alpha_stable=1e-1).thresholds
        self.assertLess(loose["trigger_change_z"], strict["trigger_change_z"])
        self.assertLess(loose["fluct_scale"], strict["fluct_scale"])
        self.assertLess(loose["stable_change_z"], strict["stable_change_z"])

    def test_band_fraction_threshold_bounds(self):
        stats = self._stats(seed=2)
        value = band_fraction_threshold(stats, alpha=1e-2)
        self.assertTrue(0.5 <= value <= 0.98)


if __name__ == "__main__":
    unittest.main()
