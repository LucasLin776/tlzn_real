"""Business-definition tests for the v3 stability detector.

Each test states the rule it protects, so a future change that breaks the
definition fails here instead of silently fitting the old behaviour.
"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wind_audio import cli
from wind_audio.detectors import SpectralChangeDetector, StabilityDetector, load_detector
from wind_audio.features import FeatureConfig, extract_features
from wind_audio.stability import StabilityConfig, analyze, audio_quality
from wind_audio.state import EventConfig

CFG = FeatureConfig(sr=8000, n_fft=512, hop=80, n_mels=32, fmax=3500)
SR = CFG.sr


def tone(duration=6.0, frequency=500.0, amplitude=0.1, noise=0.001, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * frequency * t)
            + rng.normal(0, noise, len(t))).astype(np.float32)


def broadband(duration=6.0, amplitude=0.02, seed=1):
    return np.random.default_rng(seed).normal(0, amplitude, int(duration * SR)).astype(np.float32)


def detect(x, **event):
    detector = StabilityDetector(CFG, events=EventConfig(**event) if event else None)
    bundle = extract_features(np.asarray(x, np.float32), SR, CFG, "x")
    return detector.analyze(bundle)


class StabilitySignalTests(unittest.TestCase):
    def test_silence_stable_tone_broadband_and_small_noise_are_normal(self):
        """Stable sound of any level/shape is normal; small noise is tolerated."""
        for name, signal in (
                ("silence", np.zeros(int(6 * SR), np.float32)),
                ("tone", tone()),
                ("quiet tone", tone(amplitude=0.01)),
                ("broadband", broadband()),
                ("tone with allowed jitter", tone(noise=0.002, seed=3)),
        ):
            with self.subTest(name):
                events, traces = detect(signal)
                self.assertEqual(events, [], f"{name} 不应产生持续报警")
                self.assertGreater(float(np.mean(traces["stable"])), 0.5)

    def test_impulse_and_step_trigger(self):
        """A change clearly above the local tolerance must trigger."""
        impulse = tone()
        impulse[int(2 * SR):int(2 * SR) + 400] += np.random.default_rng(2).normal(0, 0.2, 400)
        events, _ = detect(impulse)
        self.assertEqual(len(events), 1)
        self.assertLessEqual(events[0]["start_s"], 2.05)
        self.assertGreaterEqual(events[0]["end_s"], 2.05)

        step = tone()
        step[int(2 * SR):] *= 3
        events, _ = detect(step)
        self.assertEqual(len(events), 1)
        self.assertLessEqual(events[0]["start_s"], 2.05)

    def test_new_spectral_state_ends_promptly(self):
        """After a shape change re-stabilises, the event closes without matching the old spectra."""
        t = np.arange(int(6 * SR)) / SR
        frequency = np.where(t < 2, 500.0, 1800.0)
        signal = (0.1 * np.sin(2 * np.pi * np.cumsum(frequency) / SR)
                  + np.random.default_rng(4).normal(0, 0.001, len(t))).astype(np.float32)
        events, traces = detect(signal)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["boundary_reason"], "stable_confirmed")
        self.assertLess(events[0]["end_s"], 3.4, "新稳态应在确认后及时结束")
        self.assertTrue(np.all(traces["stable"][int(4 * SR / SR * 100):]))  # 后段恢复稳定

    def test_level_up_and_down_end_without_returning_to_old_level(self):
        """A gain step is an event even though the new level is never left again."""
        for factor in (3.0, 0.25):
            with self.subTest(factor=factor):
                signal = tone(8.0)
                signal[int(2 * SR):] *= factor
                events, _ = detect(signal)
                self.assertEqual(len(events), 1)
                self.assertLess(events[0]["end_s"], 3.6)
                self.assertGreaterEqual(events[0]["end_s"], 2.0)

    def test_short_bursts_merge_but_stable_gap_separates(self):
        """Continuous short events may merge; a recovered stable interval must split them."""
        merged = tone(8.0)
        for start in (1.0, 1.3, 1.6):
            i = int(start * SR)
            merged[i:i + 240] = merged[i:i + 240] + np.random.default_rng(5).normal(0, 0.2, 240)
        events, _ = detect(merged)
        self.assertEqual(len(events), 1)
        self.assertLessEqual(events[0]["start_s"], 1.05)
        self.assertGreaterEqual(events[0]["end_s"], 1.63)   # 覆盖第三个短事件

        split = tone(8.0)
        for start in (1.0, 4.0):
            i = int(start * SR)
            split[i:i + 240] = split[i:i + 240] + np.random.default_rng(6).normal(0, 0.2, 240)
        events, _ = detect(split)
        self.assertEqual(len(events), 2)
        self.assertLess(events[0]["end_s"], events[1]["start_s"])

    def test_repeated_short_events_are_not_averaged_into_stability(self):
        """A long window that looks flat on average must not hide short events inside it."""
        signal = tone(6.0)
        rng = np.random.default_rng(7)
        for start in np.arange(1.0, 4.0, 0.35):
            i = int(start * SR)
            signal[i:i + 160] += rng.normal(0, 0.2, 160)
        events, _ = detect(signal)
        self.assertGreaterEqual(len(events), 1)
        covered = sum(e["end_s"] - e["start_s"] for e in events)
        self.assertGreater(covered, 1.0)
        # 事件区域不应被判成稳定
        traces_stable = detect(signal)[1]["stable"]
        span = slice(int(1.0 * SR / SR * 100 / 2), int(3.9 * SR / SR * 100 / 2))
        self.assertLess(float(np.mean(traces_stable[span])), 0.5)

    def test_near_threshold_jitter_does_not_flap(self):
        """Small wobble around the tolerance must not switch the alarm on and off."""
        t = np.arange(int(8 * SR)) / SR
        wobble = 1.0 + 0.02 * np.sin(2 * np.pi * 0.7 * t)  # ~0.17 dB, far below tolerance
        signal = (0.1 * wobble * np.sin(2 * np.pi * 500 * t)
                  + np.random.default_rng(8).normal(0, 0.001, len(t))).astype(np.float32)
        events, _ = detect(signal)
        self.assertEqual(events, [])

    def test_weak_and_strong_events_both_detected(self):
        """A loud event elsewhere in the file must not delete a weaker one."""
        signal = tone(8.0)
        i = int(1.5 * SR)
        signal[i:i + 400] += np.random.default_rng(9).normal(0, 0.06, 400)
        i = int(5.0 * SR)
        signal[i:i + 400] += np.random.default_rng(10).normal(0, 0.5, 400)
        events, _ = detect(signal)
        spans = [(e["start_s"], e["end_s"]) for e in events]
        self.assertEqual(len(events), 2, spans)
        self.assertLess(spans[0][0], 1.7)
        self.assertGreater(spans[1][0], 4.8)

    def test_gain_scaling_does_not_change_windows(self):
        """Pure gain without clipping must not reshape the result."""
        signal = tone()
        impulse = signal.copy()
        impulse[int(2 * SR):int(2 * SR) + 400] += np.random.default_rng(11).normal(0, 0.2, 400)
        a, _ = detect(impulse)
        b, _ = detect(impulse * 0.2)
        self.assertEqual(len(a), len(b))
        for left, right in zip(a, b):
            self.assertAlmostEqual(left["start_s"], right["start_s"], delta=0.06)
            self.assertAlmostEqual(left["end_s"], right["end_s"], delta=0.06)


class StabilityEdgeTests(unittest.TestCase):
    def test_short_files_report_insufficient_context(self):
        for duration in (0.02, 0.2, 0.6):
            with self.subTest(duration):
                events, traces = detect(np.zeros(int(duration * SR), np.float32))
                self.assertEqual(events, [])
                self.assertTrue(traces["context_insufficient"])

    def test_event_at_file_end_is_censored_not_closed(self):
        signal = tone(6.0)
        signal[int(4 * SR):] += np.random.default_rng(12).normal(0, 0.2, len(signal) - int(4 * SR))
        events, _ = detect(signal)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["right_censored"])
        self.assertEqual(events[0]["boundary_reason"], "end_of_file")
        self.assertAlmostEqual(events[0]["end_s"], 6.0, places=3)

    def test_schema_dispatch_and_unknown_keys(self):
        root = Path(__file__).parents[1]
        self.assertIsInstance(load_detector(root / "config/l1.json"), SpectralChangeDetector)
        self.assertIsInstance(load_detector(root / "config/l1_v3.json"), StabilityDetector)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cfg.json"
            path.write_text(json.dumps({"schema_version": 9, "detector": "x", "version": "y"}))
            with self.assertRaises(ValueError):
                load_detector(path)
            path.write_text(json.dumps({"schema_version": 2, "detector": "spectral_stability",
                                        "version": "spectral_stability_v3",
                                        "feature_config": {}, "mystery": 1}))
            with self.assertRaises(ValueError):
                load_detector(path)
            path.write_text(json.dumps({"schema_version": 2, "detector": "spectral_stability",
                                        "version": "spectral_stability_v3",
                                        "stability": {"typo": 1}}))
            with self.assertRaises(ValueError):
                load_detector(path)

    def test_invalid_parameters_rejected(self):
        for kwargs in ({"confirm_s": 0}, {"trigger_exceed": 0}, {"confirm_ratio": 0.1},
                       {"trigger_change_z": float("nan")}):
            with self.subTest(kwargs), self.assertRaises(ValueError):
                EventConfig(**kwargs)
        for kwargs in ({"stable_s": 0}, {"fluct_scale": 0}, {"weight_exp": 2},
                       {"sigma_floor_db": 10}):
            with self.subTest(kwargs), self.assertRaises(ValueError):
                StabilityConfig(**kwargs)

    def test_detection_and_test_entry_share_the_algorithm(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td)
            source = root / "audio" / "event.wav"
            source.parent.mkdir()
            signal = tone(4.0)
            i = int(1.5 * SR)
            signal[i:i + 400] += np.random.default_rng(13).normal(0, 0.2, 400)
            from wind_audio.audio import write_wav
            write_wav(source, signal, SR)
            detect_out, test_out = root / "detect", root / "test"
            labels = root / "labels.csv"
            labels.write_text("file,label\nevent.wav,anomaly\n")
            self.assertEqual(cli.main(["l1-detect", "--input", str(source), "--out", str(detect_out)]), 0)
            self.assertEqual(cli.main(["l1-test", "--audio-dir", str(source.parent),
                                       "--labels", str(labels), "--out", str(test_out)]), 0)
            a = json.loads((detect_out / "events.json").read_text())
            b = json.loads((test_out / "metrics.json").read_text())
            self.assertEqual([(e["start_s"], e["end_s"]) for e in a["events"]],
                             [(e["start_s"], e["end_s"]) for e in b["per_file"][0]["predicted_events"]])
            self.assertEqual(b["detector_config"]["version"], "spectral_stability_v3")

    def test_quality_diagnostics_never_modify_audio(self):
        signal = tone(2.0) * 12.0
        signal = np.clip(signal, -1, 1).astype(np.float32)
        quality = audio_quality(signal, SR)
        self.assertGreater(quality["clip_ratio"], 0.01)
        self.assertLess(quality["crest_db"], 6.0)
        events, traces = detect(signal)
        self.assertIn("exceed", traces)
        self.assertNotIn("quality", traces)  # computed by the caller, never applied to audio


if __name__ == "__main__":
    unittest.main()
