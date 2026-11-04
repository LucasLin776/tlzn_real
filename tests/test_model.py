"""Window tests: stable states differ in level; only transitions are candidates."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from wind_audio import cli
from wind_audio.audio import wav_info, write_wav
from wind_audio.detectors import SpectralChangeDetector, load_detector
from wind_audio.features import FeatureConfig, extract_features
from wind_audio.l2.cli import cmd_pipeline, run_events
from types import SimpleNamespace

CFG = FeatureConfig(sr=8000, n_fft=512, hop=80, n_mels=32, fmax=3500)


def wave(duration=5, sr=8000, seed=1):
    t = np.arange(round(duration * sr)) / sr
    return (0.1 * np.sin(2 * np.pi * 500 * t) + 0.03 * np.sin(2 * np.pi * 1500 * t)
            + np.random.default_rng(seed).normal(0, 0.001, len(t))).astype(np.float32)


def detect(x, threshold=6, cfg=CFG):
    detector = SpectralChangeDetector(cfg, threshold)
    bundle = extract_features(x, cfg.sr, cfg, "a")
    return detector, bundle, detector.predict_events(bundle)


class WindowTests(unittest.TestCase):
    def test_stable_tones_noise_and_silence(self):
        for x in (wave(), np.zeros(40000, np.float32),
                  np.random.default_rng(2).normal(0, 0.03, 40000).astype(np.float32)):
            _, _, events = detect(x)
            self.assertEqual(events, [])

    def test_constant_gain_does_not_change_scores(self):
        x = wave()
        x[16000:16400] += 0.2 * np.random.default_rng(0).normal(size=400)
        det, a, _ = detect(x)
        _, b, _ = detect(x * 0.3)
        np.testing.assert_allclose(det.score(a), det.score(b), rtol=1e-3, atol=0.01)

    def test_gain_step_does_not_alarm_whole_new_state(self):
        x = wave()
        x[16000:] *= 3
        _, _, events = detect(x)
        self.assertEqual(len(events), 1)
        self.assertLessEqual(events[0]["start_s"], 2)
        self.assertGreaterEqual(events[0]["end_s"], 2)
        self.assertLess(events[0]["end_s"], 2.5)

    def test_impulse_has_a_window_not_just_peak(self):
        x = wave()
        x[16000:16400] += np.random.default_rng(0).normal(0, 0.2, 400)
        _, _, events = detect(x)
        self.assertEqual(len(events), 1)
        e = events[0]
        self.assertLessEqual(e["start_s"], 2)
        self.assertGreaterEqual(e["end_s"], 2.05)
        self.assertGreater(e["duration_s"], 0.05)
        self.assertLess(e["duration_s"], 0.7)
        self.assertLessEqual(e["start_s"], e["trigger_start_s"])
        self.assertGreaterEqual(e["end_s"], e["trigger_end_s"])

    def test_frequency_sweep_recovers_to_new_horizontal_lines(self):
        sr = CFG.sr
        t = np.arange(5 * sr) / sr
        frequency = np.where(t < 2, 500, np.where(t < 2.6, 500 + (t - 2) * 2200, 1820))
        x = (0.1 * np.sin(2 * np.pi * np.cumsum(frequency) / sr)).astype(np.float32)
        _, _, events = detect(x)
        self.assertEqual(len(events), 1)
        self.assertLessEqual(events[0]["start_s"], 2.1)
        self.assertGreaterEqual(events[0]["end_s"], 2.5)
        self.assertLess(events[0]["end_s"], 3)

    def test_two_events_separate_and_edges_clipped(self):
        x = wave()
        for a in (1, 3):
            x[int(a * 8000):int((a + 0.05) * 8000)] += np.random.default_rng(3).normal(0, .3, 400)
        _, _, events = detect(x)
        self.assertEqual(len(events), 2)
        self.assertLess(events[0]["end_s"], events[1]["start_s"])
        for index in (400, len(x) - 400):
            y = wave()
            y[index:index + 200] += .3 * np.random.default_rng(4).normal(size=200)
            _, b, events = detect(y)
            self.assertTrue(events)
            self.assertTrue(all(0 <= e["start_s"] < e["end_s"] <= b.duration_s for e in events))

    def test_padding_merges_overlapping_windows(self):
        x = wave()
        for i in (8000, 12000):
            x[i:i + 400] += np.random.default_rng(9).normal(0, .2, 400)
        det, bundle, _ = detect(x)
        events = det.predict_events(bundle, postprocess={"merge_gap": 0, "padding": .25, "quiet_s": .1})
        self.assertEqual(len(events), 1)

    def test_threshold_and_no_model_input(self):
        det, bundle, _ = detect(wave())
        self.assertEqual(det.predict_events(bundle, threshold=1000), [])
        self.assertEqual(load_detector().detector_type, "spectral_stability")
        for threshold in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                SpectralChangeDetector(threshold=threshold)
        with tempfile.TemporaryDirectory() as td, self.assertRaises(ValueError):
            load_detector(td)
        with self.assertRaises(ValueError):
            det.score(extract_features(wave(), 8000, FeatureConfig(sr=8000, fmax=3500), "bad"))

    def test_short_inputs_do_not_crash(self):
        for n in (1, 100, 800):
            _, _, events = detect(np.zeros(n, np.float32))
            self.assertEqual(events, [])

    def test_cli_outputs_windows_and_separate_context(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td)
            x = wave(sr=44100)
            x[88200:90405] += np.random.default_rng(9).normal(0, .2, 2205)
            path, out = root / "a.wav", root / "run"
            write_wav(path, x, 44100)
            self.assertEqual(cli.main(["run", "--input", str(path), "--out", str(out)]), 0)
            obj = json.loads((out / "events.json").read_text())
            self.assertTrue(obj["events"])
            e = obj["events"][0]
            self.assertAlmostEqual(wav_info(out / e["window"]["path"]).duration_s, e["duration_s"], places=4)
            self.assertLessEqual(e["clip"]["start_s"], e["start_s"])
            self.assertGreaterEqual(e["clip"]["end_s"], e["end_s"])
            self.assertTrue((out / "scores.csv").exists())
            self.assertTrue((out / "report.html").exists())
            self.assertEqual(load_detector(out / "l1_config.json").to_dict(), load_detector().to_dict())
            self.assertEqual(cli.main(["run", "--input", str(path), "--out", str(out)]), 1)

    def test_l2_receives_detection_not_listening_window(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = write_wav(root / "a.wav", wave(), 8000)
            event = {"start_s": 2, "end_s": 2.3, "clip": {"start_s": 1, "end_s": 3.3}}
            (root / "events.json").write_text(json.dumps({"audio": {"path": str(source)}, "events": [event]}))
            config = __import__("wind_audio.l2.encoder", fromlist=["EncoderConfig"]).EncoderConfig()
            with patch("wind_audio.l2.cli.Model.load") as mdl, patch("wind_audio.l2.cli.PannsEncoder") as enc:
                mdl.return_value.meta = {"version": "v", "encoder": config.to_dict()}
                mdl.return_value.predict.return_value = [{"status": "unknown"}]
                enc.return_value.config = config
                enc.return_value.encode.return_value = np.ones(2048)
                run_events(root / "events.json", "unused", "unused", root / "l2.json")
                self.assertEqual(enc.return_value.encode.call_args.args[1:], (2, 2.3))

    def test_evaluate_single_recording_without_training_split(self):
        from wind_audio.data import AudioRecord, Annotation, write_manifest, write_annotations
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td)
            x = wave(sr=44100)
            x[88200:90405] += np.random.default_rng(10).normal(0, .2, 2205)
            path = write_wav(root / "a.wav", x, 44100)
            write_manifest([AudioRecord("a", str(path), source_id="a", duration_s=5,
                                        fully_reviewed=True)], root / "manifest.csv")
            write_annotations([Annotation("e", "a", 2, 2.05, "anomaly")], root / "annotations.csv")
            self.assertEqual(cli.main(["evaluate", "--manifest", str(root / "manifest.csv"),
                                       "--annotations", str(root / "annotations.csv"),
                                       "--out", str(root / "eval")]), 0)
            result = json.loads((root / "eval/metrics.json").read_text())
            self.assertEqual(result["n_gt_events"], 1)
            self.assertEqual(result["event_recall"], 1)
            self.assertLessEqual(result["alarm_ratio"], 1)

    def test_pipeline_does_not_require_l1_model(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td)
            source = write_wav(root / "a.wav", wave(), 8000)
            args = SimpleNamespace(input=str(source), out=str(root / "out"), l1_config=None,
                                   l1_threshold=7, l2_model="l2", checkpoint="weights", device="cpu")
            with patch("wind_audio.cli.main", return_value=0) as main, patch("wind_audio.l2.cli.run_events"):
                cmd_pipeline(args)
                self.assertNotIn("--model", main.call_args.args[0])
                self.assertIn("7", main.call_args.args[0])
