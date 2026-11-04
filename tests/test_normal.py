"""Normal-only model: leakage, edge artefacts, weak noise and CLI round trips."""
import contextlib
import csv
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tests.test_stable_labels import CFG, SR, tone
from wind_audio import cli
from wind_audio.audio import write_wav
from wind_audio.data import Annotation, AudioRecord
from wind_audio.detectors import load_detector
from wind_audio.features import extract_features
from wind_audio.markers import import_markers
from wind_audio.normal import NormalConfig, NormalModelDetector, normal_features, normal_windows
from wind_audio.normal_fit import fit_normal, normal_core, split_records


def model():
    cfg = NormalConfig()
    arrays = [normal_features(extract_features(tone(10, seed=i), SR, CFG, str(i)), cfg)[100:-100]
              for i in (20, 21)]
    return NormalModelDetector.fit(*arrays, feature_config=CFG, normal=cfg)


class NormalTests(unittest.TestCase):
    def test_steady_new_pitch_gain_and_silence_are_normal_including_edges(self):
        detector = model()
        for x in (tone(), tone(frequency=1400, amplitude=.25), np.zeros(8 * SR, np.float32)):
            events, traces = detector.analyze(extract_features(x, SR, CFG, "x"))
            self.assertEqual(events, [])
            self.assertTrue(traces["stable"].all())

    def test_weak_click_ignored_strong_short_event_kept(self):
        detector = model()
        for amplitude, should_detect in ((0.002, False), (0.25, True)):
            x = tone()
            x[3 * SR:3 * SR + 160] += np.random.default_rng(9).normal(0, amplitude, 160)
            events, _ = detector.analyze(extract_features(x, SR, CFG, "x"))
            self.assertEqual(bool(events), should_detect)

    def test_step_ends_after_new_stable_state(self):
        detector = model()
        x = tone()
        x[4 * SR:] = tone(4, frequency=1300, amplitude=.2)
        events, _ = detector.analyze(extract_features(x, SR, CFG, "x"))
        self.assertTrue(events)
        self.assertLess(events[-1]["end_s"], 6)

    def test_irregular_stretch_is_covered_and_inference_does_not_adapt_model(self):
        detector = model()
        before = detector.to_dict()
        x = tone()
        rng = np.random.default_rng(7)
        for t in np.arange(2, 5, .2):
            i = int(t * SR)
            x[i:i + 800] += rng.normal(0, .12, 800)
        events, _ = detector.analyze(extract_features(x, SR, CFG, "x"))
        coverage = sum(max(0, min(e["end_s"], 5) - max(e["start_s"], 2)) for e in events)
        self.assertGreater(coverage, 2.7)
        self.assertLess(events[-1]["end_s"], 6.5)
        self.assertEqual(detector.to_dict(), before)

    def test_repeated_bursts_merge_and_release_is_backdated(self):
        cfg = NormalConfig(padding=0)
        times = (np.arange(500) + .5) * .02
        score = np.zeros(500)
        score[100:125] = 2
        score[140:165] = 2
        events, _ = normal_windows(score, times, 10, 1, cfg, .02)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0]["end_s"], 3.3)
        score[:] = 0
        score[100:200:10] = 1.2
        events, trace = normal_windows(score, times, 10, 1, cfg, .02)
        self.assertEqual(events, [])
        self.assertTrue(trace["rejected_triggers"])

    def test_normal_core_excludes_all_boundary_context_and_unknown(self):
        times = (np.arange(500) + .5) * .02
        anns = [Annotation("n", "a", 2, 6, "normal")]
        mask = normal_core(times, anns, 10, .7)
        self.assertTrue(np.all(times[mask] >= 2.7))
        self.assertTrue(np.all(times[mask] <= 5.3))
        self.assertFalse(mask[times > 6].any())

    def test_calibration_not_training_scale_and_invalid_inputs(self):
        train = np.ones((200, 4))
        calibrated = NormalModelDetector.fit(train, train * 3)
        np.testing.assert_array_equal(calibrated.scales, [1, 1, 1, 1])
        self.assertEqual(calibrated.threshold, 3)
        for alpha in (0, 1, float("nan")):
            with self.assertRaises(ValueError):
                NormalModelDetector.fit(train, train, alpha=alpha)
        with self.assertRaises(ValueError):
            NormalModelDetector.fit(train[:1], train)
        with self.assertRaises(ValueError):
            NormalModelDetector([1, 2], 1)

    def test_roundtrip_and_feature_override_rejection(self):
        detector = model()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.json"
            path.write_text(json.dumps(detector.to_dict()))
            restored = load_detector(path)
            self.assertEqual(restored.to_dict(), detector.to_dict())
            args = cli.build_parser().parse_args(["run", "--input", "x", "--config", str(path), "--n-mels", "64"])
            with self.assertRaisesRegex(ValueError, "频谱配置不一致"):
                cli._detector_from_args(args)
            args = cli.build_parser().parse_args(["run", "--input", "x", "--config", str(path), "--trigger-exceed", ".2"])
            with self.assertRaisesRegex(ValueError, "不适用于"):
                cli._detector_from_args(args)

    def test_short_files_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "太短"):
            model().analyze(extract_features(tone(.2), SR, CFG, "x"))

    def test_split_is_ordered_disjoint_and_validated(self):
        records = [AudioRecord(str(i), f"a{i}.wav") for i in reversed(range(10))]
        split = split_records(records)
        self.assertEqual([r.audio_id for r in split["train"]], list(map(str, range(6))))
        self.assertEqual([r.audio_id for r in split["test"]], ["8", "9"])
        for a, b in ((0, .2), (.5, .5), (float("nan"), .2)):
            with self.assertRaises(ValueError):
                split_records(records, a, b)


class NormalIntegrationTests(unittest.TestCase):
    def corpus(self, root, count=10):
        audio, labels = root / "audio", root / "labels"
        audio.mkdir()
        labels.mkdir()
        for i in range(count):
            write_wav(audio / f"a{i:02}.wav", tone(4, seed=i), SR)
            with (labels / f"a{i:02}.csv").open("w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["Name", "Start", "Duration"])
                writer.writerow(["正常", 0, 4])
        return audio, labels

    def test_empty_normal_markers_are_fully_anomalous(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audio, labels = self.corpus(root, 1)
            (labels / "a00.csv").write_text("Name,Start,Duration\n")
            records, anns, _ = import_markers(audio, labels, root / "manifest.csv", root / "anns.csv",
                                               stable_markers=True)
            self.assertEqual([(a.start_s, a.end_s, a.label) for a in anns], [(0, 4, "anomaly")])
            self.assertTrue(records[0].fully_reviewed)

    def test_duplicate_recordings_and_wrong_label_semantics_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audio, labels = self.corpus(root)
            (labels / "a00.csv").write_text("Name,Start,Duration\n撞击,0,4\n")
            with self.assertRaisesRegex(ValueError, "仅接收"):
                fit_normal(audio, labels, root / "wrong", fully_reviewed=True, feature_config=CFG, verbose=False)
            (labels / "a00.csv").write_text("Name,Start,Duration\n正常,0,4\n")
            (audio / "a09.wav").write_bytes((audio / "a00.wav").read_bytes())
            with self.assertRaisesRegex(ValueError, "重复音频"):
                fit_normal(audio, labels, root / "duplicate", fully_reviewed=True, feature_config=CFG, verbose=False)

    def test_train_save_production_and_evaluation_use_same_model(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td)
            audio, labels = self.corpus(root)
            detector, report = fit_normal(audio, labels, root / "trained", fully_reviewed=True,
                                          feature_config=CFG, verbose=False)
            path = root / "trained/model.json"
            path.write_text(json.dumps(detector.to_dict()))
            self.assertEqual(set(report["metrics"]), {"train", "calibration", "test"})
            self.assertEqual([len(v) for v in detector.training["split_ids"].values()], [6, 2, 2])
            with patch("wind_audio.cli.read_annotations", side_effect=AssertionError("label leakage")):
                result = cli.main(["l1-detect", "--input", str(audio / "a09.wav"),
                                   "--config", str(path), "--out", str(root / "detect")])
            self.assertEqual(result, 0)
            payload = json.loads((root / "detect/events.json").read_text())
            self.assertTrue(payload["model"]["requires_training"])
            self.assertEqual(payload["diagnostics"]["anomaly_source"], "learned_normal_deviation")
            html = (root / "detect/report.html").read_text()
            self.assertNotIn("无需训练", html)
            self.assertIn("normal_deviation.png", html)
            for asset in re.findall(r'src="([^"]+)"', html):
                self.assertTrue((root / "detect" / asset).is_file(), asset)
            self.assertEqual(cli.main(["l1-test", "--audio-dir", str(audio), "--labels", str(labels),
                                       "--labels-are-stable", "--config", str(path),
                                       "--no-plots", "--out", str(root / "eval")]), 0)
            metrics = json.loads((root / "eval/metrics.json").read_text())
            row = next(r for r in metrics["per_file"] if r["audio_id"] == "a09")
            self.assertEqual(row["predicted_events"], [])
            with self.assertRaisesRegex(ValueError, "非空"):
                fit_normal(audio, labels, root / "trained", fully_reviewed=True)
            with self.assertRaisesRegex(ValueError, "确认"):
                fit_normal(audio, labels, root / "unconfirmed")


if __name__ == "__main__":
    unittest.main()
