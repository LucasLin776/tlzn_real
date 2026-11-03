"""Regression cases for repository-hardening changes; no production data required."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from wind_audio import cli
from wind_audio.backends import select_backend
from wind_audio.calibrate import _candidate_thresholds, calibrate_detector
from wind_audio.data import Annotation, AudioRecord, read_annotations
from wind_audio.events import extract_events
from wind_audio.features import FeatureConfig


class AnnotationInputTests(unittest.TestCase):
    def test_missing_annotations_are_not_empty_annotations(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                read_annotations(Path(td) / "missing.csv")

    def test_explicit_empty_annotation_file_is_allowed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "annotations.csv"
            path.write_text("annotation_id,audio_id,start_s,end_s,label,precision\n", encoding="utf-8")
            self.assertEqual(read_annotations(path), [])


class EventGapTests(unittest.TestCase):
    def test_gap_uses_interval_edges_including_exact_boundary(self):
        times = (np.arange(5) + 0.5) * 0.1
        scores = np.array([1, 0, 0, 0, 1])
        merged = extract_events(scores, times, 0.5, min_duration=0.1,
                                merge_gap=0.3, duration_s=0.5, step_s=0.1)
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0]["duration_s"], 0.5)
        separate = extract_events(scores, times, 0.5, min_duration=0.1,
                                  merge_gap=0.299, duration_s=0.5, step_s=0.1)
        self.assertEqual(len(separate), 2)


class CalibrationTests(unittest.TestCase):
    def test_candidate_count_is_bounded_for_long_recordings(self):
        for kind in ("baseline_z", "probability_uncalibrated"):
            scores = np.linspace(0.001, 0.999, 36000, dtype=np.float32)
            values = _candidate_thresholds(scores, kind)
            self.assertLessEqual(len(values), 256)
            self.assertTrue(all(np.isfinite(x) and x > 0 for x in values))

    def test_candidate_midpoints_and_extrema_are_preserved(self):
        scores = np.array([0.0005, 0.9995], dtype=np.float32)
        values = _candidate_thresholds(scores, "probability_uncalibrated")
        self.assertTrue(any(abs(v - 0.5) < 1e-6 for v in values))
        self.assertGreater(max(values), float(scores.max()))

    def test_nonfinite_scores_are_filtered(self):
        self.assertEqual(_candidate_thresholds(np.array([np.nan, np.inf]), "baseline_z"), [])
        values = _candidate_thresholds(np.array([0.2, np.nan, 0.8]), "probability_uncalibrated")
        self.assertTrue(values)
        self.assertTrue(all(np.isfinite(v) for v in values))

    def test_invalid_recall_is_rejected_before_io(self):
        for recall in (0, -1, 1.01, np.nan):
            with self.subTest(recall=recall), self.assertRaises(ValueError):
                calibrate_detector(None, [], [], [], "unused", target_recall=recall)

    def test_unreachable_target_prefers_recall_to_silence(self):
        cfg = FeatureConfig()
        times = (np.arange(100) + 0.5) * 0.1
        score = np.full(100, 0.01)
        score[((times >= 1) & (times < 1.5)) | ((times >= 4) & (times < 4.5))] = 0.8
        record = AudioRecord("a", "unused.wav", duration_s=10, fully_reviewed=True)
        annotations = [Annotation("g1", "a", 1, 1.5, "anomaly"),
                       Annotation("g2", "a", 7, 7.5, "anomaly")]
        bundle = SimpleNamespace(times=times, duration_s=10, config=cfg)
        detector = SimpleNamespace(feature_config=cfg, score_type="probability_uncalibrated",
                                   score=lambda _: score)
        with patch("wind_audio.calibrate.load_wav", return_value=(np.zeros(1), 44100)), \
                patch("wind_audio.calibrate.extract_features", return_value=bundle):
            result = calibrate_detector(detector, [record], annotations, ["a"], "unused.csv",
                                        target_recall=0.95, min_durations=(0.1,),
                                        merge_gaps=(0.1,), low_ratios=(0.6,))
        self.assertFalse(result["target_reached"])
        self.assertEqual(result["validation"]["event_recall"], 0.5)
        self.assertGreater(result["validation"]["fa_per_hour"], 0)


class CommandPolicyTests(unittest.TestCase):
    def test_parser_rejects_test_set_calibration(self):
        argv = ["calibrate", "--model", "unused", "--manifest", "unused",
                "--annotations", "unused", "--split", "test"]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.build_parser().parse_args(argv)

    def test_direct_calibration_call_rejects_test_set(self):
        with self.assertRaises(ValueError):
            cli.cmd_calibrate(SimpleNamespace(split="test"))

    def test_partial_and_total_inference_failure_return_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ("a.wav", "b.wav"):
                (root / name).touch()
            detector = SimpleNamespace(feature_config=FeatureConfig(), threshold=0.5,
                                       postprocess={}, calibrated=False)
            args = cli.build_parser().parse_args(["run", "--input", str(root), "--model", "unused"])
            ok = {"events": 0, "report": "unused"}
            for outcomes, expected in (([ok, ok], 0), ([ok, ValueError("bad WAV")], 1),
                                       ([ValueError("bad WAV"), ValueError("bad WAV")], 1)):
                with self.subTest(expected=expected), \
                        patch("wind_audio.cli.load_detector", return_value=detector), \
                        patch("wind_audio.cli._run_one", side_effect=outcomes), \
                        contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(cli.cmd_run(args), expected)


if __name__ == "__main__":
    unittest.main()
