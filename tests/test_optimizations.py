"""Cross-device splitting, event-level thresholds, joint scoring and segment rules."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.test_batches import make_batch
from tests.test_stable_labels import CFG, SR, tone
from wind_audio import cli
from wind_audio.data import Annotation, AudioRecord
from wind_audio.detectors import load_detector
from wind_audio.evaluate import _reviewed_regions, evaluate_file_events
from wind_audio.features import extract_features
from wind_audio.normal import (
    NormalConfig,
    NormalModelDetector,
    event_level_threshold,
    frame_score,
    normal_windows,
    short_events,
)
from wind_audio.normal_fit import fit_normal, split_records


def records(batch, count):
    return [AudioRecord(f"{batch}-{i:03d}", f"{batch}/a{i:02d}.wav", duration_s=4.0)
            for i in range(count)]


def score_ramp(count):
    """Strictly increasing scores: threshold -> covered fraction is predictable."""
    times = np.arange(count) * 0.02
    values = np.linspace(0.1, 5.0, count)
    return times, values


class HoldoutSplitTests(unittest.TestCase):
    def test_holdout_ids_form_the_check_set(self):
        items = records("B05", 6) + records("B06", 6)
        held = ["B06-000", "B06-001", "B06-002"]
        split = split_records(items, mode="shuffle", seed=0, holdout_ids=held)
        self.assertEqual([r.audio_id for r in split["test"]], held)
        self.assertEqual({r.audio_id for r in split["train"]} | {r.audio_id for r in split["calibration"]},
                         {r.audio_id for r in items} - set(held))
        self.assertEqual(len(split["calibration"]), 2)      # 20% of the 9 remaining, rounded
        with self.assertRaisesRegex(ValueError, "未知 audio_id"):
            split_records(items, holdout_ids=["nope"])

    def test_holdout_needs_enough_remaining_files(self):
        items = records("B05", 3)
        with self.assertRaisesRegex(ValueError, "至少两条"):
            split_records(items, holdout_ids=["B05-000", "B05-001", "B05-002"])


class CrossBatchTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        # 10 per batch: seed 0 leaves >= 2 train and >= 2 calibration files in
        # each batch, which the leave-one-batch-out folds need.
        make_batch(cls.root, "B05", 10)
        make_batch(cls.root, "B06", 10, seed_base=100)
        cls.out = cls.root / "run"
        with contextlib.redirect_stdout(io.StringIO()):
            cls.detector, cls.report = fit_normal(
                dataset_dir=cls.root, out_dir=cls.out, fully_reviewed=True, feature_config=CFG,
                normal=NormalConfig(), split_mode="shuffle", seed=0, target_fp=0.05,
                score_mode="joint", verbose=False)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_report_carries_the_cross_batch_check(self):
        cross = self.report["cross_batch"]
        self.assertEqual(set(cross), {"B05", "B06"})
        for batch in ("B05", "B06"):
            self.assertIn("normal_match", cross[batch])
            self.assertIn("anomaly_recall", cross[batch])
            self.assertIn("threshold", cross[batch])
        self.assertIn("跨设备", self.report["cross_batch_note"])
        self.assertEqual(self.detector.training["score_mode"], "joint")

    def test_threshold_is_calibrated_at_event_level(self):
        self.assertEqual(self.detector.training["threshold_policy"], "event_level")
        info = self.report["threshold_calibration"]
        self.assertEqual(info["target_fp"], 0.05)
        self.assertGreaterEqual(self.detector.threshold, 1.0)
        if info["achieved_calibration_fp"] is not None:
            self.assertLessEqual(info["achieved_calibration_fp"], 0.05 + 1e-9)
        for group in ("train", "calibration", "test"):
            summary = self.report["metrics"][group]
            self.assertIn("normal_match", summary)
            self.assertIn("model_normal_precision", summary)
            self.assertIn("short_channel", summary)
            self.assertLessEqual(summary["boundary_tolerant"]["tolerance_s"], 0.2)

    def test_holdout_batch_moves_a_whole_batch_to_the_check_set(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            detector, report = fit_normal(
                dataset_dir=self.root, out_dir=Path(td) / "held", fully_reviewed=True,
                feature_config=CFG, normal=NormalConfig(), split_mode="shuffle", seed=0,
                holdout_batches=["B06"], target_fp=None, verbose=False)
            self.assertEqual(detector.training["holdout_batches"], ["B06"])
            self.assertTrue(all(i.startswith("B06") for i in detector.training["split_ids"]["test"]))
            self.assertTrue(all(not i.startswith("B06") for i in detector.training["split_ids"]["train"]))
            self.assertIn("留出批次", report["split_policy"])
            self.assertEqual(report["cross_batch"]["B05"].get("skipped"), "训练或校核剩余文件不足两条")
            with self.assertRaisesRegex(ValueError, "未在数据中找到"):
                fit_normal(dataset_dir=self.root, out_dir=Path(td) / "bad", fully_reviewed=True,
                           feature_config=CFG, holdout_batches=["B99"], verbose=False)


class EventThresholdTests(unittest.TestCase):
    def detector(self):
        times, values = score_ramp(600)
        return NormalModelDetector([1, 1, 1, 1], 1.0, CFG, NormalConfig()), times, values

    def test_budget_is_met_and_a_tighter_budget_never_lowers_the_threshold(self):
        det, times, values = self.detector()
        entries = [{"times": times, "values": np.column_stack([values] * 4),
                    "duration": times[-1] + 0.02, "normal": [(0.0, times[-1] + 0.02)]}]
        loose, _ = event_level_threshold(det, entries, 0.25, floor=1.0)
        tight, info = event_level_threshold(det, entries, 0.05, floor=1.0)
        self.assertGreaterEqual(tight, loose)
        self.assertGreater(tight, 1.0)
        self.assertLessEqual(info["achieved_calibration_fp"], 0.05 + 0.03)
        self.assertGreater(info["achieved_calibration_fp"], 0.0)

    def test_floor_is_respected(self):
        det, times, values = self.detector()
        entries = [{"times": times, "values": np.column_stack([values] * 4),
                    "duration": times[-1] + 0.02, "normal": [(0.0, times[-1] + 0.02)]}]
        chosen, _ = event_level_threshold(det, entries, 0.5, floor=3.0)
        self.assertGreaterEqual(chosen, 3.0 - 1e-12)
        with self.assertRaisesRegex(ValueError, "目标误报率"):
            event_level_threshold(det, entries, 1.5)


class JointScoreTests(unittest.TestCase):
    def corpus(self):
        rng = np.random.default_rng(4)
        common = rng.normal(1.0, 0.35, 4000)                 # shared activity factor
        train = np.column_stack([common + rng.normal(0, 0.05, 4000) for _ in range(4)])
        return np.clip(train, 0.01, None)

    def test_joint_score_discounts_the_shared_activity_axis(self):
        train = self.corpus()
        det = NormalModelDetector.fit(train, train[:800], score_mode="joint")
        self.assertEqual(det.score_config.mode, "joint")
        self.assertGreater(det.score_config.normalizer, 0)
        both = frame_score([[3.0, 3.0, 3.0, 3.0]], det.scales, det.score_config)[0]
        single = frame_score([[3.0, 4.5, 4.5, 4.5]], det.scales, det.score_config)[0]
        self.assertLess(both, single)       # a common shift is discounted, not amplified
        legacy = frame_score([[3.0, 3.0, 3.0, 3.0]], det.scales,
                             NormalModelDetector.fit(train, train[:800], score_mode="max").score_config)[0]
        self.assertGreater(both, legacy)    # rescaled so the median matches the legacy score

    def test_score_mode_round_trips_and_legacy_json_still_loads(self):
        train = self.corpus()
        joint = NormalModelDetector.fit(train, train[:800], score_mode="joint")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.json"
            path.write_text(json.dumps(joint.to_dict()))
            self.assertEqual(load_detector(path).to_dict(), joint.to_dict())
            legacy = json.loads(path.read_text())
            legacy.pop("score")
            path.write_text(json.dumps(legacy))
            restored = load_detector(path)
            self.assertEqual(restored.score_config.mode, "max")

    def test_cli_rebuild_keeps_the_joint_score(self):
        train = self.corpus()
        joint = NormalModelDetector.fit(train, train[:800], score_mode="joint", feature_config=CFG)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.json"
            path.write_text(json.dumps(joint.to_dict()))
            args = cli.build_parser().parse_args(["run", "--input", "x", "--config", str(path)])
            rebuilt = cli._detector_from_args(args)
            self.assertEqual(rebuilt.score_config.mode, "joint")
            self.assertEqual(rebuilt.to_dict(), joint.to_dict())

    def test_constant_training_data_falls_back_to_max(self):
        flat = np.ones((300, 4))
        det = NormalModelDetector.fit(flat, flat * 3, score_mode="joint")
        self.assertEqual(det.score_config.mode, "max")
        self.assertEqual(det.threshold, 3)


class SegmentRuleTests(unittest.TestCase):
    def build(self, spans, total=10.0):
        step = 0.02
        n = int(total / step)
        score = np.full(n, 0.1)
        for start, end, level in spans:
            score[int(start / step):int(end / step)] = level
        return score, np.arange(n) * step

    def test_short_weak_fragments_are_absorbed_but_strong_ones_survive(self):
        cfg = NormalConfig(min_anomaly_s=0.30)
        score, times = self.build([(2.0, 2.2, 2.0)])
        events, traces = normal_windows(score, times, 10.0, 1.0, cfg, 0.02)
        self.assertEqual(events, [])
        self.assertTrue(any(r["reason"] == "shorter_than_min_anomaly" for r in traces["rejected_triggers"]))
        score, times = self.build([(2.0, 2.2, 3.0)])
        events, _ = normal_windows(score, times, 10.0, 1.0, cfg, 0.02)
        self.assertEqual(len(events), 1)          # 3.0 >= strong_ratio * threshold

    def test_short_normal_gaps_are_bridged(self):
        cfg = NormalConfig(min_normal_s=1.0)
        score, times = self.build([(2.0, 2.6, 2.0), (3.4, 4.0, 2.0)])
        events, _ = normal_windows(score, times, 10.0, 1.0, cfg, 0.02)
        self.assertEqual(len(events), 1)          # 0.8 s normal gap < min_normal_s
        self.assertLess(events[0]["start_s"], 2.0 + 1e-6)
        self.assertGreater(events[0]["end_s"], 4.0 - 1e-6)
        strict = NormalConfig(min_normal_s=0.0)
        events, _ = normal_windows(score, times, 10.0, 1.0, strict, 0.02)
        self.assertEqual(len(events), 2)

    def test_boundary_tolerance_stops_charging_off_by_a_bit_edges(self):
        record = AudioRecord("a", "a.wav", duration_s=10.0, fully_reviewed=True)
        anns = [Annotation("a1", "a", 3.0, 5.0, "anomaly")]
        strict = _reviewed_regions(record, anns)
        tolerant = _reviewed_regions(record, anns, 0.2)
        self.assertEqual(sum(r["end_s"] - r["start_s"] for r in strict if r["label"] == "normal"), 8.0)
        self.assertAlmostEqual(sum(r["end_s"] - r["start_s"] for r in tolerant if r["label"] == "normal"), 7.6)
        self.assertTrue(any(r["label"] == "uncertain" and abs(r["start_s"] - 2.8) < 1e-9 for r in tolerant))
        pred = [{"start_s": 2.85, "end_s": 5.0}]          # starts 0.15 s early
        plain = evaluate_file_events(record, anns, pred)
        loose = evaluate_file_events(record, anns, pred, boundary_tolerance=0.2)
        self.assertGreater(plain["time_metrics"]["fp"], 0.1)
        self.assertLess(loose["time_metrics"]["fp"], 1e-9)
        self.assertGreater(loose["unknown_s"], plain["unknown_s"])


class ShortChannelTests(unittest.TestCase):
    def test_impulses_are_reported_but_never_change_events_or_regions(self):
        step, n = 0.02, 500
        score = np.full(n, 0.1)
        score[200:205] = 2.2                        # short, but below the strong-event bypass
        times = np.arange(n) * step
        cfg = NormalConfig()
        hints = short_events(score, times, n * step, 1.0, cfg, step)
        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0]["channel"], "short_impulse")
        self.assertAlmostEqual(hints[0]["peak_score"], 2.2)
        events, _ = normal_windows(score, times, n * step, 1.0, cfg, step)
        self.assertEqual(events, [])                # still below the main detector
        covered = short_events(score, times, n * step, 1.0, cfg, step,
                               covered=[{"start_s": 3.5, "end_s": 5.0}])
        self.assertEqual(covered, [])               # suppressed inside a main event
        self.assertEqual(short_events(score, times, n * step, 1.0,
                                      NormalConfig(short_channel=False), step), [])

    def test_analyze_exposes_the_channel_without_touching_normal_regions(self):
        train = np.clip(np.random.default_rng(1).normal(1.5, 0.3, (3000, 4)), 0.05, None)
        det = NormalModelDetector.fit(train, train[:500], feature_config=CFG)
        signal = tone(8.0, seed=2)
        signal[4 * SR:4 * SR + int(0.02 * SR)] *= 6.0     # one-frame impulse
        events, traces = det.analyze(extract_features(signal, SR, CFG, "x"))
        self.assertIn("short_events", traces)
        self.assertIn("normal_regions", traces)
        for hint in traces["short_events"]:
            self.assertEqual(hint["channel"], "short_impulse")
            self.assertFalse(any(hint["start_s"] < e["end_s"] and hint["end_s"] > e["start_s"]
                                 for e in events))   # main events are the excluded set


class CliOptionTests(unittest.TestCase):
    def test_parser_defaults_and_overrides(self):
        args = cli.build_parser().parse_args(
            ["l1-train-normal", "--dataset-dir", "d", "--out", "o", "--fully-reviewed-normal"])
        self.assertEqual(args.target_fp, 0.02)
        self.assertEqual(args.score_mode, "max")
        self.assertEqual(args.holdout_batch, "")
        self.assertFalse(args.no_short_channel)
        args = cli.build_parser().parse_args(
            ["l1-train-normal", "--dataset-dir", "d", "--out", "o", "--fully-reviewed-normal",
             "--target-fp", "0", "--score-mode", "max", "--holdout-batch", "B06,B07",
             "--no-short-channel", "--min-anomaly-s", "0.5"])
        self.assertEqual(args.target_fp, 0.0)
        self.assertEqual(args.score_mode, "max")
        self.assertEqual(args.holdout_batch, "B06,B07")
        self.assertTrue(args.no_short_channel)
        self.assertEqual(args.min_anomaly_s, 0.5)

    def test_frame_quantile_policy_is_still_selectable(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td)
            make_batch(root, "B05", 6)
            make_batch(root, "B06", 6, seed_base=100)
            self.assertEqual(cli.main(["l1-train-normal", "--dataset-dir", str(root),
                                       "--out", str(root / "legacy"), "--fully-reviewed-normal",
                                       "--target-fp", "0", "--score-mode", "max"]), 0)
            model = json.loads((root / "legacy/model.json").read_text())
            self.assertEqual(model["training"]["threshold_policy"], "frame_quantile")
            self.assertEqual(model["score"]["mode"], "max")
            report = json.loads((root / "legacy/training_report.json").read_text())
            self.assertIn("cross_batch", report)


if __name__ == "__main__":
    unittest.main()
