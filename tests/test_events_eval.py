import unittest

import numpy as np

from wind_audio.events import extract_events
from wind_audio.evaluate import evaluate_file_events, match_events

from tests.helpers import ann, record


class MatchingTests(unittest.TestCase):
    def test_one_to_one(self):
        gt = [{"start_s": 0.0, "end_s": 1.0}, {"start_s": 2.0, "end_s": 3.0}]
        pred = [{"start_s": 0.0, "end_s": 1.0}, {"start_s": 2.0, "end_s": 3.0}]
        m = match_events(pred, gt, 0.2)
        self.assertEqual(len(m["matches"]), 2)
        self.assertFalse(m["unmatched_pred"])

    def test_huge_prediction_cannot_hit_every_event(self):
        gt = [{"start_s": 0.0, "end_s": 1.0}, {"start_s": 2.0, "end_s": 3.0}]
        pred = [{"start_s": 0.0, "end_s": 10.0}]
        m = match_events(pred, gt, 0.2)
        self.assertEqual(len(m["matches"]), 0)
        self.assertEqual(len(m["unmatched_gt"]), 2)

    def test_duplicate_predictions_are_not_double_counted(self):
        gt = [{"start_s": 0.0, "end_s": 1.0}]
        pred = [{"start_s": 0.0, "end_s": 1.0}, {"start_s": 0.05, "end_s": 1.05}]
        m = match_events(pred, gt, 0.2)
        self.assertEqual(len(m["matches"]), 1)
        self.assertEqual(len(m["unmatched_pred"]), 1)


class EventExtractionTests(unittest.TestCase):
    def test_edges_and_min_duration(self):
        times = (np.arange(10) + 0.5) * 0.1
        score = np.zeros(10)
        score[0:2] = 5.0
        score[8:10] = 5.0
        ev = extract_events(score, times, 1.0, min_duration=0.15, duration_s=1.0, step_s=0.1)
        self.assertEqual(len(ev), 2)
        self.assertAlmostEqual(ev[0]["start_s"], 0.0, places=6)
        self.assertAlmostEqual(ev[1]["end_s"], 1.0, places=6)
        self.assertTrue(ev[0]["at_edge"] and ev[1]["at_edge"])

        tiny = np.zeros(10)
        tiny[4] = 5.0
        ev2 = extract_events(tiny, times, 1.0, min_duration=0.2, duration_s=1.0, step_s=0.1)
        self.assertEqual(ev2, [])

    def test_merge_gap(self):
        times = (np.arange(20) + 0.5) * 0.1
        score = np.zeros(20)
        score[2:4] = 5.0
        score[5:7] = 5.0
        ev = extract_events(score, times, 1.0, min_duration=0.1, merge_gap=0.2, duration_s=2.0, step_s=0.1)
        self.assertEqual(len(ev), 1)
        ev2 = extract_events(score, times, 1.0, min_duration=0.1, merge_gap=0.05, duration_s=2.0, step_s=0.1)
        self.assertEqual(len(ev2), 2)


class FileEvaluationTests(unittest.TestCase):
    def test_short_ground_truth_uses_continuous_boundaries(self):
        rec = record("a", "a.wav", 1, fully_reviewed=False)
        anns = [ann("x", "a", 0.011, 0.029, "anomaly")]
        pred = [{"start_s": 0.01, "end_s": 0.03}]
        result = evaluate_file_events(rec, anns, pred)
        self.assertEqual(result["n_matched"], 1)
        self.assertAlmostEqual(result["anomaly_s"], .018)
        self.assertAlmostEqual(result["alarm_s"], .018)

    def test_matched_windows_still_count_normal_overlap(self):
        rec = record("a", "a.wav", 10, fully_reviewed=True)
        anns = [ann("x", "a", 2, 3, "anomaly")]
        result = evaluate_file_events(rec, anns, [{"start_s": 1, "end_s": 4}])
        self.assertEqual(result["n_matched"], 1)
        self.assertAlmostEqual(result["fp_normal_s"], 2)

    def test_unknown_time_is_not_alarm_time(self):
        rec = record("a", "a.wav", 10, fully_reviewed=False)
        anns = [ann("x", "a", 2, 3, "anomaly")]
        result = evaluate_file_events(rec, anns, [{"start_s": 0, "end_s": 10}])
        self.assertAlmostEqual(result["alarm_s"], 1)
        self.assertEqual(result["n_matched"], 1)

    def test_unknown_is_excluded_and_normal_hours_are_denominator(self):
        rec = record("a", "a.wav", 10.0, fully_reviewed=False)
        anns = [ann("x", "a", 2.0, 3.0, "anomaly"), ann("n", "a", 4.0, 9.0, "normal")]
        pred = [
            {"start_s": 2.05, "end_s": 2.95},
            {"start_s": 4.0, "end_s": 5.0},
            {"start_s": 0.0, "end_s": 1.0},  # unknown only -> excluded
        ]
        row = evaluate_file_events(rec, anns, pred, iou_threshold=0.2)
        self.assertEqual(row["n_matched"], 1)
        self.assertEqual(row["n_fp_events"], 1)
        self.assertEqual(row["excluded_unknown_events"], 1)
        self.assertAlmostEqual(row["normal_s"], 5.0, places=6)
        self.assertAlmostEqual(row["anomaly_s"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
