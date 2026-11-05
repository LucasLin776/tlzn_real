"""The delivery rule: one label per recording, >5 pieces means whole file."""
import unittest

from wind_audio.labeling import collapse_anomalies, label_from_normal, label_text


class LabelRuleTests(unittest.TestCase):
    def test_empty_means_no_anomaly(self):
        label = collapse_anomalies([], 30.0)
        self.assertEqual(label["intervals"], [])
        self.assertEqual(label["rule"], "none")

    def test_five_or_fewer_kept_as_detected(self):
        pieces = [(1.0, 2.0), (5.0, 6.0), (8.0, 9.0), (12.0, 13.0), (20.0, 21.0)]
        label = collapse_anomalies(pieces, 30.0)
        self.assertEqual(label["rule"], "as_detected")
        self.assertEqual(label["count"], 5)
        self.assertEqual(label["intervals"], [[1.0, 2.0], [5.0, 6.0], [8.0, 9.0],
                                              [12.0, 13.0], [20.0, 21.0]])

    def test_more_than_five_becomes_the_whole_recording(self):
        pieces = [(i, i + 0.5) for i in range(6)]
        label = collapse_anomalies(pieces, 30.0)
        self.assertEqual(label["rule"], "whole_file")
        self.assertEqual(label["intervals"], [[0.0, 30.0]])
        self.assertEqual(label["count"], 6)

    def test_touching_pieces_merge_before_counting(self):
        pieces = [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0), (9.0, 10.0)]
        label = collapse_anomalies(pieces, 30.0)
        self.assertEqual(label["count"], 4)
        self.assertEqual(label["rule"], "as_detected")

    def test_label_from_normal_is_the_complement(self):
        label = label_from_normal([(0.0, 21.73), (23.11, 25.77), (27.33, 30.0)], 30.0)
        self.assertEqual(label["rule"], "as_detected")
        self.assertEqual(len(label["intervals"]), 2)
        self.assertAlmostEqual(label["intervals"][0][0], 21.73, places=6)
        self.assertEqual(label_text(label), "21.73-23.11; 25.77-27.33")

    def test_dict_intervals_are_accepted(self):
        pieces = [{"start_s": 0.0, "end_s": 21.73}, {"start_s": 23.11, "end_s": 25.77},
                  {"start_s": 27.33, "end_s": 30.0}]
        label = label_from_normal(pieces, 30.0)
        self.assertEqual(label["count"], 2)
        self.assertEqual(label["rule"], "as_detected")
        self.assertAlmostEqual(label["intervals"][0][0], 21.73, places=6)
        direct = collapse_anomalies([{"start_s": 1.0, "end_s": 2.0}], 30.0)
        self.assertEqual(direct["intervals"], [[1.0, 2.0]])

    def test_threshold_is_configurable(self):
        pieces = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        self.assertEqual(collapse_anomalies(pieces, 30.0, max_intervals=2)["rule"], "whole_file")
        self.assertEqual(collapse_anomalies(pieces, 30.0, max_intervals=3)["rule"], "as_detected")


if __name__ == "__main__":
    unittest.main()
