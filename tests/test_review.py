import csv
import tempfile
import unittest
from pathlib import Path

from wind_audio.review import (
    ReviewRecord,
    export_template,
    import_reviews,
    read_reviews,
    reviews_to_annotations,
    review_summary,
)


class ReviewTests(unittest.TestCase):
    def test_feedback_semantics(self):
        self.assertEqual(ReviewRecord("e", "a", l1_label="anomaly", l2_label="nonfault").l1_feedback(), 1)
        self.assertEqual(ReviewRecord("e", "a", l1_label="anomaly", l2_label="fault").l1_feedback(), 1)
        self.assertEqual(ReviewRecord("e", "a", l1_label="normal", l2_label="nonfault").l1_feedback(), 0)
        self.assertIsNone(ReviewRecord("e", "a", l1_label="uncertain").l1_feedback())

    def test_anomaly_nonfault_stays_l1_positive(self):
        reviews = [
            ReviewRecord(event_id="e1", audio_id="a", predicted_start_s=1.0, predicted_end_s=2.0, l1_label="anomaly", l2_label="nonfault"),
            ReviewRecord(event_id="e2", audio_id="a", predicted_start_s=3.0, predicted_end_s=4.0, l1_label="normal", l2_label="nonfault"),
            ReviewRecord(event_id="e3", audio_id="a", predicted_start_s=5.0, predicted_end_s=6.0, l1_label="uncertain"),
        ]
        anns = reviews_to_annotations(reviews)
        self.assertEqual(len(anns), 2)
        self.assertEqual(anns[0].label, "anomaly")
        self.assertEqual(anns[1].label, "normal")

    def test_export_import_roundtrip(self):
        payload = {
            "audio": {"id": "a1", "file": "a1.wav", "path": "data/a1.wav"},
            "model": {"version": "m1"},
            "events": [
                {"event_id": "e1", "start_s": 1.0, "end_s": 2.0, "clip": {"path": "clips/e1.wav"}},
                {"event_id": "e2", "start_s": 3.0, "end_s": 4.0, "clip": {"path": "clips/e2.wav"}},
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            template = d / "review.csv"
            export_template(payload, template, d / "review.json")
            with template.open(encoding="utf-8-sig", newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 2)
            rows[0]["l1_label"] = "anomaly"
            rows[0]["l2_label"] = "nonfault"
            rows[0]["note"] = "异常但非故障"
            rows[1]["l1_label"] = "normal"
            with template.open("w", encoding="utf-8-sig", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            out = d / "reviews" / "reviews.csv"
            merged = import_reviews(template, out, d / "reviews.json")
            self.assertEqual(len(merged), 2)
            self.assertEqual(review_summary(merged)["l1_positive"], 1)
            self.assertEqual(review_summary(merged)["l1_negative"], 1)
            self.assertEqual(len(read_reviews(out)), 2)

    def test_invalid_label_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.csv"
            p.write_text(
                "event_id,audio_id,predicted_start_s,predicted_end_s,l1_label\n"
                "e1,a,1,2,bogus\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                import_reviews(p, Path(td) / "out.csv")


if __name__ == "__main__":
    unittest.main()
