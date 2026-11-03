import tempfile
import unittest
from pathlib import Path

from wind_audio.data import (
    Annotation,
    AudioRecord,
    dataset_summary,
    import_simple,
    label_mask,
    validate_dataset,
)

from tests.helpers import ann, make_wav, record


class LabelMaskTests(unittest.TestCase):
    def test_full_review_derives_normal_but_uncertain_stays_unknown(self):
        rec = record("a", "a.wav", 10.0)
        anns = [
            ann("x1", "a", 2.0, 3.0, "anomaly"),
            ann("x2", "a", 4.0, 5.0, "normal"),
            ann("x3", "a", 6.0, 7.0, "uncertain"),
        ]
        mask = label_mask(rec, anns, 0.1)

        def at(t):
            idx = int(round(t / 0.1 - 0.5))
            return int(mask.labels[idx])

        self.assertEqual(at(2.5), 1)   # explicit anomaly
        self.assertEqual(at(3.5), 0)   # derived normal
        self.assertEqual(at(4.5), 0)   # explicit normal
        self.assertEqual(at(6.5), -1)  # explicit uncertain never becomes normal
        self.assertEqual(at(9.5), 0)   # derived normal
        self.assertAlmostEqual(mask.anomaly_s, 1.0, places=6)
        self.assertAlmostEqual(mask.unknown_s, 1.0, places=6)
        self.assertAlmostEqual(mask.normal_s, 8.0, places=6)

    def test_coarse_file_label_never_expands(self):
        rec = record("a", "a.wav", 10.0, fully_reviewed=False)
        anns = [ann("x1", "a", 0.0, 10.0, "anomaly", precision="file")]
        mask = label_mask(rec, anns, 0.1)
        self.assertTrue((mask.labels == -1).all())
        summary = dataset_summary([rec], anns)
        self.assertEqual(summary["anomaly_s"], 0.0)
        self.assertEqual(summary["coarse_label_files"], 1)

    def test_unreviewed_outside_intervals_is_unknown(self):
        rec = record("a", "a.wav", 10.0, fully_reviewed=False)
        anns = [ann("x1", "a", 2.0, 3.0, "anomaly")]
        mask = label_mask(rec, anns, 0.1)
        self.assertEqual(int(mask.labels[0]), -1)
        self.assertEqual(int(mask.labels[25]), 1)
        self.assertAlmostEqual(mask.unknown_s, 9.0, places=6)


class SimpleImportTests(unittest.TestCase):
    def test_reads_wav_header_and_reviewed_list(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            audio = td / "audio"
            audio.mkdir()
            make_wav(audio / "a.wav", sr=8000, duration=2.0, seed=1)
            make_wav(audio / "b.wav", sr=8000, duration=2.0, seed=2)
            labels = td / "labels.csv"
            labels.write_text(
                "file,start_s,end_s,label,group\n"
                "a.wav,0.5,0.7,anomaly,G1\n",
                encoding="utf-8",
            )
            records, anns, warnings = import_simple(
                audio, labels, td / "manifest.csv", td / "annotations.csv",
                reviewed_files={"b.wav"},
            )
            by_id = {r.audio_id: r for r in records}
            self.assertAlmostEqual(by_id["a"].duration_s, 2.0, places=6)
            self.assertEqual(by_id["a"].sample_rate, 8000)
            self.assertFalse(by_id["a"].fully_reviewed)
            self.assertTrue(by_id["b"].fully_reviewed)

            a_mask = label_mask(by_id["a"], anns, 0.1)
            self.assertEqual(int(a_mask.labels[5]), 1)
            self.assertEqual(int(a_mask.labels[15]), -1)
            b_mask = label_mask(by_id["b"], anns, 0.1)
            self.assertTrue((b_mask.labels == 0).all())
            self.assertTrue(warnings == [])


class ValidationTests(unittest.TestCase):
    def test_out_of_range_and_order(self):
        rec = record("a", "a.wav", 10.0)
        errors, _ = validate_dataset(
            [rec], [ann("x1", "a", 8.0, 11.0)], "manifest.csv", check_audio=False
        )
        self.assertTrue(any("超出" in e for e in errors))
        errors, _ = validate_dataset(
            [rec], [ann("x2", "a", 5.0, 5.0)], "manifest.csv", check_audio=False
        )
        self.assertTrue(any("大于" in e for e in errors))

    def test_conflicting_overlap(self):
        rec = record("a", "a.wav", 10.0)
        anns = [ann("n", "a", 2.0, 4.0, "normal"), ann("x", "a", 3.0, 5.0, "anomaly")]
        errors, _ = validate_dataset([rec], anns, "manifest.csv", check_audio=False)
        self.assertTrue(any("冲突" in e for e in errors))

    def test_fully_reviewed_with_only_coarse_anomaly_is_rejected(self):
        rec = record("a", "a.wav", 10.0, fully_reviewed=True)
        anns = [ann("x", "a", 0.0, 10.0, "anomaly", precision="file")]
        errors, _ = validate_dataset([rec], anns, "manifest.csv", check_audio=False)
        self.assertTrue(any("file 级 anomaly" in e for e in errors))

    def test_unknown_audio_id(self):
        rec = record("a", "a.wav", 10.0)
        errors, _ = validate_dataset(
            [rec], [ann("x", "missing", 1.0, 2.0)], "manifest.csv", check_audio=False
        )
        self.assertTrue(any("未知 audio_id" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
