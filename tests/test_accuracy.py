"""Hand-calculated accuracy cases and real WAV -> CLI -> report checks."""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import ann, make_wav, record
from wind_audio import cli
from wind_audio.data import import_simple, read_annotations, validate_dataset
from wind_audio.evaluate import aggregate_file_metrics, evaluate_file_events
from wind_audio.markers import import_markers, _seconds


class AccuracyTests(unittest.TestCase):
    def test_time_confusion_uses_union_and_excludes_unknown(self):
        rec = record("a", "a.wav", 10, fully_reviewed=False)
        labels = [ann("a1", "a", 2, 4), ann("n1", "a", 5, 9, "normal")]
        predictions = [{"start_s": 1, "end_s": 3}, {"start_s": 2, "end_s": 3},
                       {"start_s": 6, "end_s": 8}]
        row = evaluate_file_events(rec, labels, predictions)
        metrics = row["time_metrics"]
        self.assertEqual([metrics[k] for k in ("tp", "tn", "fp", "fn")], [1, 2, 2, 1])
        self.assertEqual(metrics["accuracy"], .5)
        self.assertEqual(row["unknown_s"], 4)
        self.assertEqual(row["n_matched"], 1)
        self.assertEqual(len(row["matches"]), 1)

    def test_file_confusion_and_coarse_labels_do_not_expand_to_time(self):
        rows = []
        for index, (truth, has_prediction) in enumerate([
                ("anomaly", True), ("normal", False), ("normal", True), ("anomaly", False)]):
            aid = str(index)
            rec = record(aid, aid + ".wav", 10, fully_reviewed=False)
            labels = [ann(aid, aid, 0, 10, truth, precision="file")]
            rows.append(evaluate_file_events(rec, labels, [{"start_s": 1, "end_s": 2}] if has_prediction else []))
        rows.append(evaluate_file_events(record("unknown", "u.wav", 10, fully_reviewed=False), [], []))
        result = aggregate_file_metrics(rows)
        self.assertEqual(result["file_accuracy"], .5)
        self.assertEqual([result["file_metrics"][k] for k in ("tp", "tn", "fp", "fn", "excluded")], [1]*5)
        self.assertIsNone(result["time_accuracy"])
        self.assertIsNone(result["event_recall"])

    def test_unknown_normal_remainder_is_not_negative_file(self):
        rec = record("a", "a.wav", 10, fully_reviewed=False)
        result = evaluate_file_events(rec, [ann("n", "a", 1, 2, "normal")], [])
        self.assertIsNone(result["file_true_label"])
        self.assertEqual(result["time_accuracy"], 1)

    def test_time_aggregation_weights_seconds_not_files(self):
        a = evaluate_file_events(record("a", "a.wav", 1), [], [])
        b = evaluate_file_events(record("b", "b.wav", 9), [], [{"start_s": 0, "end_s": 9}])
        result = aggregate_file_metrics([a, b])
        self.assertAlmostEqual(result["time_accuracy"], .1)
        self.assertEqual(result["file_accuracy"], .5)

    def test_conflicting_file_truth_rejected(self):
        rec = record("a", "a.wav", 10, fully_reviewed=False)
        for labels in ([ann("n", "a", 0, 10, "normal", "file"), ann("a", "a", 0, 10, "anomaly", "file")],
                       [ann("n", "a", 0, 10, "normal", "file"), ann("a", "a", 1, 2)]):
            errors, _ = validate_dataset([rec], labels, "manifest.csv", check_audio=False)
            self.assertTrue(any("冲突" in e for e in errors))


class ImportAccuracyTests(unittest.TestCase):
    def test_nonfinite_manifest_annotation_is_not_silently_zero(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "annotations.csv"
            path.write_text("annotation_id,audio_id,start_s,end_s,label,precision\na,a,nan,2,anomaly,interval\n")
            labels = read_annotations(path)
            errors, _ = validate_dataset([record("a", "a.wav", 3)], labels, "manifest.csv", check_audio=False)
            self.assertTrue(any("有限" in e for e in errors))

    def test_file_labels_json_numeric_zero_and_uppercase_wav(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_wav(root / "audio/a.WAV", duration=2)
            labels = root / "labels.json"
            labels.write_text(json.dumps([{"file": "a.WAV", "label": 0}]))
            records, anns, _ = import_simple(root / "audio", labels, root / "manifest.csv", root / "annotations.csv")
            self.assertEqual(anns[0].label, "normal")
            self.assertEqual(anns[0].precision, "file")
            self.assertEqual(anns[0].end_s, 2)
            result = evaluate_file_events(records[0], anns, [])
            self.assertEqual(result["file_outcome"], "TN")
            self.assertIsNone(result["time_accuracy"])

    def test_partial_time_field_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_wav(root / "audio/a.wav", duration=2)
            labels = root / "labels.csv"
            labels.write_text("file,label,start_s\na.wav,anomaly,1\n")
            with self.assertRaisesRegex(ValueError, "同时填写"):
                import_simple(root / "audio", labels, root / "manifest.csv", root / "annotations.csv")
            self.assertFalse((root / "manifest.csv").exists())

    def test_marker_times(self):
        self.assertAlmostEqual(_seconds("0:10.877"), 10.877)
        self.assertEqual(_seconds("1:02:03.5"), 3723.5)
        for value in ("NaN", "inf", "0:60", "-1:20", "abc"):
            with self.assertRaises(ValueError):
                _seconds(value)

    def test_marker_cli_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td)
            audio, labels, out = root / "audio", root / "labels", root / "result"
            labels.mkdir()
            make_wav(audio / "202601010000.wav", sr=44100, duration=4, level=0)
            make_wav(audio / "202601010001.wav", sr=44100, duration=4,
                     level=.001, burst=(1.5, .5, .2))
            header = "Name\tStart\tDuration\tTime Format\tType\tDescription\n"
            (labels / "B-001.csv").write_text(header)
            (labels / "B-002.csv").write_text(header + "启动\t0:01.500\t0:00.500\tdecimal\tCue\t\n")
            with self.assertRaisesRegex(ValueError, "一一对应"):
                import_markers(audio, labels, out / "manifest.csv", out / "annotations.csv")
            args = ["evaluate", "--audio-dir", str(audio), "--labels", str(labels), "--pairing", "sorted",
                    "--assume-unlabeled-normal", "--out", str(out)]
            self.assertEqual(cli.main(args), 0)
            result = json.loads((out / "metrics.json").read_text())
            self.assertEqual(result["file_accuracy"], 1)
            self.assertEqual(result["file_metrics"]["tn"], 1)
            self.assertEqual(result["file_metrics"]["tp"], 1)
            self.assertEqual(result["n_gt_events"], 1)
            self.assertEqual(result["n_matched"], 1)
            self.assertEqual(result["unknown_s"], 0)
            self.assertEqual(result["detector_config"]["version"], "spectral_event_v2")
            for name in ("brief.md", "per_file.csv", "events.csv", "manifest.csv", "annotations.csv", "pairing.csv"):
                self.assertTrue((out / name).is_file(), name)
            with (out / "pairing.csv").open(encoding="utf-8-sig") as fh:
                mapping = list(csv.DictReader(fh))
            self.assertEqual(Path(mapping[0]["audio_path"]).name, "202601010000.wav")
            before = (out / "metrics.json").read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(args), 2)
            self.assertEqual((out / "metrics.json").read_bytes(), before)

    def test_empty_markers_are_unknown_without_review_declaration(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_wav(root / "audio/a.wav", duration=2)
            labels = root / "labels"
            labels.mkdir()
            (labels / "a.csv").write_text("Name\tStart\tDuration\n")
            records, anns, _ = import_markers(root / "audio", labels, root / "manifest.csv", root / "annotations.csv")
            row = evaluate_file_events(records[0], anns, [])
            self.assertIsNone(row["file_true_label"])
            self.assertIsNone(row["time_accuracy"])
