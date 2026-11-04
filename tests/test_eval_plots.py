"""Visual export ordering and separation from official evaluation metrics."""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import make_wav
from wind_audio import cli
from wind_audio.eval_plots import ComparisonReport, merge_intervals


class ComparisonPlotTests(unittest.TestCase):
    def test_clustering_copies_intervals_without_changing_originals(self):
        original = [{"start_s": 2., "end_s": 3.}, {"start_s": 1., "end_s": 1.7},
                    {"start_s": 4., "end_s": 4.5}]
        snapshot = json.dumps(original)
        self.assertEqual(merge_intervals(original, .3),
                         [{"start_s": 1., "end_s": 3.}, {"start_s": 4., "end_s": 4.5}])
        self.assertEqual(json.dumps(original), snapshot)

    def test_invalid_visual_gap_rejected(self):
        for gap in (-1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "plot-merge-gap"):
                ComparisonReport("unused", cluster_gap=gap)

    def test_pdf_index_and_images_follow_natural_order_without_changing_metrics(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td)
            for name in ("B10", "B2"):
                make_wav(root / "audio" / f"{name}.wav", duration=1, level=0)
            labels = root / "labels.csv"
            labels.write_text("file,start_s,end_s,label\nB10.wav,0,1,normal\nB2.wav,0,1,normal\n")
            args = ["l1-test", "--audio-dir", str(root / "audio"), "--labels", str(labels)]
            full, numeric = root / "full", root / "numeric"
            self.assertEqual(cli.main([*args, "--plot-merge-gap", ".8", "--out", str(full)]), 0)
            self.assertEqual(cli.main([*args, "--no-plots", "--out", str(numeric)]), 0)
            a, b = [json.loads((folder / "metrics.json").read_text()) for folder in (full, numeric)]
            for key in ("file_metrics", "time_metrics", "event_recall", "event_precision", "n_pred_events"):
                self.assertEqual(a[key], b[key])
            self.assertEqual(a["visual_report"]["order"], ["B2", "B10"])
            self.assertEqual(a["visual_report"]["pages"], 2)
            self.assertEqual(a["visual_report"]["cluster_gap_s"], .8)
            self.assertFalse(a["visual_report"]["cluster_changes_metrics"])
            self.assertFalse((numeric / "comparison.pdf").exists())
            self.assertFalse((full / "comparison.partial.pdf").exists())
            self.assertTrue((full / "comparison.pdf").read_bytes().startswith(b"%PDF-"))
            with (full / "plots/index.csv").open(encoding="utf-8-sig") as fh:
                pages = list(csv.DictReader(fh))
            self.assertEqual([p["audio_id"] for p in pages], ["B2", "B10"])
            for page in pages:
                self.assertTrue((full / page["image"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

