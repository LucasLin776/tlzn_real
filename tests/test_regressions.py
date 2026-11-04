import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from wind_audio import cli
from wind_audio.data import read_annotations
from wind_audio.detectors import SpectralChangeDetector


class RegressionTests(unittest.TestCase):
    def test_missing_annotations_fail(self):
        with self.assertRaises(FileNotFoundError):
            read_annotations("/missing/annotations.csv")

    def test_no_train_or_model_required(self):
        args = cli.build_parser().parse_args(["run", "--input", "a.wav"])
        self.assertIsNone(args.config)
        for command in ("train", "calibrate"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.build_parser().parse_args([command])

    def test_batch_partial_failures_return_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ("a.wav", "b.wav"):
                (root / name).touch()
            args = cli.build_parser().parse_args(["run", "--input", str(root), "--out", str(root / "out")])
            ok = {"events": 0, "report": str(root / "out/a/report.html"), "duration_s": 1,
                  "alarm_s": 0, "windows": [], "file": "a.wav", "filtered_candidates": 0}
            for outcomes, expected in (([ok, ok], 0), ([ok, ValueError("bad")], 1),
                                       ([ValueError("bad"), ValueError("bad")], 1)):
                with patch("wind_audio.cli._run_one", side_effect=outcomes), \
                        contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.cmd_run(args), expected)
