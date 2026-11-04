"""The production path cannot consume truth; both modes use the same detector."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.helpers import make_wav
from wind_audio import cli


class L1InterfaceTests(unittest.TestCase):
    def test_detection_rejects_every_label_input_including_old_alias(self):
        for command in ("l1-detect", "run"):
            for flag in ("--labels", "--annotations", "--manifest", "--review"):
                with self.subTest(command=command, flag=flag), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    cli.build_parser().parse_args([command, "--input", "audio.wav", flag, "labels.csv"])
                self.assertEqual(error.exception.code, 2)

    def test_testing_requires_audio_and_labels_including_old_alias(self):
        for command in ("l1-test", "evaluate"):
            for supplied in (["--audio-dir", "audio"], ["--manifest", "manifest.csv"],
                             ["--labels", "labels.csv"]):
                with self.subTest(command=command, supplied=supplied), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    cli.build_parser().parse_args([command, *supplied])
                self.assertEqual(error.exception.code, 2)

    def test_actual_audio_has_same_windows_and_labels_cannot_change_prediction(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td)
            audio = root / "audio"
            make_wav(audio / "event.wav", sr=44100, duration=4, level=.001, burst=(1.5, .5, .2))
            detect_out = root / "detect"
            # Detection must work without reading any manifest or annotation.
            with patch("wind_audio.cli.read_manifest", side_effect=AssertionError("manifest read")), \
                    patch("wind_audio.cli.read_annotations", side_effect=AssertionError("labels read")):
                self.assertEqual(cli.main(["l1-detect", "--input", str(audio / "event.wav"),
                                           "--threshold", "7", "--out", str(detect_out)]), 0)
            detection = json.loads((detect_out / "events.json").read_text())
            self.assertEqual(detection["mode"], "l1-detect")
            self.assertEqual(detection["annotations"], {"regions": [], "gt_events": [], "source": None})
            windows = [(e["start_s"], e["end_s"]) for e in detection["events"]]
            self.assertTrue(windows)

            for truth in ("anomaly", "normal"):
                labels, test_out = root / f"{truth}.csv", root / f"test_{truth}"
                labels.write_text(f"file,label\nevent.wav,{truth}\n")
                self.assertEqual(cli.main(["l1-test", "--audio-dir", str(audio), "--labels", str(labels),
                                           "--threshold", "7", "--out", str(test_out)]), 0)
                result = json.loads((test_out / "metrics.json").read_text())
                self.assertEqual(result["mode"], "l1-test")
                self.assertEqual(result["detector_config"]["threshold"], 7)
                predicted = result["per_file"][0]["predicted_events"]
                self.assertEqual(windows, [(e["start_s"], e["end_s"]) for e in predicted])
                self.assertEqual(result["file_accuracy"], 1 if truth == "anomaly" else 0)
