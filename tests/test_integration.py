import json
import tempfile
import unittest
from pathlib import Path

from wind_audio import cli, synthetic


class IntegrationTests(unittest.TestCase):
    def test_synthetic_end_to_end_baseline(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "synthetic"
            synthetic.generate(root, seed=0)
            manifest = root / "manifest.csv"
            annotations = root / "annotations.csv"
            parser = cli.build_parser()

            def run(argv):
                args = parser.parse_args(argv)
                return args.func(args)

            self.assertEqual(run(["check", "--manifest", str(manifest), "--annotations", str(annotations)]), 0)
            self.assertEqual(
                run([
                    "split", "--manifest", str(manifest), "--annotations", str(annotations),
                    "--out", str(root / "splits.json"), "--seed", "7",
                ]),
                0,
            )
            model = Path(td) / "model"
            self.assertEqual(
                run([
                    "train", "--manifest", str(manifest), "--annotations", str(annotations),
                    "--splits", str(root / "splits.json"), "--detector", "baseline",
                    "--out", str(model), "--version", "it_base", "--target-recall", "0.9",
                ]),
                0,
            )
            self.assertTrue((model / "meta.json").exists())
            self.assertTrue((model / "calibration.json").exists())

            self.assertEqual(
                run([
                    "evaluate", "--model", str(model), "--manifest", str(manifest),
                    "--annotations", str(annotations), "--splits", str(root / "splits.json"),
                    "--split", "test", "--out", str(Path(td) / "eval"),
                ]),
                0,
            )
            metrics = json.loads((Path(td) / "eval" / "metrics.json").read_text(encoding="utf-8"))
            self.assertIn("event_recall", metrics)
            self.assertIn("fa_per_hour", metrics)
            self.assertNotIn("NaN", json.dumps(metrics))

            run_dir = Path(td) / "run"
            self.assertEqual(
                run([
                    "run", "--input", str(root / "audio" / "syn_mixed.wav"),
                    "--model", str(model), "--manifest", str(manifest),
                    "--annotations", str(annotations), "--out", str(run_dir),
                    "--threshold", "2.5",
                ]),
                0,
            )
            self.assertTrue((run_dir / "report.html").exists())
            self.assertTrue((run_dir / "waveform.png").exists())
            self.assertTrue((run_dir / "spectrogram.png").exists())
            self.assertTrue((run_dir / "score.png").exists())
            self.assertTrue((run_dir / "review.csv").exists())
            payload = json.loads((run_dir / "events.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["model"]["threshold"], 2.5)
            self.assertEqual(payload["model"]["threshold_source"], "cli")
            self.assertTrue(payload["audio"]["duration_s"] > 0)
            self.assertTrue((run_dir / "audio").exists())


if __name__ == "__main__":
    unittest.main()
