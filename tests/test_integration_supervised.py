import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from wind_audio import cli, synthetic

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn 未安装，跳过有监督集成测试")
class SupervisedIntegrationTests(unittest.TestCase):
    def test_synthetic_end_to_end_supervised(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "synthetic"
            synthetic.generate(root, seed=0)
            manifest = root / "manifest.csv"
            annotations = root / "annotations.csv"
            parser = cli.build_parser()

            def run(argv):
                return parser.parse_args(argv).func(parser.parse_args(argv))

            run(["split", "--manifest", str(manifest), "--annotations", str(annotations),
                 "--out", str(root / "splits.json"), "--seed", "7"])
            model = Path(td) / "model"
            run([
                "train", "--manifest", str(manifest), "--annotations", str(annotations),
                "--splits", str(root / "splits.json"), "--detector", "supervised",
                "--algo", "sklearn", "--out", str(model), "--version", "it_sup",
                "--target-recall", "0.9",
            ])
            meta = json.loads((model / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["detector"], "supervised")
            self.assertIn(meta["backend"]["used"], {"sklearn", "xgboost"})
            self.assertIn("feature_names", meta)
            self.assertTrue((model / "calibration.json").exists())

            run([
                "evaluate", "--model", str(model), "--manifest", str(manifest),
                "--annotations", str(annotations), "--splits", str(root / "splits.json"),
                "--split", "test", "--out", str(Path(td) / "eval"),
            ])
            metrics = json.loads((Path(td) / "eval" / "metrics.json").read_text(encoding="utf-8"))
            self.assertIsNotNone(metrics["event_recall"])
            self.assertNotIn("NaN", json.dumps(metrics))

            report_dir = Path(td) / "run"
            run([
                "run", "--input", str(root / "audio" / "syn_mixed.wav"),
                "--model", str(model), "--manifest", str(manifest),
                "--annotations", str(annotations), "--out", str(report_dir),
            ])
            html = (report_dir / "report.html").read_text(encoding="utf-8")
            self.assertIn("L1 异常检测报告", html)
            self.assertTrue((report_dir / "events.json").exists())


if __name__ == "__main__":
    unittest.main()
