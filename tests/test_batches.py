"""Multi-batch dataset folders: discovery, merged import and mixed splitting."""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from tests.test_stable_labels import CFG, tone
from wind_audio import cli
from wind_audio.audio import write_wav
from wind_audio.data import AudioRecord
from wind_audio.markers import find_batches, import_batches
from wind_audio.normal import NormalConfig
from wind_audio.normal_fit import fit_normal, split_records


def make_batch(root: Path, name: str, count: int, duration: float = 4.0, seed_base: int = 0):
    """One batch in the delivered layout: Bxx/音频 + Bxx/标注, 正常 ranges only."""
    audio, labels = root / name / "音频", root / name / "标注"
    audio.mkdir(parents=True)
    labels.mkdir()
    for i in range(count):
        # Long device name + increasing counter: sorted order is chronological.
        write_wav(audio / f"10.128.20.1_{i:04d}.wav", tone(duration, seed=seed_base + i), CFG.sr)
        with (labels / f"{name}-{i + 1:03d}.csv").open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(["Name", "Start", "Duration", "Time Format", "Type", "Description"])
            writer.writerow(["正常", "0:00.000", f"0:{duration:06.3f}", "decimal", "Cue", ""])
    return audio, labels


def plain_records(count):
    return [AudioRecord(str(i), f"a{i:02d}.wav") for i in range(count)]


def batch_records(count):
    """Interleaved batches so a shuffled split has to mix, not just slice."""
    return [AudioRecord(f"{n:02d}", f"{'B05' if n % 2 else 'B06'}/a{n:02d}.wav") for n in range(count)]


class BatchDiscoveryTests(unittest.TestCase):
    def test_batch_folders_are_discovered_and_merged(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_batch(root, "B05", 3)
            make_batch(root, "B06", 2)
            batches = find_batches(root)
            self.assertEqual([b[0] for b in batches], ["B05", "B06"])
            self.assertEqual([p.name for p in batches[0][1:]], ["音频", "标注"])
            records, anns, warnings, batch_of = import_batches(
                root, root / "out/manifest.csv", root / "out/annotations.csv", stable_markers=True)
            self.assertEqual(len(records), 5)
            self.assertEqual(len(anns), 5)                      # one full-length 正常 range each
            self.assertEqual(batch_of["B06-001"], "B06")
            self.assertTrue(all(r.fully_reviewed for r in records))
            self.assertTrue(any("批次 B05" in w for w in warnings))
            with (root / "out/pairing.csv").open(encoding="utf-8-sig") as fh:
                pairing = list(csv.DictReader(fh))
            self.assertEqual(pairing[0]["audio_id"], "B05-001")
            self.assertTrue(pairing[0]["audio_path"].endswith("0000.wav"))

    def test_single_directory_root_is_treated_as_one_batch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audio, _ = make_batch(root, "B05", 2)
            self.assertEqual([b[0] for b in find_batches(audio.parent)], ["B05"])

    def test_duplicate_audio_ids_across_batches_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ("B05", "B06"):
                make_batch(root, name, 1)
                (root / name / "标注" / f"{name}-001.csv").rename(root / name / "标注" / "001.csv")
            with self.assertRaisesRegex(ValueError, "重名"):
                import_batches(root, root / "m.csv", root / "a.csv", stable_markers=True)

    def test_missing_batch_layout_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "B05/音频").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "缺少音频目录|标注目录"):
                find_batches(root)


class MixedSplitTests(unittest.TestCase):
    def test_shuffle_is_seeded_disjoint_and_covers_everything(self):
        records = batch_records(30)
        first = split_records(records, mode="shuffle", seed=7)
        second = split_records(records, mode="shuffle", seed=7)
        for group in ("train", "calibration", "test"):
            self.assertEqual([r.audio_id for r in first[group]], [r.audio_id for r in second[group]])
        ids = [r.audio_id for group in first.values() for r in group]
        self.assertEqual(sorted(ids), sorted(r.audio_id for r in records))
        self.assertEqual([len(first[g]) for g in ("train", "calibration", "test")], [18, 6, 6])
        for group in first.values():
            self.assertEqual({r.path.split("/")[0] for r in group}, {"B05", "B06"})
        self.assertNotEqual([r.audio_id for r in split_records(records, mode="shuffle", seed=8)["test"]],
                            [r.audio_id for r in first["test"]])

    def test_temporal_split_remains_the_default_and_unknown_modes_fail(self):
        split = split_records(plain_records(10))
        self.assertEqual([r.audio_id for r in split["train"]], [str(i) for i in range(6)])
        self.assertEqual([r.audio_id for r in split["test"]], ["8", "9"])
        with self.assertRaisesRegex(ValueError, "temporal / shuffle"):
            split_records(plain_records(10), mode="random")


class DatasetTrainingTests(unittest.TestCase):
    def test_one_run_pools_every_batch_and_records_the_policy(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td)
            make_batch(root, "B05", 5)
            make_batch(root, "B06", 5, seed_base=100)
            out = root / "run"
            detector, report = fit_normal(
                dataset_dir=root, out_dir=out, fully_reviewed=True, feature_config=CFG,
                normal=NormalConfig(), split_mode="shuffle", seed=3, verbose=False)
            self.assertEqual(len(report["files"]), 10)
            self.assertEqual({f["batch"] for f in report["files"]}, {"B05", "B06"})
            self.assertEqual([len(v) for v in detector.training["split_ids"].values()], [6, 2, 2])
            self.assertEqual(detector.training["batches"], {"B05": 5, "B06": 5})
            self.assertEqual(detector.training["split_mode"], "shuffle")
            self.assertEqual(detector.training["split_seed"], 3)
            self.assertIn("seed=3", report["split_policy"])
            self.assertTrue(any("打散" in s for s in report["limitations"]))
            self.assertFalse((out / "model.json").exists())     # written by the CLI, not by fit
            self.assertTrue((out / "manifest.csv").is_file())
            self.assertTrue((out / "pairing.csv").is_file())

    def test_cli_pools_batches_and_defaults_to_shuffle(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_batch(root, "B05", 5)
            make_batch(root, "B06", 5, seed_base=100)
            args = cli.build_parser().parse_args(
                ["l1-train-normal", "--dataset-dir", str(root), "--out", str(root / "run"),
                 "--fully-reviewed-normal"])
            self.assertIsNone(args.split)          # resolved to shuffle for batch folders
            self.assertIsNone(args.pairing)        # resolved to sorted for batch folders
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(cli.main(["l1-train-normal", "--dataset-dir", str(root),
                                           "--out", str(root / "cli_run"), "--fully-reviewed-normal",
                                           "--seed", "5"]), 0)
            printed = stdout.getvalue()
            self.assertIn("正常模型已保存", printed)
            self.assertIn("批次 B05", printed)
            self.assertIn("seed=5", printed)
            model = json.loads((root / "cli_run/model.json").read_text())
            self.assertEqual(model["training"]["dataset_dir"], str(root.resolve()))
            self.assertEqual(model["training"]["split_mode"], "shuffle")
            self.assertEqual(model["training"]["batches"], {"B05": 5, "B06": 5})
            report = json.loads((root / "cli_run/training_report.json").read_text())
            self.assertEqual(len(report["files"]), 10)

    def test_dataset_evaluation_writes_the_full_visual_report(self):
        """The delivered report folder: CSVs, brief, per-file plots and one PDF."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_batch(root, "B05", 5)
            make_batch(root, "B06", 5, seed_base=100)
            model_dir, eval_dir = root / "model", root / "report"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["l1-train-normal", "--dataset-dir", str(root),
                                           "--out", str(model_dir), "--fully-reviewed-normal"]), 0)
                self.assertEqual(cli.main(["l1-test", "--dataset-dir", str(root),
                                           "--config", str(model_dir / "model.json"),
                                           "--out", str(eval_dir)]), 0)
            for name in ("annotations.csv", "brief.md", "comparison.pdf", "labels.csv",
                         "manifest.csv", "metrics.json", "normal_regions.csv",
                         "pairing.csv", "per_file.csv"):
                self.assertTrue((eval_dir / name).is_file(), name)
            self.assertGreater((eval_dir / "comparison.pdf").stat().st_size, 1000)
            with (eval_dir / "plots/index.csv").open(encoding="utf-8-sig") as fh:
                pages = list(csv.DictReader(fh))
            self.assertEqual([p["audio_id"] for p in pages],
                             [f"B05-{i:03d}" for i in range(1, 6)] + [f"B06-{i:03d}" for i in range(1, 6)])
            self.assertEqual([int(p["page"]) for p in pages], list(range(1, 11)))
            for page in pages:
                self.assertTrue((eval_dir / page["image"]).is_file(), page["image"])
            metrics = json.loads((eval_dir / "metrics.json").read_text())
            self.assertEqual(metrics["mode"], "l1-test-normal")
            self.assertEqual(metrics["input"]["dataset_dir"], str(root.resolve()))
            self.assertEqual(metrics["visual_report"]["pages"], 10)
            self.assertTrue(any("合并 2 个批次" in w for w in metrics["warnings"]))

    def test_both_input_shapes_at_once_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audio, labels = make_batch(root, "B05", 2)
            with self.assertRaisesRegex(ValueError, "不能同时使用"):
                fit_normal(audio, labels, root / "run", dataset_dir=root, fully_reviewed=True,
                           feature_config=CFG, verbose=False)


if __name__ == "__main__":
    unittest.main()
