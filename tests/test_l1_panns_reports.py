"""Synthetic input/PDF contracts. Training and calibration are NEVER executed."""
import contextlib
import copy
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from wind_audio.audio import wav_info, write_wav
from wind_audio.cli import main
from wind_audio.l1_panns.config import EventConfig, NetworkConfig
from wind_audio.l1_panns.data import from_directories, provenance
from wind_audio.l1_panns.events import segment_scores
from wind_audio.l1_panns.reporting import write_evaluation_report
from wind_audio.l1_panns.workflow import preflight, run_training


def make_inputs(root):
    audio, labels = root / "audio", root / "labels"
    audio.mkdir(parents=True)
    labels.mkdir()
    assignments = [("rec1", "train"), ("rec2", "calibration"), ("rec3", "val"),
                   ("rec10", "test"), ("rec4", "test")]
    for i, (aid, split) in enumerate(assignments):
        x = np.random.default_rng(i + 1).normal(0, .02, 8000 * 5).astype(np.float32)
        write_wav(audio / f"{aid}.wav", x, 8000)
        if split in {"train", "calibration"}:
            text = "正常,0,5\n"
        else:
            text = "正常,0,1.6\n冲击,2.0,0.5\n正常,3,2\n"  # gaps remain unknown
        (labels / f"{aid}.csv").write_text("Name,Start,Duration\n" + text, encoding="utf-8")
    (root / "splits.csv").write_text("audio_id,source_id,split\n" + "".join(
        f"{aid},session_{aid},{split}\n" for aid, split in assignments), encoding="utf-8")
    (root / "semantics.json").write_text(json.dumps({"name_map": {"正常": "normal", "冲击": "anomaly"}}), encoding="utf-8")
    # Existence-only preflight fixture, never opened as a model checkpoint.
    (root / "DO_NOT_LOAD.pth").write_bytes(b"SYNTHETIC PATH CHECK ONLY")
    return from_directories(audio, labels, pairing="stem", semantics={"name_map": {"正常": "normal", "冲击": "anomaly"}},
                            splits_path=root / "splits.csv")


class FixedPredictionModel:
    """Hand-authored residuals/thresholds. Not a trained or calibrated model."""
    def __init__(self, data):
        self.cfg = NetworkConfig()
        self.meta = {"training": {"provenance": provenance([r for r in data["records"] if r["split"] == "train"])},
                     "calibration": {"provenance": provenance([r for r in data["records"] if r["split"] == "calibration"])}}
        self.calls = []

    def require_ready(self):
        pass  # this object is a test double, never serializable as a real model

    def predict(self, path):
        self.calls.append(str(path))
        duration = wav_info(path).duration_s
        times = self.cfg.offset_s + np.arange(int(duration / self.cfg.step_s)) * self.cfg.step_s
        valid = (times >= self.cfg.context_radius_s) & (times <= duration - self.cfg.context_radius_s)
        scores = np.full(len(times), .03)
        scores[(times > 2.05) & (times < 2.35)] = .8
        thresholds = {"low": .1, "high": .2, "impulse": .4}
        result = segment_scores(times, scores, valid, duration, thresholds, EventConfig(), self.cfg.step_s)
        states = result.pop("states")
        return {"audio": {"path": str(path), "duration_s": duration}, "model": {"thresholds": thresholds},
                **result, "scores": {"time_s": times.tolist(), "state": states.tolist(), "valid": valid.tolist(),
                                      "values": [float(x) if v else None for x, v in zip(scores, valid)]},
                "diagnostics": self.cfg.timing()}


class DirectoryInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = make_inputs(self.root)

    def direct(self, **kwargs):
        options = {"pairing": "stem", "semantics": {"all_markers": "normal"},
                   "splits_path": self.root / "splits.csv", **kwargs}
        return from_directories(self.root / "audio", self.root / "labels", **options)

    def command(self):
        return ["l1-panns-train", "--audio-dir", str(self.root / "audio"), "--labels", str(self.root / "labels"),
                "--pairing", "stem", "--splits", str(self.root / "splits.csv"),
                "--semantics", str(self.root / "semantics.json"), "--checkpoint", str(self.root / "DO_NOT_LOAD.pth"),
                "--out", str(self.root / "new_run"), "--min-frames", "2"]

    def test_explicit_stem_sorted_and_table_pairing(self):
        self.assertEqual(len(self.data["records"]), 5)
        sorted_data = self.direct(pairing="sorted")
        self.assertEqual([r["audio_id"] for r in sorted_data["records"]], ["rec1", "rec2", "rec3", "rec4", "rec10"])
        table = self.root / "pairs.csv"
        table.write_text("audio_id,audio_path,label_path\n" + "".join(
            f"{r['audio_id']},{r['audio_id']}.wav,{r['audio_id']}.csv\n" for r in self.data["records"]))
        mapped = self.direct(pairing="table", pairing_file=table)
        self.assertEqual(len(mapped["records"]), 5)
        with self.assertRaisesRegex(ValueError, "显式指定"):
            self.direct(pairing=None)

    def test_invalid_audio_and_label_paths_counts_and_semantics(self):
        for root, name in ((self.root / "missing_audio", "音频"), (self.root / "missing_labels", "标注")):
            with self.assertRaisesRegex(ValueError, name + "目录"):
                from_directories(root if name == "音频" else self.root / "audio",
                                 root if name == "标注" else self.root / "labels", pairing="stem",
                                 semantics={"all_markers": "normal"}, splits_path=self.root / "splits.csv")
        with self.assertRaisesRegex(ValueError, "语义"):
            self.direct(semantics=None)
        (self.root / "labels/extra.csv").write_text("Name,Start,Duration\n")
        with self.assertRaisesRegex(ValueError, "数量不同"):
            self.direct()

    def test_malformed_and_out_of_range_labels_fail_with_path(self):
        label = self.root / "labels/rec1.csv"
        for text in ("wrong,header\n", "Name,Start,Duration\n正常,4,2\n", "Name,Start,Duration\n正常,nan,2\n"):
            label.write_text(text)
            with self.assertRaisesRegex(ValueError, "rec1"):
                self.direct()

    def test_invalid_wav_and_semantics_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "name_map"):
            self.direct(semantics={"name_map": []})
        (self.root / "audio/rec1.wav").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "rec1"):
            self.direct()

    def test_missing_splits_cross_source_and_duplicate_audio_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "splits"):
            self.direct(splits_path=None)
        split_path = self.root / "splits.csv"
        original = split_path.read_text()
        split_path.write_text(original.replace("session_rec2", "session_rec1"))
        with self.assertRaisesRegex(ValueError, "来源跨"):
            self.direct()
        split_path.write_text(original)
        (self.root / "audio/rec2.wav").write_bytes((self.root / "audio/rec1.wav").read_bytes())
        with self.assertRaisesRegex(ValueError, "重复音频"):
            self.direct()

    def test_preflight_context_and_required_partitions_no_fitting(self):
        with patch("numpy.quantile", side_effect=AssertionError("NO FITTING")):
            check = preflight(self.data, NetworkConfig(), min_frames=2)
        self.assertGreater(check["normal_training_frames"], 0)
        self.assertGreater(check["normal_calibration_frames"], 0)
        self.assertFalse(check["features_extracted"])
        missing_val = copy.deepcopy(self.data)
        missing_val["records"] = [r for r in missing_val["records"] if r["split"] != "val"]
        with self.assertRaisesRegex(ValueError, "val"):
            preflight(missing_val, NetworkConfig(), min_frames=2)
        short = copy.deepcopy(self.data)
        short["records"][0]["intervals"] = [{"start_s": 0, "end_s": .5, "label": "normal"}]
        with self.assertRaisesRegex(ValueError, "完整特征"):
            preflight(short, NetworkConfig(), min_frames=2)
        with self.assertRaisesRegex(ValueError, "少于"):
            preflight(self.data, NetworkConfig(), min_frames=100000)

    def test_check_only_never_loads_model_or_creates_outputs(self):
        with patch("wind_audio.l1_panns.cli.model_module", side_effect=AssertionError("NO MODEL")), \
                patch("wind_audio.l1_panns.workflow.run_training", side_effect=AssertionError("NO TRAINING")), \
                patch("numpy.quantile", side_effect=AssertionError("NO CALIBRATION")), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main([*self.command(), "--check-only"]), 0)
        self.assertIn("仅输入校验完成", output.getvalue())
        self.assertFalse((self.root / "new_run").exists())

    def test_bad_input_does_not_enter_training(self):
        (self.root / "labels/rec1.csv").write_text("bad,header\n")
        with patch("wind_audio.l1_panns.workflow.run_training", side_effect=AssertionError("NO TRAINING")), \
                contextlib.redirect_stderr(io.StringIO()) as output:
            self.assertEqual(main(self.command()), 2)
        self.assertIn("rec1", output.getvalue())
        self.assertFalse((self.root / "new_run").exists())

    def test_snapshot_and_direct_label_arguments_cannot_be_mixed(self):
        with patch("wind_audio.l1_panns.cli.model_module", side_effect=AssertionError("NO MODEL")), \
                contextlib.redirect_stderr(io.StringIO()) as output:
            self.assertEqual(main(["l1-panns-train", "--dataset", "unused.json", "--labels", "unused",
                                   "--checkpoint", str(self.root / "DO_NOT_LOAD.pth"),
                                   "--out", str(self.root / "new_run"), "--check-only"]), 2)
        self.assertIn("不能与", output.getvalue())


class SyntheticReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = make_inputs(self.root)
        # Prevent accidental execution of ANY real training/calibration code.
        for target in ("wind_audio.l1_panns.model.train", "wind_audio.l1_panns.model.calibrate", "numpy.quantile"):
            guard = patch(target, side_effect=AssertionError("NO TRAINING OR CALIBRATION"))
            guard.start()
            self.addCleanup(guard.stop)

    def test_pdf_csv_json_natural_order_and_predictions_unchanged(self):
        from wind_audio.l1_panns.model import evaluate
        model = FixedPredictionModel(self.data)
        expected = evaluate(model, self.data, "test")
        model.calls.clear()
        snapshot = copy.deepcopy(self.data)
        out = self.root / "report"
        result = write_evaluation_report(model, self.data, "test", out)
        self.assertEqual(len(model.calls), 2)  # one shared prediction per recording, no second inference for plots
        self.assertEqual(result["per_file"], expected["per_file"])
        self.assertEqual(result["duration_metrics"], expected["duration_metrics"])
        self.assertEqual(self.data, snapshot)
        self.assertEqual(result["visual_report"]["order"], ["rec4", "rec10"])
        for name in ("comparison.pdf", "metrics.json", "per_file.csv", "normal_regions.csv", "events.csv",
                     "unknown_regions.csv", "labels.csv", "manifest.csv", "annotations.csv", "pairing.csv", "brief.md"):
            self.assertTrue((out / name).is_file(), name)
        self.assertTrue((out / "comparison.pdf").read_bytes().startswith(b"%PDF-"))
        self.assertFalse((out / "comparison.partial.pdf").exists())
        with (out / "plots/index.csv").open(encoding="utf-8-sig") as f:
            pages = list(csv.DictReader(f))
        self.assertEqual([p["audio_id"] for p in pages], ["rec4", "rec10"])
        self.assertTrue(all((out / p["image"]).read_bytes().startswith(b"\x89PNG") for p in pages))
        with self.assertRaisesRegex(ValueError, "已存在"):
            write_evaluation_report(model, self.data, "test", out)

    def test_mocked_training_workflow_produces_both_split_pdfs(self):
        from wind_audio.l1_panns.model import new_directory
        model, calls = FixedPredictionModel(self.data), []

        def fake_train(*args, **kwargs):
            calls.append("train_stub")
            return SimpleNamespace(meta={"training": {"optimizer_steps": 0, "fixture_only": True}})

        def fake_calibrate(*args, **kwargs):
            calls.append("calibration_stub")
            return model

        stub = SimpleNamespace(new_directory=new_directory, train=fake_train, calibrate=fake_calibrate)
        validation = preflight(self.data, NetworkConfig(), min_frames=2)
        with patch("wind_audio.l1_panns.cli.model_module", return_value=stub):
            run_training(self.data, self.root / "DO_NOT_LOAD.pth", self.root / "run", cfg=NetworkConfig(),
                         event_cfg=EventConfig(), validation=validation, min_frames=2)
        self.assertEqual(calls, ["train_stub", "calibration_stub"])
        self.assertEqual(len(model.calls), 3)
        self.assertTrue((self.root / "run/val/comparison.pdf").is_file())
        self.assertTrue((self.root / "run/test/comparison.pdf").is_file())
        run = json.loads((self.root / "run/run.json").read_text())
        self.assertEqual(run["status"], "complete")
        self.assertEqual(set(run["reports"]), {"val", "test"})

    def test_json_only_evaluation_uses_same_logic_without_pdf(self):
        model = FixedPredictionModel(self.data)
        with patch("wind_audio.l1_panns.model.Model.load", return_value=model), \
                patch("wind_audio.l1_panns.cli.load_dataset", return_value=self.data), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["l1-panns-evaluate", "--model", "unused", "--dataset", "unused", "--split", "val",
                                   "--out", str(self.root / "only.json"), "--json-only"]), 0)
        self.assertTrue((self.root / "only.json").is_file())
        self.assertEqual(len(model.calls), 1)

    def test_font_failure_precedes_workflow_training(self):
        validation = preflight(self.data, NetworkConfig(), min_frames=2)
        with patch("wind_audio.l1_panns.cli.model_module", side_effect=AssertionError("NO MODEL")), \
                self.assertRaisesRegex(ValueError, "字体不存在"):
            run_training(self.data, "unused", self.root / "fail", cfg=NetworkConfig(), event_cfg=EventConfig(),
                         validation=validation, min_frames=2, font_path=self.root / "missing.ttf")
        self.assertFalse((self.root / "fail").exists())

    def test_train_only_has_no_calibration_or_pdfs(self):
        from wind_audio.l1_panns.model import new_directory
        stub = SimpleNamespace(new_directory=new_directory,
                               train=lambda *a, **kw: SimpleNamespace(meta={"training": {"optimizer_steps": 0}}),
                               calibrate=lambda *a, **kw: self.fail("train-only must never calibrate"))
        with patch("wind_audio.l1_panns.cli.model_module", return_value=stub):
            run_training(self.data, "unused", self.root / "train_only", cfg=NetworkConfig(), event_cfg=EventConfig(),
                         validation=preflight(self.data, NetworkConfig(), reports=False), train_only=True)
        self.assertFalse((self.root / "train_only/val").exists())
        run = json.loads((self.root / "train_only/run.json").read_text())
        self.assertEqual(run["status"], "trained_uncalibrated")

    def test_report_rejects_training_source_before_creating_output(self):
        model = FixedPredictionModel(self.data)
        bad = copy.deepcopy(self.data)
        training_source = model.meta["training"]["provenance"][0]["source_id"]
        next(r for r in bad["records"] if r["split"] == "val")["source_id"] = training_source
        with self.assertRaisesRegex(ValueError, "泄漏"):
            write_evaluation_report(model, bad, "val", self.root / "leaked")
        self.assertFalse((self.root / "leaked").exists())
