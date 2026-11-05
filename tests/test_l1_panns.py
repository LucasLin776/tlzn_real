"""Synthetic contract/forward tests ONLY. Never optimize or fit thresholds.

Checkpoint fixtures have official-compatible SHAPES but random CNN parameters;
they are NOT pretrained PANNs and cannot establish detection performance.
"""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from wind_audio.audio import write_wav
from wind_audio.cli import build_parser, main
from wind_audio.l1_panns import MODEL_TYPE
from wind_audio.l1_panns.config import EventConfig, NetworkConfig
from wind_audio.l1_panns.data import (assert_disjoint, canonical_intervals, contained, load_audio,
                                     load_dataset, normal_crops, prepare, read_markers, validate)
from wind_audio.l1_panns.events import segment_scores

try:
    import torch
    from wind_audio.l1_panns.model import Model, PREPROCESSING, evaluate
    from wind_audio.l1_panns.network import ConvBlock, FrozenPanns, TemporalDAE, dynamics
except (ImportError, OSError):
    torch = None


def fixture_metadata(cfg):
    """Simulated lifecycle flags for serialization tests; NO optimizer has run."""
    return {"schema_version": 4, "model_type": MODEL_TYPE, "trained": True,
            "network_config": cfg.to_dict(), "event_config": EventConfig().to_dict(),
            "preprocessing": PREPROCESSING, "timing": cfg.timing(),
            "normalization_scale": [0.01] * cfg.input_dim,
            "training": {"optimizer_steps": 1, "provenance": [], "fixture_only": True},
            "calibration": {"thresholds": {"low": 0.1, "high": 0.2, "impulse": 0.4},
                            "provenance": [], "fixture_only": True}}


class PannsDataTests(unittest.TestCase):
    def test_semantics_are_explicit_not_names_and_empty_is_unknown(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "labels.csv"
            p.write_text("\ufeffName\tStart\tDuration\tTime Format\n正常\t0:01.000\t0:02.000\tdecimal\n")
            self.assertEqual(read_markers(p, {"all_markers": "anomaly"})[0]["label"], "anomaly")
            with self.assertRaises(ValueError):
                read_markers(p, {})
            with self.assertRaises(ValueError):
                read_markers(p, {"name_map": {"different": "normal"}})
            p.write_text("Name,Start,Duration\n")
            self.assertEqual(read_markers(p, {"all_markers": "normal"}), [])
        self.assertFalse(contained(np.arange(3), [], 0).any())

    def test_import_explicit_pairs_and_hash_duplicates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name, value in (("z.wav", 0.1), ("a.wav", 0.2)):
                write_wav(root / name, np.full(32000, value), 32000)
            (root / "ordinal-001.csv").write_text("Name,Start,Duration\nreview,0.2,0.5\n")
            (root / "ordinal-002.csv").write_text("Name,Start,Duration\n")
            (root / "semantics.json").write_text('{"all_markers":"normal"}')
            p = root / "pairing.csv"
            p.write_text("audio_id,audio_path,label_path,source_id,split\n"
                         "one,z.wav,ordinal-001.csv,session-a,train\n"
                         "two,a.wav,ordinal-002.csv,session-b,test\n")
            obj = prepare(p, root / "semantics.json", root / "data.json")
            self.assertTrue(obj["records"][0]["audio_path"].endswith("z.wav"))
            self.assertEqual(obj["records"][1]["intervals"], [])
            self.assertEqual(load_dataset(root / "data.json")["unmarked"], "unknown")
            other = copy.deepcopy(obj)
            other["records"][1]["source_id"] = "session-a"
            with self.assertRaisesRegex(ValueError, "来源跨"):
                validate(other)
            other = copy.deepcopy(obj)
            other["records"][1]["pcm_sha256"] = other["records"][0]["pcm_sha256"]
            with self.assertRaisesRegex(ValueError, "重复"):
                validate(other)
            with self.assertRaises(FileExistsError):
                prepare(p, root / "semantics.json", root / "data.json")
            write_wav(root / "z.wav", np.full(32000, 0.3), 32000)
            with self.assertRaisesRegex(ValueError, "变更"):
                load_dataset(root / "data.json")

    def test_no_complement_labels_conflicts_and_no_cropping_overflow(self):
        with self.assertRaises(ValueError):
            canonical_intervals([{"start_s": 0, "end_s": 2, "label": "normal"}], 1)
        with self.assertRaises(ValueError):
            canonical_intervals([{"start_s": 0, "end_s": 2, "label": "normal"},
                                 {"start_s": 1, "end_s": 3, "label": "anomaly"}], 4)
        cfg = NetworkConfig()
        row = {"duration_s": 30, "intervals": [
            {"start_s": 2.2, "end_s": 16.3, "label": "normal"},
            {"start_s": 16.3, "end_s": 20, "label": "anomaly"},
            {"start_s": 25, "end_s": 25.5, "label": "normal"}]}
        crops = list(normal_crops(row, cfg, .001))
        self.assertTrue(crops)
        for i, (a, b) in enumerate(crops):
            self.assertGreaterEqual(a / cfg.sample_rate - .001, 2.2)
            self.assertLessEqual(b / cfg.sample_rate + .001, 16.3)
            if i:
                self.assertGreaterEqual(a, crops[i - 1][1])
        self.assertAlmostEqual(cfg.context_radius_s, .991)
        self.assertAlmostEqual(cfg.offset_s, .051)
        mask = contained(np.array([2.5, 4, 16]), row["intervals"], cfg.context_radius_s)
        np.testing.assert_array_equal(mask, [False, True, False])

    def test_local_resampling_cannot_leak_distant_unknown_impulse(self):
        with tempfile.TemporaryDirectory() as td:
            wave = np.zeros(44100 * 3, np.float32)
            wave[44100 * 2] = 1
            path = Path(td) / "source.wav"
            write_wav(path, wave, 44100)
            result, _, margin = load_audio(path)
            self.assertEqual(len(result), 32000 * 3)
            self.assertEqual(margin, 16 / 44100)
            self.assertEqual(float(np.abs(result[:32000]).max()), 0)

    def test_cross_artifact_leakage_is_rejected(self):
        one = {"source_id": "a", "audio_sha256": "1", "pcm_sha256": "x"}
        for key in one:
            other = {"source_id": "b", "audio_sha256": "2", "pcm_sha256": "y", key: one[key]}
            with self.assertRaisesRegex(ValueError, "泄漏"):
                assert_disjoint([other], [one])


class PannsEventTests(unittest.TestCase):
    def segment(self, scores, *, valid=None, event_cfg=None):
        scores = np.array(scores, float)
        return segment_scores((np.arange(len(scores)) + .5) * .08, scores,
                              np.ones(len(scores), bool) if valid is None else valid,
                              len(scores) * .08, {"low": 1, "high": 2, "impulse": 4},
                              event_cfg or EventConfig(), .08)

    def test_stable_and_small_noise_do_not_alarm_strong_single_impulse_does(self):
        stable = self.segment([0] * 20)
        self.assertEqual(len(stable["events"]), 0)
        self.assertEqual(len(stable["normal_regions"]), 1)
        mild = self.segment([0] * 8 + [2.5] + [0] * 8)
        self.assertEqual(mild["events"], [])
        impulse = self.segment([0] * 8 + [5] + [0] * 8)
        self.assertEqual(len(impulse["events"]), 1)
        self.assertAlmostEqual(impulse["events"][0]["duration_s"], .08)

    def test_merge_respects_confirmed_normal_even_with_large_merge_gap(self):
        cfg = EventConfig(confirm_s=.32, merge_gap_s=10)
        separated = self.segment([0] * 5 + [5] + [0] * 4 + [5] + [0] * 5, event_cfg=cfg)
        self.assertEqual(len(separated["events"]), 2)
        close = self.segment([0] * 5 + [5, 0, 5] + [0] * 5)
        self.assertEqual(len(close["events"]), 1)
        unknown = self.segment([5, 0, 5], valid=[True, False, True], event_cfg=cfg)
        self.assertEqual(len(unknown["events"]), 2)

    def test_settled_new_background_ends_event_and_edges_remain_unknown(self):
        result = self.segment([0] * 5 + [3] * 3 + [0] * 8,
                              valid=[False] * 2 + [True] * 12 + [False] * 2)
        self.assertEqual(len(result["events"]), 1)
        self.assertAlmostEqual(result["events"][0]["end_s"], 8 * .08)
        self.assertEqual(result["unknown_regions"][0]["start_s"], 0)
        self.assertEqual(result["unknown_regions"][-1]["end_s"], 16 * .08)
        short = self.segment([0] * 3, valid=[False] * 3)
        self.assertEqual(short["normal_regions"], [])
        self.assertEqual(short["unknown_regions"], [{"start_s": 0, "end_s": .24, "label": "unknown"}])

    def test_many_events_never_collapse_over_normal(self):
        result = self.segment(([0] * 5 + [5]) * 8 + [0] * 5)
        self.assertEqual(len(result["events"]), 8)
        self.assertGreater(len(result["normal_regions"]), 5)


@unittest.skipIf(torch is None, "PyTorch 不可用：跳过仅前向的合成网络检查")
class PannsForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.checkpoint = Path(cls.tmp.name) / "SYNTHETIC_NOT_PRETRAINED.pth"
        state = {}
        for name, module in (("bn0", torch.nn.BatchNorm2d(64)),
                             ("conv_block1", ConvBlock(1, 64)),
                             ("conv_block2", ConvBlock(64, 128)),
                             ("conv_block3", ConvBlock(128, 256))):
            state.update({f"{name}.{k}": v for k, v in module.state_dict().items()})
        # Synthetic Fourier-shaped frontend; no real AudioSet weights are needed.
        n = torch.arange(1024)
        phase = 2 * torch.pi * torch.arange(513)[:, None] * n[None] / 1024
        state["spectrogram_extractor.stft.conv_real.weight"] = (torch.cos(phase) * torch.hann_window(1024))[:, None]
        state["spectrogram_extractor.stft.conv_imag.weight"] = (-torch.sin(phase) * torch.hann_window(1024))[:, None]
        state["logmel_extractor.melW"] = torch.rand(513, 64) / 513
        torch.save({"model": state}, cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        # Fail the test immediately if any training or threshold fitting is introduced.
        self.no_backward = patch.object(torch.Tensor, "backward", side_effect=AssertionError("NO TRAINING"))
        self.no_optimizer = patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("NO TRAINING"))
        self.no_quantile = patch("numpy.quantile", side_effect=AssertionError("NO CALIBRATION"))
        for p in (self.no_backward, self.no_optimizer, self.no_quantile):
            p.start()
            self.addCleanup(p.stop)
        self.cfg = NetworkConfig(core_frames=16)
        self.encoder = FrozenPanns(self.checkpoint, self.cfg)

    def model(self):
        meta = fixture_metadata(self.cfg)
        meta.update(panns_checkpoint=str(self.checkpoint), panns_sha256=self.encoder.sha256)
        return Model(self.encoder, TemporalDAE(self.cfg), self.cfg, meta["normalization_scale"], meta)

    def test_bn_and_all_encoder_params_frozen_even_parent_train(self):
        before = {k: v.clone() for k, v in self.encoder.state_dict().items()}
        parent = torch.nn.Sequential(self.encoder).train()
        with torch.no_grad():
            z = parent(torch.randn(1, self.cfg.samples_for_frames(30)))
        self.assertEqual(z.shape, (1, 448, 30))
        self.assertFalse(z.requires_grad)
        self.assertFalse(self.encoder.training)
        self.assertTrue(all(not p.requires_grad for p in self.encoder.parameters()))
        for k, v in self.encoder.state_dict().items():
            self.assertTrue(torch.equal(before[k], v), k)

    def test_constant_new_timbres_map_to_zero_dynamic_and_zero_reconstruction(self):
        with torch.no_grad():
            for _ in range(3):
                x = torch.randn(1, 448, 1).repeat(1, 1, 50)
                d = dynamics(x, self.cfg)
                self.assertLess(float(d.abs().max()), 1e-6)
            head = TemporalDAE(self.cfg)
            self.assertEqual(float(head(torch.zeros(1, self.cfg.input_dim, 50)).abs().max()), 0)
            a = dynamics(torch.cat([torch.zeros(1, 448, 25), torch.ones(1, 448, 25)], -1), self.cfg)
            self.assertEqual(float(a[:, :, 35:].abs().max()), 0)

    def test_chunked_forward_matches_whole_only_on_true_support(self):
        model = self.model()
        wave = np.random.default_rng(2).normal(0, .02, 32000 * 6).astype(np.float32)
        times, scores, valid = model.score_wave(wave)
        with torch.no_grad():
            target = dynamics(self.encoder(torch.from_numpy(wave)[None]), self.cfg) / model.scales
            full = (model.head(target) - target).square().mean(1).sqrt()[0].numpy()
        np.testing.assert_allclose(scores[valid], full[valid], atol=2e-6, rtol=2e-4)
        self.assertTrue(valid.any())
        self.assertTrue(np.isnan(scores[~valid]).all())
        self.assertTrue((times[valid] >= self.cfg.context_radius_s).all())

    def test_unknown_context_cannot_affect_a_supported_normal_score(self):
        model = self.model()
        wave = np.random.default_rng(12).normal(0, .01, 32000 * 7).astype(np.float32)
        changed = wave.copy()
        changed[32000 * 5:] = np.random.default_rng(13).normal(0, .5, 32000 * 2)
        times, original, valid = model.score_wave(wave)
        _, after, _ = model.score_wave(changed)
        safe = valid & (times + model.cfg.context_radius_s < 5)
        np.testing.assert_allclose(original[safe], after[safe], atol=2e-6, rtol=2e-4)

    def test_constant_gain_is_removed_by_recorded_preprocessing(self):
        wave = torch.from_numpy(np.random.default_rng(14).normal(
            0, .01, self.cfg.samples_for_frames(40)).astype(np.float32))[None]
        with torch.no_grad():
            a, b = self.encoder(wave), self.encoder(wave * 3)
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-6)

    def test_l2_consumes_new_event_fields_without_interface_changes(self):
        from wind_audio.l1_panns.cli import write_detection
        from wind_audio.l2.cli import run_events
        from wind_audio.l2.encoder import EncoderConfig
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audio = write_wav(root / "input.wav", np.zeros(32000 * 4), 32000)
            model = self.model()
            times = model.cfg.offset_s + np.arange(49) * model.cfg.step_s
            scores = np.zeros(49)
            scores[25] = 1  # predetermined synthetic residual, not a fitted threshold
            valid = (times >= model.cfg.context_radius_s) & (times <= 4 - model.cfg.context_radius_s)
            with patch.object(model, "score_wave", return_value=(times, scores, valid)):
                payload = write_detection(model, audio, root / "detect")
            self.assertEqual(len(payload["events"]), 1)
            event = payload["events"][0]
            self.assertTrue(event["event_id"])
            config = EncoderConfig()
            with patch("wind_audio.l2.cli.Model.load") as mdl, patch("wind_audio.l2.cli.PannsEncoder") as enc:
                mdl.return_value.meta = {"version": "fixture", "encoder": config.to_dict()}
                mdl.return_value.predict.return_value = [{"status": "unknown"}]
                enc.return_value.config = config
                enc.return_value.encode.return_value = np.ones(2048)
                result = run_events(root / "detect/events.json", "unused", "unused", root / "l2.json")
                self.assertEqual(enc.return_value.encode.call_args.args[1:], (event["start_s"], event["end_s"]))
                self.assertEqual(result["events"][0]["event_id"], event["event_id"])

    def test_save_load_hash_status_and_short_audio_contract(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = self.model()
            model.save(root / "fixture")
            restored = Model.load(root / "fixture")
            for k, v in model.head.state_dict().items():
                self.assertTrue(torch.equal(v, restored.head.state_dict()[k]))
            with self.assertRaises(ValueError):
                model.save(root / "fixture")
            audio = root / "short.wav"
            write_wav(audio, np.zeros(800), 32000)
            payload = restored.predict(audio)
            self.assertTrue(payload["diagnostics"]["context_insufficient"])
            self.assertEqual(payload["normal_regions"], [])
            self.assertEqual(payload["events"], [])
            self.assertEqual(payload["unknown_regions"][0]["end_s"], .025)
            restored.meta["calibration"] = None
            with self.assertRaisesRegex(ValueError, "未正常校准"):
                restored.predict(audio)
            restored.meta["trained"] = False
            with self.assertRaisesRegex(ValueError, "未训练"):
                restored.score_wave(np.zeros(32000, np.float32))
            with (root / "fixture/temporal.pt").open("ab") as f:
                f.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "哈希"):
                Model.load(root / "fixture")
            with self.assertRaisesRegex(ValueError, "哈希"):
                FrozenPanns(self.checkpoint, expected_sha="invalid")

    def test_detect_evaluate_share_prediction_and_never_call_legacy(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audio = root / "synthetic.wav"
            write_wav(audio, np.random.default_rng(3).normal(0, .01, 32000 * 3), 32000)
            model = self.model()
            model.save(root / "fixture")
            with patch("wind_audio.cli.extract_features", side_effect=AssertionError("OLD L1")), \
                    patch("wind_audio.cli.read_annotations", side_effect=AssertionError("LABEL READ")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["l1-detect", "--input", str(audio), "--config",
                                       str(root / "fixture/model.json"), "--out", str(root / "detect")]), 0)
            detected = json.loads((root / "detect/events.json").read_text())
            self.assertTrue((root / "detect/review.csv").is_file())
            record = {"audio_id": "s", "audio_path": str(audio), "split": "test", "source_id": "test",
                      "audio_sha256": "a", "pcm_sha256": "b",
                      "intervals": [{"start_s": 0, "end_s": 3, "label": "normal"}]}
            first = evaluate(model, {"records": [record]})
            record["intervals"][0]["label"] = "anomaly"
            second = evaluate(model, {"records": [record]})
            self.assertEqual(first["per_file"][0]["prediction"], detected)
            self.assertEqual(second["per_file"][0]["prediction"], detected)
            self.assertGreater(first["duration_metrics"]["abstained_reviewed_s"], 0)


class PannsCLITests(unittest.TestCase):
    def test_detect_rejects_labels_and_override_threshold(self):
        for extra in (["--labels", "truth.csv"], ["--threshold", "1"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                build_parser().parse_args(["l1-panns-detect", "--model", "m", "--input", "a.wav",
                                           "--out", "o", *extra])

    def test_new_commands_parse_without_running_training_or_calibration(self):
        for command in ("l1-panns-train", "l1-panns-calibrate", "l1-panns-evaluate"):
            args = [command, "--dataset", "/PLACEHOLDER/data.json", "--out", "/PLACEHOLDER/new"]
            args += ["--checkpoint" if command.endswith("train") else "--model", "/PLACEHOLDER/model"]
            self.assertEqual(build_parser().parse_args(args).cmd, command)

    def test_legacy_detection_route_rejects_overrides_before_loading_model(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.json"
            path.write_text(json.dumps({"schema_version": 4, "model_type": MODEL_TYPE}))
            for flag in ("--threshold", "--merge-gap", "--min-dur", "--sr", "--padding"):
                with contextlib.redirect_stderr(io.StringIO()) as output:
                    result = main(["l1-detect", "--input", "unused.wav", "--config", str(path),
                                   "--out", str(Path(td) / "new"), flag, "1"])
                self.assertEqual(result, 2)
                self.assertIn("禁止覆盖", output.getvalue())

    def test_untrained_uncalibrated_fail_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.json"
            obj = {"model_type": MODEL_TYPE, "schema_version": 4, "trained": False,
                   "network_config": NetworkConfig().to_dict(), "timing": NetworkConfig().timing()}
            if torch is None:
                return
            obj["preprocessing"] = PREPROCESSING
            path.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError, "未训练"):
                Model.load(path)
            obj.update(trained=True, training={"optimizer_steps": 1})
            path.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError, "未正常校准"):
                Model.load(path)
