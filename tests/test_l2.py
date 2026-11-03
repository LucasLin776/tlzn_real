"""Small regression suite; all generated fixtures live in temporary directories."""
import dataclasses
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from wind_audio import cli
from wind_audio.audio import write_wav
from wind_audio.data import AudioRecord, write_manifest
from wind_audio.review import ReviewRecord, read_reviews, import_reviews, write_reviews
from wind_audio.splits import SplitSet, save_splits
from wind_audio.l2.common import file_hash
from wind_audio.l2.data import Dataset, Sample, assert_continuation, prepare
from wind_audio.l2.encoder import EncoderConfig, Features, PannsEncoder, windows
from wind_audio.l2.model import Model, evaluate, train


def fixture(extra=False):
    classes = [{"id": "impact", "name": "冲击"}]
    if extra:
        classes.append({"id": "rub", "name": "摩擦"})
    rows, vectors = [], []
    for split in ("train", "val", "test"):
        for label in (0, 1):
            for n in range(2):
                sid = f"{split}_{label}_{n}"
                rows.append(Sample(sid, "/unused.wav", hashlib.sha256(sid.encode()).hexdigest(),
                                   0, 1, sid, split, label,
                                   ["impact"] if label else [], [] if label else ["impact"]))
                v = np.zeros(2048, dtype=np.float32)
                v[label] = 1
                vectors.append(v)
    if extra:
        for n in range(2):
            sid = f"new_{n}"
            rows.append(Sample(sid, "/unused.wav", hashlib.sha256(sid.encode()).hexdigest(),
                               0, 1, sid, "train", 1, ["rub"], ["impact"]))
            v = np.zeros(2048, dtype=np.float32)
            v[2] = 1
            vectors.append(v)
    ds = Dataset(classes, rows)
    fs = Features([r.sample_id for r in rows], [r.content_key() for r in rows],
                  np.stack(vectors), EncoderConfig(checkpoint_sha256="0" * 64).to_dict())
    return ds, fs


class ContinualTests(unittest.TestCase):
    def test_mask_and_incremental_new_class(self):
        ds, fs = fixture()
        y, mask = ds.targets()
        self.assertEqual(mask[:, 0].sum(), len(ds.samples))
        model = train(ds, fs, version="v1", epochs=60)
        self.assertTrue(model.meta["calibration"][0]["active"])
        self.assertTrue(model.meta["calibration"][1]["active"])
        result = evaluate(model, ds, fs)
        self.assertEqual(result["macro_f1"], 1)
        ds2, fs2 = fixture(extra=True)
        y2, m2 = ds2.targets()
        self.assertTrue(np.all(m2[:len(ds.samples), 2] == 0))
        revised = train(ds2, fs2, version="v2", parent=model, epochs=40)
        self.assertEqual(revised.W.shape, (2048, 3))
        self.assertEqual(revised.meta["training"]["new_or_revised"], 2)
        self.assertFalse(revised.meta["calibration"][2]["active"])
        self.assertTrue(all(r["group_id"].startswith(("train", "new"))
                            for r in revised.meta["memory"]))
        self.assertTrue(all(p["keep_l1_alarm"] for p in revised.predict(fs2.X)))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model"
            revised.save(path)
            loaded = Model.load(path)
            np.testing.assert_allclose(loaded.W, revised.W)
            self.assertEqual(loaded.predict(fs2.X), revised.predict(fs2.X))
            with self.assertRaises(FileExistsError):
                revised.save(path)

    def test_no_replay_and_full_retrain_controls(self):
        ds, fs = fixture()
        base = train(ds, fs, version="v1", epochs=2)
        ds2, fs2 = fixture(extra=True)
        naive = train(ds2, fs2, version="naive", parent=base, epochs=2, replay=False, distill=0)
        full = train(ds2, fs2, version="full", parent=base, epochs=2, full_retrain=True)
        self.assertEqual(naive.meta["training"]["selected"], 2)
        self.assertEqual(full.meta["training"]["selected"], 6)

    def test_split_and_revision_guards(self):
        ds, fs = fixture()
        base = train(ds, fs, version="v1", epochs=2)
        rows = list(ds.samples)
        rows[0] = dataclasses.replace(rows[0], positive_types=["impact"], anomaly=1, negative_types=[])
        with self.assertRaisesRegex(ValueError, "revision"):
            assert_continuation(base.meta, Dataset(ds.classes, rows))
        rows[0] = dataclasses.replace(rows[0], review_revision=2)
        assert_continuation(base.meta, Dataset(ds.classes, rows))
        rows = list(ds.samples)
        rows[-1] = dataclasses.replace(rows[-1], group_id=rows[0].group_id)
        with self.assertRaisesRegex(ValueError, "跨"):
            Dataset(ds.classes, rows)
        with self.assertRaisesRegex(ValueError, "历史样本"):
            assert_continuation(base.meta, Dataset(ds.classes, ds.samples[1:]))

    def test_mismatched_embedding_rejected(self):
        ds, fs = fixture()
        broken = Features(fs.ids, ["wrong"] * len(fs.ids), fs.X, fs.encoder)
        with self.assertRaisesRegex(ValueError, "重新提取"):
            train(ds, broken, version="v1", epochs=2)

    def test_not_enough_labels_stays_unknown(self):
        ds, fs = fixture()
        rows = [dataclasses.replace(r, positive_types=[], negative_types=[]) for r in ds.samples]
        model = train(Dataset(ds.classes, rows), fs, version="v1", epochs=2)
        self.assertFalse(model.meta["calibration"][1]["active"])
        self.assertEqual(model.predict(fs.X[2:3])[0]["status"], "unknown")

    def test_windows_keep_context_and_pad(self):
        cfg = EncoderConfig(window_seconds=1, stride_seconds=0.5)
        x = np.ones(64000, np.float32)
        first = list(windows(x, 0, 0.1, cfg))[0]
        self.assertEqual(first.shape, (32000,))
        self.assertEqual(first[0], 0)
        self.assertEqual(first[-1], 1)
        self.assertEqual(len(list(windows(x, 0, 2, cfg))), 3)
        with self.assertRaises(ValueError):
            list(windows(x, 0, 3, cfg))

    def test_review_roundtrip_revision(self):
        with tempfile.TemporaryDirectory() as td:
            incoming, master = Path(td) / "in.csv", Path(td) / "reviews.csv"
            r = ReviewRecord("event", "audio", predicted_end_s=1, l1_label="anomaly",
                             positive_types="impact", types_complete=True)
            write_reviews([r], incoming)
            import_reviews(incoming, master)
            self.assertTrue(read_reviews(master)[0].types_complete)
            r.positive_types = "rub"
            write_reviews([r], incoming)
            import_reviews(incoming, master)
            self.assertEqual(read_reviews(master)[0].review_revision, 2)

    def test_prepare_does_not_label_new_types_negative(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wave = root / "a.wav"
            write_wav(wave, np.zeros(32000, np.float32), 32000)
            record = AudioRecord("a", "a.wav", source_id="source", duration_s=1)
            write_manifest([record], root / "manifest.csv")
            save_splits(SplitSet(["a"], ["val"], ["test"]), root / "splits.json")
            r = ReviewRecord("e", "a", predicted_end_s=1, l1_label="anomaly",
                             positive_types="impact", types_complete=True, reviewed_at="2026-11-03")
            write_reviews([r], root / "reviews.csv")
            classes = [{"id": "impact", "name": "冲击"}]
            with patch("wind_audio.l2.data.taxonomy", return_value=classes):
                before = prepare(root / "reviews.csv", root / "manifest.csv",
                                 root / "splits.json", "unused")
            with patch("wind_audio.l2.data.taxonomy", return_value=classes + [{"id": "rub", "name": "摩擦"}]):
                after = prepare(root / "reviews.csv", root / "manifest.csv",
                                root / "splits.json", "unused", previous=before)
            self.assertEqual(after.samples[0].negative_types, [])
            self.assertEqual(after.targets()[1][0, 2], 0)
            before.save(root / "snapshot.json")
            self.assertEqual(Dataset.load(root / "snapshot.json").fingerprint, before.fingerprint)

    def test_cli_train_evaluate_and_event_inference(self):
        from wind_audio.l2.cli import run_events
        from wind_audio.l2.common import write_json
        ds, fs = fixture()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i, r in enumerate(ds.samples):
                path = root / (r.sample_id + ".wav")
                write_wav(path, np.random.default_rng(i).normal(0, 0.02, 32000), 32000)
                r.audio_path, r.audio_sha256 = str(path), file_hash(path)
            ds.save(root / "data.json")
            fs = Features(fs.ids, [r.content_key() for r in ds.samples], fs.X, fs.encoder)
            fs.save(root / "features.npz")
            common = ["--dataset", str(root / "data.json"), "--features", str(root / "features.npz")]
            self.assertEqual(cli.main(["l2-train", *common, "--version", "v1",
                                      "--out", str(root / "model"), "--epochs", "50"]), 0)
            self.assertEqual(cli.main(["l2-evaluate", *common, "--model", str(root / "model"),
                                      "--out", str(root / "metrics.json")]), 0)
            row = ds.samples[-1]
            write_json(root / "events.json", {"audio": {"path": row.audio_path, "sha256": row.audio_sha256},
                                             "events": [{"event_id": "e", "start_s": 0, "end_s": 1}]})
            with patch("wind_audio.l2.cli.PannsEncoder") as enc:
                enc.return_value.config = EncoderConfig()
                enc.return_value.encode.return_value = fs.X[-1]
                payload = run_events(root / "events.json", root / "model", "local-only",
                                     root / "result.json")
            self.assertEqual(payload["events"][0]["l2"]["status"], "known_anomaly")
            self.assertTrue(payload["events"][0]["l2"]["keep_l1_alarm"])

    def test_demo_entry_removed(self):
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.build_parser().parse_args(["run", "--input", "x.wav", "--demo"])


@unittest.skipUnless(os.environ.get("WIND_AUDIO_TEST_PANNS") == "1",
                     "显式启用 CNN14 大模型 CPU 结构测试")
class PannsStructureTests(unittest.TestCase):
    def test_official_layout_and_frontend_without_downloading_weights(self):
        import torch
        torch.set_num_threads(2)
        state = {}
        n = torch.arange(1024, dtype=torch.float32)
        phase = 2 * torch.pi * torch.arange(513)[:, None] * n / 1024
        window = torch.hann_window(1024)
        state["spectrogram_extractor.stft.conv_real.weight"] = (phase.cos() * window)[:, None, :]
        state["spectrogram_extractor.stft.conv_imag.weight"] = (-phase.sin() * window)[:, None, :]
        state["logmel_extractor.melW"] = torch.ones(513, 64) / 513

        def bn(name, size):
            for key, value in torch.nn.BatchNorm2d(size).state_dict().items():
                state[name + "." + key] = value
        bn("bn0", 64)
        channels = (1, 64, 128, 256, 512, 1024, 2048)
        for i in range(6):
            a, b = channels[i:i + 2]
            state[f"conv_block{i + 1}.conv1.weight"] = torch.zeros(b, a, 3, 3)
            state[f"conv_block{i + 1}.conv2.weight"] = torch.zeros(b, b, 3, 3)
            bn(f"conv_block{i + 1}.bn1", b)
            bn(f"conv_block{i + 1}.bn2", b)
        state["fc1.weight"], state["fc1.bias"] = torch.zeros(2048, 2048), torch.ones(2048)
        state["fc_audioset.weight"], state["fc_audioset.bias"] = torch.zeros(527, 2048), torch.zeros(527)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cnn14.pth"
            torch.save({"model": state}, path)
            state.clear()
            encoder = PannsEncoder(path, config=EncoderConfig(window_seconds=1, stride_seconds=0.5))
            captured = []
            hook = encoder.model.bn0.register_forward_pre_hook(lambda _, args: captured.append(args[0].clone()))
            waveform = np.random.default_rng(0).normal(size=32000).astype(np.float32)
            vector = encoder.encode(waveform, 0, 1)
            hook.remove()
            self.assertEqual(vector.shape, (2048,))
            self.assertAlmostEqual(float(np.linalg.norm(vector)), 1, places=5)
            reference = torch.stft(torch.from_numpy(waveform)[None], n_fft=1024,
                                   hop_length=320, window=window, return_complex=True).abs().square()
            reference = 10 * torch.log10(torch.clamp(reference.transpose(1, 2).mean(-1), min=1e-10))
            torch.testing.assert_close(captured[0][:, 0, :, 0], reference, rtol=2e-4, atol=2e-3)
            self.assertFalse(any(p.requires_grad for p in encoder.model.parameters()))
            with self.assertRaisesRegex(ValueError, "校验和"):
                PannsEncoder(path, config=EncoderConfig(checkpoint_sha256="f" * 64))
