import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wind_audio.audio import load_wav
from wind_audio.data import write_annotations, write_manifest
from wind_audio.backends import select_backend
from wind_audio.detectors import (
    BaselineDetector,
    SupervisedDetector,
    cap_training_data,
    load_detector,
)
from wind_audio.features import FeatureConfig, build_matrix, extract_features

from tests.helpers import ann, make_wav, record

CFG = FeatureConfig(
    sr=8000, n_fft=512, hop=128, n_mels=16, fmin=20, fmax=3000,
    target_step=0.2, context_s=0.4, scales=(0.2, 0.4), normalize="per_file",
)


def _tiny_dataset(td: Path):
    audio = td / "audio"
    audio.mkdir()
    make_wav(audio / "n1.wav", sr=8000, duration=6.0, seed=1)
    make_wav(audio / "n2.wav", sr=8000, duration=6.0, seed=2)
    make_wav(audio / "a1.wav", sr=8000, duration=6.0, seed=3, burst=(2.0, 2.0, 0.3))
    recs = [
        record("n1", "audio/n1.wav", 6.0, source="S1", device="D1"),
        record("n2", "audio/n2.wav", 6.0, source="S2", device="D2"),
        record("a1", "audio/a1.wav", 6.0, source="S3", device="D2"),
    ]
    anns = [ann("x1", "a1", 2.0, 4.0)]
    manifest = td / "manifest.csv"
    annotations = td / "annotations.csv"
    write_manifest(recs, manifest)
    write_annotations(anns, annotations)
    return recs, anns, manifest, annotations


class ModelTests(unittest.TestCase):
    def test_baseline_save_load_and_threshold_override(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            recs, anns, manifest, _ = _tiny_dataset(td)
            det = BaselineDetector.train(
                recs, anns, ["n1", "n2"], manifest, CFG,
                version="test_base", threshold=5.0, max_frames_per_file=2000,
            )
            x, sr = load_wav(td / "audio" / "n1.wav", target_sr=CFG.sr)
            bundle = extract_features(x, sr, CFG, "n1")
            before = det.score(bundle)
            det.save(td / "model")
            loaded = load_detector(td / "model")
            after = loaded.score(bundle)
            np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)

            low_thr = max(1e-3, float(np.min(before) - 1.0))
            low = loaded.predict_events(bundle, threshold=low_thr)
            high = loaded.predict_events(bundle, threshold=float(np.max(before) + 100.0))
            self.assertGreaterEqual(len(low), 1)
            self.assertEqual(high, [])
            self.assertAlmostEqual(loaded.threshold, 5.0)

    def test_run_report_and_versioned_event_id(self):
        from wind_audio.cli import _run_one
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            recs, anns, manifest, _ = _tiny_dataset(root)
            det = BaselineDetector.train(recs, anns, ["n1", "n2"], manifest, CFG, version="v1")
            args = dict(threshold=0.001, threshold_source="cli",
                        postprocess={"min_duration": 0.1, "merge_gap": 0.3, "low_ratio": 0.5},
                        postprocess_source="cli", fmax_plot=3000, clip_context=0.2,
                        records=recs, anns=anns, manifest_path=manifest)
            path = root / "audio/a1.wav"
            _run_one(path, det, CFG, root / "v1", **args)
            det.version = "v2"
            _run_one(path, det, CFG, root / "v2", **args)
            import json
            before = json.loads((root / "v1/events.json").read_text())
            after = json.loads((root / "v2/events.json").read_text())
            self.assertTrue(before["events"])
            self.assertNotEqual(before["events"][0]["event_id"], after["events"][0]["event_id"])
            self.assertTrue((root / "v1/report.html").exists())
            with self.assertRaisesRegex(ValueError, "审核记录"):
                _run_one(path, det, CFG, root / "v1", **args)

    def test_supervised_save_load_and_config_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            recs, anns, manifest, _ = _tiny_dataset(td)
            data = build_matrix(recs, anns, ["n1", "n2", "a1"], manifest, CFG, td / "cache")
            capped = cap_training_data(
                data["X"], data["y"], data["groups"], data["file_ids"], data["times"],
                max_total=10000, max_per_file=10000, seed=0,
            )
            det, sel = SupervisedDetector.train(
                capped["X"], capped["y"], data["feature_names"], CFG,
                version="test_sup", seed=0, algo="numpy",
                metadata={"file_ids": capped["file_ids"].tolist()},
            )
            self.assertEqual(sel.used, "numpy")
            x, sr = load_wav(td / "audio" / "n1.wav", target_sr=CFG.sr)
            bundle = extract_features(x, sr, CFG, "n1")
            before = det.score(bundle)
            det.save(td / "model")
            loaded = load_detector(td / "model")
            after = loaded.score(bundle)
            np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)

            bad_cfg = FeatureConfig(
                sr=8000, n_fft=512, hop=128, n_mels=32, fmin=20, fmax=3000,
                target_step=0.2, context_s=0.4, scales=(0.2, 0.4), normalize="per_file",
            )
            bad_bundle = extract_features(x, sr, bad_cfg, "n1")
            with self.assertRaises(ValueError):
                loaded.score(bad_bundle)


if __name__ == "__main__":
    unittest.main()
