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

    @unittest.skipIf(importlib.util.find_spec("xgboost") is not None, "xgboost 已安装，不测回退")
    def test_missing_xgboost_is_recorded_not_silent(self):
        sel = select_backend("xgboost", allow_fallback=True, seed=0)
        self.assertTrue(sel.fallback)
        self.assertIn(sel.used, {"sklearn", "numpy"})
        self.assertIn("xgboost", sel.reason)

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
            self.assertIn(sel.used, {"sklearn", "numpy"})
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
