import unittest
import numpy as np
from wind_audio.features import FeatureConfig, extract_features


class FeatureTests(unittest.TestCase):
    def test_features_are_unfitted_spectra(self):
        cfg = FeatureConfig(sr=8000, n_fft=512, hop=80, n_mels=32, fmax=3500)
        bundle = extract_features(np.ones(16000) * 0.01, 8000, cfg, "a")
        self.assertEqual(bundle.mel_power.shape[1], 32)
        self.assertEqual(len(bundle.times), 100)
        self.assertLess(bundle.times[-1], bundle.duration_s)
        self.assertTrue(np.isfinite(bundle.spec_db).all())

    def test_empty_nonfinite_and_bad_config(self):
        for x in (np.array([]), np.array([np.nan])):
            with self.assertRaises(ValueError):
                extract_features(x, 44100, FeatureConfig(), "bad")
        for fields in ({"compare_s": 0}, {"threshold": 0}, {"target_step": 0.001},
                       {"smooth_s": float("nan")}, {"normalize": "per_file"}):
            with self.assertRaises(ValueError):
                FeatureConfig.from_dict(fields)
