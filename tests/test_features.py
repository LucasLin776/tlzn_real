import unittest

import numpy as np

from wind_audio.features import FeatureConfig, extract_features

CFG = FeatureConfig(
    sr=8000, n_fft=512, hop=128, n_mels=16, fmin=20, fmax=3000,
    target_step=0.2, context_s=0.4, scales=(0.2, 0.4), normalize="per_file",
)


class FeatureTests(unittest.TestCase):
    def test_feature_names_match_matrix(self):
        rng = np.random.default_rng(0)
        x = 0.01 * rng.standard_normal(8000 * 2).astype(np.float32)
        bundle = extract_features(x, 8000, CFG, "x")
        self.assertEqual(bundle.X.shape[1], len(bundle.feature_names))
        self.assertEqual(bundle.X.shape[0], len(bundle.times))
        self.assertLess(bundle.times[-1], bundle.duration_s)
        self.assertFalse(np.isnan(bundle.X).any())

    def test_fixed_gain_is_removed(self):
        rng = np.random.default_rng(1)
        x = 0.01 * rng.standard_normal(8000 * 2).astype(np.float32)
        a = extract_features(x, 8000, CFG, "a")
        b = extract_features(0.4 * x, 8000, CFG, "b")
        np.testing.assert_allclose(a.mel_norm, b.mel_norm, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(a.X, b.X, rtol=1e-4, atol=1e-3)

    def test_short_and_silent_audio(self):
        for x in (np.zeros(1600, dtype=np.float32), 0.01 * np.ones(1600, dtype=np.float32)):
            bundle = extract_features(x, 8000, CFG, "short")
            self.assertGreaterEqual(len(bundle.times), 1)
            self.assertTrue(np.isfinite(bundle.X).all())

    def test_none_normalisation_keeps_gain_difference(self):
        cfg = FeatureConfig(
            sr=8000, n_fft=512, hop=128, n_mels=16, fmin=20, fmax=3000,
            target_step=0.2, context_s=0.4, scales=(0.2, 0.4), normalize="none",
        )
        rng = np.random.default_rng(2)
        x = 0.01 * rng.standard_normal(8000).astype(np.float32)
        a = extract_features(x, 8000, cfg, "a")
        b = extract_features(0.4 * x, 8000, cfg, "b")
        self.assertGreater(float(np.abs(a.X - b.X).max()), 0.1)


if __name__ == "__main__":
    unittest.main()
