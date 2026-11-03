"""Optional real XGBoost fit/serialize/predict check, exercised by its CI job."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wind_audio.backends import XGBBackend


@unittest.skipUnless(importlib.util.find_spec("xgboost") is not None, "xgboost not installed")
class XGBoostTests(unittest.TestCase):
    def test_fit_save_load_consistency(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(80, 4)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int8)
        backend = XGBBackend(seed=7, n_estimators=5, max_depth=2)
        backend.fit(X, y, sample_weight=np.ones(len(y)))
        before = backend.predict_proba(X)
        self.assertEqual(before.shape, (80, 2))
        self.assertTrue(np.isfinite(before).all())
        with tempfile.TemporaryDirectory() as td:
            backend.save(Path(td))
            restored = XGBBackend.load(Path(td))
            np.testing.assert_allclose(before, restored.predict_proba(X), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
