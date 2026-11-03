import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_distribution.py"
SPEC = importlib.util.spec_from_file_location("check_distribution", SCRIPT)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class DistributionTests(unittest.TestCase):
    def test_rejects_recordings_models_secrets_and_unsafe_paths(self):
        for name in ("project/data/labels.csv", "project/models/meta.json", "a.WAV",
                     "x/model.pkl", "__pycache__/x.pyc", ".env.production",
                     "../escape", "/absolute", "data\\labels.csv"):
            with self.subTest(name=name):
                self.assertTrue(checker.forbidden_member(name))

    def test_allows_sources_docs_and_fictional_examples(self):
        for name in ("wind_audio/audio.py", "project/docs/MODEL.md",
                     "project/examples/labels.example.csv", ".env.example"):
            with self.subTest(name=name):
                self.assertFalse(checker.forbidden_member(name))

    def test_inspects_wheel_without_extracting(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "example.whl"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("wind_audio/audio.py", "")
                archive.writestr("data/secret.wav", "")
            self.assertEqual(checker.inspect_archive(path), ["data/secret.wav"])


if __name__ == "__main__":
    unittest.main()
