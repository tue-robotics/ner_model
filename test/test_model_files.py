import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# These tests import torch and transformers. catkin runs the tests with the system nosetests, which does not see
# packages installed in the tue-env virtualenv, so skip them instead of failing when torch can't be imported.
HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(HAS_TORCH, "torch and transformers are not importable")
class TestImport(unittest.TestCase):
    def test_import(self):
        """
        If no exception is raised, this test will succeed
        """
        from ner_model.parser import NERParser  # noqa: F401


@unittest.skipUnless(HAS_TORCH, "torch and transformers are not importable")
class TestFindDataFile(unittest.TestCase):
    def setUp(self):
        from ner_model import ner_model

        self.ner_model = ner_model
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.data_dir = Path(self.tmp_dir.name)

    def test_vocab_ships_with_package(self):
        self.assertTrue(self.ner_model.find_data_file(self.ner_model.VOCAB_FILENAME).is_file())

    def test_explicit_path_to_file_or_directory(self):
        weights = self.data_dir / "model.pth"
        weights.touch()
        self.assertEqual(self.ner_model.find_data_file("model.pth", explicit_path=str(weights)), weights)
        self.assertEqual(self.ner_model.find_data_file("model.pth", explicit_path=str(self.data_dir)), weights)

    def test_explicit_path_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.ner_model.find_data_file("model.pth", explicit_path=str(self.data_dir))

    def test_env_var_dir_is_searched_first(self):
        weights = self.data_dir / "model.pth"
        weights.touch()
        with mock.patch.dict(os.environ, {self.ner_model.DATA_DIR_ENV_VAR: str(self.data_dir)}):
            self.assertEqual(self.ner_model.find_data_file("model.pth"), weights)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.ner_model.find_data_file("does_not_exist.pth")


if __name__ == "__main__":
    unittest.main()
