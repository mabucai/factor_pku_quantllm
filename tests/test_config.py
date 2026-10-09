import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import research_config


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        public = Path(research_config.__file__).parent / "research_settings.json"
        (self.root / "research_settings.json").write_bytes(public.read_bytes())
        self.defaults = json.loads(public.read_text(encoding="utf-8"))
        patcher = patch.object(research_config, "BASE_DIR", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def write_local(self, data):
        (self.root / "config.json").write_text(json.dumps(data), encoding="utf-8")

    def test_missing_or_blank_path_fails_with_setup_instructions(self):
        for data in ({}, {"lab_path": ""}, {"lab_path": None}):
            self.write_local(data)
            with self.assertRaisesRegex(ValueError, "config.example.json"):
                research_config.load_config()

    def test_relative_path_and_default_cache_preserve_research_parameters(self):
        self.write_local({"lab_path": "data/lab/csi300", "cache_dir": ""})
        config = research_config.load_config()
        self.assertEqual(config["lab_path"], str((self.root / "data/lab/csi300").resolve()))
        self.assertEqual(config["cache_dir"], str((self.root / "runs/cache").resolve()))
        for key, value in self.defaults.items():
            self.assertEqual(config[key], value)

    def test_environment_overrides_local_paths(self):
        self.write_local({"lab_path": "old", "cache_dir": "old_cache"})
        with patch.dict(os.environ, {"FACTOR_LAB_PATH": str(self.root / "new"), "FACTOR_CACHE_DIR": str(self.root / "new_cache")}):
            config = research_config.load_config()
        self.assertEqual(config["lab_path"], str((self.root / "new").resolve()))
        self.assertEqual(config["cache_dir"], str((self.root / "new_cache").resolve()))

    def test_environment_only_does_not_require_local_file(self):
        with patch.dict(os.environ, {"FACTOR_LAB_PATH": "data/lab"}):
            config = research_config.load_config()
        self.assertEqual(config["lab_path"], str((self.root / "data/lab").resolve()))

    def test_old_config_can_override_public_parameters(self):
        self.write_local({"lab_path": "data/lab", "workers": 1})
        self.assertEqual(research_config.load_config()["workers"], 1)

    def test_malformed_config_error_does_not_echo_contents(self):
        marker = "private_" + "test_marker"
        (self.root / "config.json").write_text(marker, encoding="utf-8")
        with self.assertRaises(ValueError) as result:
            research_config.load_config()
        self.assertNotIn(marker, str(result.exception))


if __name__ == "__main__":
    unittest.main()
