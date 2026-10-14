"""
Unit tests for AgeWeaver configuration module.
"""

import unittest
import os
import tempfile
import yaml

from ageweaver.config import AgeWeaverConfig, PipelineConfig, PromptConfig, EvaluationConfig

class TestConfig(unittest.TestCase):
    def test_default_config_valid(self):
        cfg = AgeWeaverConfig()
        # Should not raise
        cfg.validate()
        self.assertEqual(cfg.project_name, "AgeWeaver")
        self.assertEqual(cfg.pipeline.image_size, 512)
        self.assertEqual(cfg.pipeline.device, "cpu")
        self.assertTrue(cfg.pipeline.local_files_only)

    def test_invalid_image_size(self):
        cfg = AgeWeaverConfig()
        cfg.pipeline.image_size = 513  # Not divisible by 8
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_invalid_strength(self):
        cfg = AgeWeaverConfig()
        cfg.pipeline.default_strength = 1.5  # Outside [0, 1]
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_invalid_prompt_mode(self):
        cfg = AgeWeaverConfig()
        cfg.prompt.mode = "invalid_mode"
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_yaml_roundtrip(self):
        cfg = AgeWeaverConfig()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cfg.to_yaml(tmp_path)
            loaded_cfg = AgeWeaverConfig.from_yaml(tmp_path)
            self.assertEqual(loaded_cfg.project_name, cfg.project_name)
            self.assertEqual(loaded_cfg.pipeline.default_strength, cfg.pipeline.default_strength)
            self.assertEqual(loaded_cfg.prompt.mode, cfg.prompt.mode)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == "__main__":
    unittest.main()

