"""
Unit tests for AgeWeaver pipeline transformation classification,
adaptive parameter scaling across progression, regression, and reconstruction,
and age bracket handling.
"""

import unittest
import os
import sys

sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("."))

from ageweaver.pipeline import AgeWeaverPipeline, classify_transformation
from ageweaver.config import PipelineConfig
from ageweaver.prompts import get_bracket_info, build_structured_prompt

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.config = PipelineConfig(
            use_adaptive_control=True,
            adaptive_strength_min=0.32,
            adaptive_strength_max=0.58,
            guidance_min=6.5,
            guidance_max=8.5
        )
        self.pipeline = AgeWeaverPipeline(config=self.config)

    def test_classify_transformation_directions(self):
        self.assertEqual(classify_transformation(25, 60), "progression")
        self.assertEqual(classify_transformation(60, 25), "regression")
        self.assertEqual(classify_transformation(28, 28), "reconstruction")
        self.assertEqual(classify_transformation(None, 50), "unconditioned")

    def test_compute_adaptive_parameters_progression(self):
        # Delta = 70 - 28 = 42 years
        strength, guidance = self.pipeline.compute_adaptive_parameters(source_age=28, target_age=70)
        self.assertGreater(strength, 0.32)
        self.assertLessEqual(strength, 0.58)
        self.assertGreater(guidance, 6.5)
        self.assertLessEqual(guidance, 8.5)

    def test_compute_adaptive_parameters_regression(self):
        # Delta = 65 - 25 = 40 years
        strength, guidance = self.pipeline.compute_adaptive_parameters(source_age=65, target_age=25)
        self.assertGreater(strength, 0.32)
        self.assertLessEqual(strength, 0.58)
        self.assertGreater(guidance, 6.5)
        self.assertLessEqual(guidance, 8.5)

    def test_compute_adaptive_parameters_reconstruction_equal_age(self):
        # Delta = 0 years -> should return exact minimum baseline
        strength, guidance = self.pipeline.compute_adaptive_parameters(source_age=30, target_age=30)
        self.assertEqual(strength, 0.32)
        self.assertEqual(guidance, 6.5)

    def test_adaptive_parameters_bidirectional_symmetry(self):
        # Forward aging 30 -> 60 (delta = 30) vs Rejuvenation 60 -> 30 (delta = 30)
        str_fwd, guid_fwd = self.pipeline.compute_adaptive_parameters(source_age=30, target_age=60)
        str_rev, guid_rev = self.pipeline.compute_adaptive_parameters(source_age=60, target_age=30)
        self.assertEqual(str_fwd, str_rev)
        self.assertEqual(guid_fwd, guid_rev)

    def test_adaptive_parameters_max_delta_clamping(self):
        # Delta = 50 vs Delta = 75 (both >= 50 years max scale)
        str_50, guid_50 = self.pipeline.compute_adaptive_parameters(source_age=20, target_age=70)
        str_75, guid_75 = self.pipeline.compute_adaptive_parameters(source_age=20, target_age=95)
        self.assertEqual(str_50, 0.58)
        self.assertEqual(str_75, 0.58)
        self.assertEqual(guid_50, 8.5)
        self.assertEqual(guid_75, 8.5)

    def test_childhood_and_regression_prompts(self):
        # Age 2 -> toddler
        bracket_2, skin_2, hair_2 = get_bracket_info(2)
        self.assertEqual(bracket_2, "toddler")
        self.assertIn("smooth", skin_2)

        # Age 12 -> child
        bracket_12, skin_12, hair_12 = get_bracket_info(12)
        self.assertEqual(bracket_12, "child")
        self.assertIn("smooth", skin_12)

        # Structured prompt for regression: e.g. 50-year-old regressing to 25
        prompt_reg = build_structured_prompt(target_age=25, source_age=50, gender_hint="man")
        self.assertIn("aged 25 years old", prompt_reg)
        self.assertIn("young adult", prompt_reg)
        self.assertIn("identical facial bone structure", prompt_reg)

if __name__ == "__main__":
    unittest.main()

