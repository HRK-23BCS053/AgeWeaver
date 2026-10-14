"""
Unit tests for AgeWeaver prompt engineering and refinement module.
"""

import unittest
from ageweaver.prompts import (
    build_generic_prompt,
    build_structured_prompt,
    build_negative_prompt,
    get_bracket_info
)

class TestPrompts(unittest.TestCase):
    def test_generic_prompt(self):
        prompt_25 = build_generic_prompt(25)
        self.assertIn("25-year-old", prompt_25)
        self.assertIn("portrait photo", prompt_25)

        prompt_60_m = build_generic_prompt(60, gender_hint="male")
        self.assertIn("60-year-old man", prompt_60_m)

    def test_bracket_info(self):
        bracket, skin, hair = get_bracket_info(25)
        self.assertEqual(bracket, "young adult")
        self.assertIn("smooth", skin)

        bracket_68, skin_68, hair_68 = get_bracket_info(68)
        self.assertEqual(bracket_68, "senior")
        self.assertIn("wrinkles", skin_68)
        self.assertIn("silver", hair_68)

    def test_structured_prompt(self):
        prompt_senior = build_structured_prompt(
            target_age=70,
            source_age=25,
            gender_hint="man",
            include_hair=True,
            include_skin=True,
            include_identity_anchor=True
        )
        self.assertIn("70 years old", prompt_senior)
        self.assertIn("identical facial bone structure", prompt_senior)
        self.assertIn("pronounced wrinkles", prompt_senior)
        self.assertIn("silver-gray hair", prompt_senior)

    def test_negative_prompt(self):
        neg = build_negative_prompt()
        self.assertIn("deformed", neg)
        self.assertIn("plastic skin", neg)
        self.assertIn("different person", neg)

if __name__ == "__main__":
    unittest.main()

