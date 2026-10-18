"""
Unit tests for AgeWeaver evaluation and metrics module.
"""

import unittest
import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("."))

from ageweaver.identity import FaceIdentityAnalyzer
from ageweaver.evaluation import EvaluationMetrics, AgeWeaverEvaluator

class TestMetrics(unittest.TestCase):
    def test_cosine_similarity_identical(self):
        v = np.random.randn(512).astype(np.float32)
        v = v / np.linalg.norm(v)
        sim = FaceIdentityAnalyzer.compute_cosine_similarity(v, v)
        self.assertIsNotNone(sim)
        self.assertAlmostEqual(sim, 1.0, places=4)

    def test_cosine_similarity_orthogonal(self):
        v1 = np.zeros(512, dtype=np.float32)
        v2 = np.zeros(512, dtype=np.float32)
        v1[0] = 1.0
        v2[1] = 1.0
        sim = FaceIdentityAnalyzer.compute_cosine_similarity(v1, v2)
        self.assertIsNotNone(sim)
        self.assertAlmostEqual(sim, 0.0, places=4)

    def test_cosine_similarity_opposite(self):
        v1 = np.random.randn(512).astype(np.float32)
        v1 = v1 / np.linalg.norm(v1)
        v2 = -v1
        sim = FaceIdentityAnalyzer.compute_cosine_similarity(v1, v2)
        self.assertIsNotNone(sim)
        self.assertAlmostEqual(sim, -1.0, places=4)

    def test_cosine_similarity_missing_embedding(self):
        v = np.random.randn(512).astype(np.float32)
        self.assertIsNone(FaceIdentityAnalyzer.compute_cosine_similarity(v, None))
        self.assertIsNone(FaceIdentityAnalyzer.compute_cosine_similarity(None, v))
        self.assertIsNone(FaceIdentityAnalyzer.compute_cosine_similarity(None, None))

    def test_metrics_serialization(self):
        metrics = EvaluationMetrics(
            source_face_detected=True,
            generated_face_detected=True,
            identity_similarity=0.825,
            source_estimated_age=25.0,
            generated_estimated_age=68.5,
            target_age=70,
            age_mae=1.5,
            notes="Nominal"
        )
        d = metrics.to_dict()
        self.assertEqual(d["target_age"], 70)
        self.assertEqual(d["identity_similarity"], 0.825)
        self.assertEqual(d["age_mae"], 1.5)
        self.assertEqual(d["cyclic_similarity_display"], "N/A (Forward Only)")

    def test_cyclic_similarity_display(self):
        # 1. Unexecuted / None
        m_none = EvaluationMetrics(
            source_face_detected=True, generated_face_detected=True,
            identity_similarity=0.8, source_estimated_age=25.0,
            generated_estimated_age=65.0, target_age=65, age_mae=0.0,
            cyclic_similarity=None
        )
        self.assertEqual(m_none.cyclic_similarity_display, "N/A (Forward Only)")

        # 2. Valid float
        m_val = EvaluationMetrics(
            source_face_detected=True, generated_face_detected=True,
            identity_similarity=0.8, source_estimated_age=25.0,
            generated_estimated_age=65.0, target_age=65, age_mae=0.0,
            cyclic_similarity=0.6543
        )
        self.assertEqual(m_val.cyclic_similarity_display, "0.6543")

        # 3. NaN float
        m_nan = EvaluationMetrics(
            source_face_detected=True, generated_face_detected=True,
            identity_similarity=0.8, source_estimated_age=25.0,
            generated_estimated_age=65.0, target_age=65, age_mae=0.0,
            cyclic_similarity=float("nan")
        )
        self.assertEqual(m_nan.cyclic_similarity_display, "N/A (Forward Only)")

    def test_missing_face_handling(self):
        # Create a blank black image with no face
        blank_img = Image.new("RGB", (320, 320), color=(0, 0, 0))
        analyzer = FaceIdentityAnalyzer()
        res = analyzer.analyze_image(blank_img)
        self.assertFalse(res.face_detected)
        self.assertIsNone(res.embedding)
        self.assertIn("No face detected", res.error_message)

    def test_evaluator_constructor_and_api(self):
        """
        Verify that AgeWeaverEvaluator supports positional and keyword initialization
        using the correct parameter name 'analyzer'.
        """
        from ageweaver.frontend.state import get_evaluator, get_analyzer
        # Positional
        evaluator_pos = AgeWeaverEvaluator(FaceIdentityAnalyzer())
        self.assertIsNotNone(evaluator_pos.analyzer)

        # Keyword
        evaluator_kw = AgeWeaverEvaluator(analyzer=FaceIdentityAnalyzer())
        self.assertIsNotNone(evaluator_kw.analyzer)

        # Singleton from frontend state
        evaluator_state = get_evaluator()
        self.assertIsInstance(evaluator_state, AgeWeaverEvaluator)
        self.assertIsNotNone(evaluator_state.analyzer)

    def test_reference_identity_similarity_display(self):
        # 1. Unmeasured / None
        m_none = EvaluationMetrics(
            source_face_detected=True, generated_face_detected=True,
            identity_similarity=0.8, source_estimated_age=25.0,
            generated_estimated_age=65.0, target_age=65, age_mae=0.0,
            reference_identity_similarity=None
        )
        self.assertEqual(m_none.reference_similarity_display, "N/A (No Ground-Truth Pair)")
        self.assertIn("N/A (No Ground-Truth Pair)", m_none.to_dict()["reference_similarity_display"])

        # 2. Valid float
        m_val = EvaluationMetrics(
            source_face_detected=True, generated_face_detected=True,
            identity_similarity=0.8, source_estimated_age=25.0,
            generated_estimated_age=65.0, target_age=65, age_mae=0.0,
            reference_identity_similarity=0.5512
        )
        self.assertEqual(m_val.reference_similarity_display, "0.5512")

        # 3. NaN float
        m_nan = EvaluationMetrics(
            source_face_detected=True, generated_face_detected=True,
            identity_similarity=0.8, source_estimated_age=25.0,
            generated_estimated_age=65.0, target_age=65, age_mae=0.0,
            reference_identity_similarity=float("nan")
        )
        self.assertEqual(m_nan.reference_similarity_display, "N/A (No Ground-Truth Pair)")

    def test_reference_similarity_non_substitution(self):
        """
        Verify that when reference_target_image is None, reference_identity_similarity
        remains strictly None, and source similarity is NEVER substituted.
        """
        evaluator = AgeWeaverEvaluator()
        # Mock images: blank images without faces
        img_src = Image.new("RGB", (64, 64), color="black")
        img_gen = Image.new("RGB", (64, 64), color="black")
        metrics = evaluator.evaluate(
            source_image=img_src,
            generated_image=img_gen,
            target_age=60,
            reference_target_image=None
        )
        self.assertIsNone(metrics.reference_identity_similarity)
        self.assertIn("Reference target image pair unavailable", metrics.notes)

if __name__ == "__main__":
    unittest.main()

