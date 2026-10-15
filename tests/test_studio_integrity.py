"""
Unit and regression tests for AgeWeaver Studio integrity.
Guarantees:
1. Uploaded custom portraits NEVER receive precomputed celebrity benchmark outputs or metrics.
2. Stale session results from prior runs or foreign identities are immediately rejected.
3. Precomputed benchmark outputs are strictly gated to their exact benchmark subject.
4. Documented chronological age is kept strictly distinct from machine CNN age estimates.
"""

import unittest
import os
import sys

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath("src"))

from ageweaver.frontend.studio import (
    resolve_studio_result,
    resolve_source_age_metadata
)

class TestStudioIntegrity(unittest.TestCase):
    """Regression tests preventing cross-identity data leaks in Studio interface."""

    def test_custom_upload_never_receives_benchmark_fallback(self):
        """
        Verify that an uploaded custom portrait with no prior run in the session
        NEVER falls back to a precomputed celebrity benchmark image or metrics.
        """
        resolved, is_benchmark = resolve_studio_result(
            source_type="Upload Custom Portrait",
            selected_benchmark=None,
            current_source_id="UPLOAD_HRK_a8b9c1d2",
            current_sha256="a8b9c1d2e3f40506",
            session_result=None
        )
        self.assertIsNone(resolved, "Uploaded portrait must not receive any fallback result when session_result is None")
        self.assertFalse(is_benchmark, "is_benchmark must be False for uploaded portraits")

    def test_stale_celebrity_result_rejected_for_custom_upload(self):
        """
        Verify that if session state contains a leftover result from Rajesh Khanna,
        it is rejected when viewing an uploaded portrait.
        """
        stale_rajesh_khanna_result = {
            "experiment_id": "EXP_ID001_RAJESH_KHANNA_structured_prompt_T70",
            "input_id": "ID001_RAJESH_KHANNA",
            "input_sha256": "bc2beae7d13cf734",
            "condition": "structured_prompt",
            "source_age": 28,
            "target_age": 70,
            "identity_similarity": 0.4052,
            "generated_estimated_age": 69.0,
            "output_path": "outputs/experiments/ID001_RAJESH_KHANNA_structured_prompt_age70_seed42.png"
        }

        resolved, is_benchmark = resolve_studio_result(
            source_type="Upload Custom Portrait",
            selected_benchmark=None,
            current_source_id="UPLOAD_HRK_99999999",
            current_sha256="99999999aaaaaaaa",
            session_result=stale_rajesh_khanna_result
        )

        self.assertIsNone(resolved, "Stale result from a different identity must be rejected")
        self.assertFalse(is_benchmark)

    def test_matching_upload_trial_accepted(self):
        """
        Verify that a genuine live trial generated for the uploaded image IS accepted.
        """
        valid_upload_result = {
            "experiment_id": "RUN_UPLOAD_HRK_T60",
            "input_id": "UPLOAD_HRK_12345678",
            "input_sha256": "12345678abcdef00",
            "condition": "structured_prompt",
            "source_age": 25,
            "target_age": 60,
            "identity_similarity": 0.5123,
            "generated_estimated_age": 59.5,
            "output_path": "outputs/experiments/UPLOAD_HRK_12345678_structured_prompt_age60_seed42.png"
        }

        resolved, is_benchmark = resolve_studio_result(
            source_type="Upload Custom Portrait",
            selected_benchmark=None,
            current_source_id="UPLOAD_HRK_12345678",
            current_sha256="12345678abcdef00",
            session_result=valid_upload_result
        )

        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["experiment_id"], "RUN_UPLOAD_HRK_T60")
        self.assertEqual(resolved["identity_similarity"], 0.5123)
        self.assertFalse(is_benchmark, "Live trial must be flagged with is_benchmark=False")

    def test_benchmark_archive_only_serves_matching_benchmark(self):
        """
        Verify that benchmark preview is strictly gated to the matching benchmark ID.
        """
        # Deepika Padukone should NOT return Rajesh Khanna's output
        resolved_dp, is_b_dp = resolve_studio_result(
            source_type="Verified Benchmark Archive",
            selected_benchmark="ID009_DEEPIKA_PADUKONE",
            current_source_id="ID009_DEEPIKA_PADUKONE",
            current_sha256="dp_hash_123",
            session_result=None
        )
        self.assertIsNone(resolved_dp, "Deepika Padukone must not receive Rajesh Khanna's precomputed output")
        self.assertFalse(is_b_dp)

        # Rajesh Khanna should return the canonical benchmark trial if files exist on disk
        resolved_rk, is_b_rk = resolve_studio_result(
            source_type="Verified Benchmark Archive",
            selected_benchmark="ID001_RAJESH_KHANNA",
            current_source_id="ID001_RAJESH_KHANNA",
            current_sha256="bc2beae7d13cf734",
            session_result=None
        )
        if os.path.exists("outputs/experiments/ID001_RAJESH_KHANNA_structured_prompt_age70_seed42.png"):
            self.assertIsNotNone(resolved_rk)
            self.assertTrue(is_b_rk)
            self.assertEqual(resolved_rk["input_id"], "ID001_RAJESH_KHANNA")
            self.assertEqual(resolved_rk["identity_similarity"], 0.4052)
            self.assertTrue(os.path.exists(resolved_rk["benchmark_ref_path"]))

    def test_documented_vs_cnn_age_distinction(self):
        """
        Verify that documented chronological age is kept strictly distinct
        from CNN biometric estimates.
        """
        # Case A: Custom upload without user-specified age
        meta_upload_auto = resolve_source_age_metadata(
            source_type="Upload Custom Portrait",
            benchmark_id=None,
            user_provided_age=0,
            cnn_estimated_age=24.7
        )
        self.assertFalse(meta_upload_auto["is_documented"])
        self.assertIsNone(meta_upload_auto["documented_age"])
        self.assertEqual(meta_upload_auto["baseline_age"], 25)
        self.assertIn("CNN estimated", meta_upload_auto["baseline_source_desc"])

        # Case B: Custom upload with user-specified chronological age
        meta_upload_user = resolve_source_age_metadata(
            source_type="Upload Custom Portrait",
            benchmark_id=None,
            user_provided_age=21,
            cnn_estimated_age=29.3
        )
        self.assertTrue(meta_upload_user["is_documented"])
        self.assertEqual(meta_upload_user["documented_age"], 21)
        self.assertEqual(meta_upload_user["baseline_age"], 21)
        self.assertIn("documented", meta_upload_user["baseline_source_desc"])

        # Case C: Benchmark portrait (Rajesh Khanna)
        meta_benchmark = resolve_source_age_metadata(
            source_type="Verified Benchmark Archive",
            benchmark_id="ID001_RAJESH_KHANNA",
            user_provided_age=None,
            cnn_estimated_age=41.0
        )
        self.assertTrue(meta_benchmark["is_documented"])
        self.assertEqual(meta_benchmark["documented_age"], 28)
        self.assertEqual(meta_benchmark["baseline_age"], 28)
        self.assertIn("documented", meta_benchmark["baseline_source_desc"])

if __name__ == "__main__":
    unittest.main()
