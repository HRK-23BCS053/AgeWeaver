"""
Unit tests for AgeWeaver dataset manifest validation.
"""

import unittest
import os
import csv
from scripts.validate_dataset import validate_manifest, REQUIRED_COLUMNS

class TestDatasetManifest(unittest.TestCase):
    def setUp(self):
        self.manifest_path = "data/dataset_manifest.csv"

    def test_manifest_file_exists(self):
        self.assertTrue(os.path.exists(self.manifest_path), "dataset_manifest.csv must exist")

    def test_manifest_schema_and_integrity(self):
        audit_res = validate_manifest(self.manifest_path)
        self.assertTrue(audit_res["valid"], f"Audit failed with errors: {audit_res.get('errors')}")
        self.assertGreaterEqual(audit_res["total_identities"], 100, "Manifest must document at least 100 identities")

    def test_rajesh_khanna_case_study_present(self):
        found = False
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "RAJESH_KHANNA" in row["identity_id"]:
                    found = True
                    self.assertEqual(row["display_name"], "Rajesh Khanna")
                    self.assertEqual(row["inclusion_status"], "qualitative_target")
                    self.assertIn("Wikimedia Commons", row["source_name"])
                    break
        self.assertTrue(found, "Rajesh Khanna must be documented in manifest as qualitative case study")

if __name__ == "__main__":
    unittest.main()

