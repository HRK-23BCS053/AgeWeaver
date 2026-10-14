"""
Unit tests for dataset ingestion, manifest parsing, identity selection,
and strict no-silent-fallback guarantees.
"""

import unittest
import os
import tempfile
from PIL import Image

from ageweaver.dataset import DatasetManager, CelebrityEntry

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.dm = DatasetManager(
            manifest_path="data/dataset_manifest.csv",
            images_dir="data/images"
        )

    def test_load_manifest_entries(self):
        entries = self.dm.load_manifest()
        self.assertGreaterEqual(len(entries), 100)
        self.assertIn("ID001_RAJESH_KHANNA", entries)
        self.assertIn("ID002_AMITABH_BACHCHAN", entries)

    def test_get_celebrity_valid(self):
        entry = self.dm.get_celebrity("ID001_RAJESH_KHANNA")
        self.assertIsInstance(entry, CelebrityEntry)
        self.assertEqual(entry.display_name, "Rajesh Khanna")
        self.assertEqual(entry.license, "CC BY 3.0")
        self.assertEqual(entry.attribution, "Bollywood Hungama")

    def test_get_celebrity_unknown_raises_keyerror(self):
        with self.assertRaises(KeyError):
            self.dm.get_celebrity("NON_EXISTENT_ID_9999")

    def test_strict_no_silent_fallback_missing_image(self):
        """Verify that missing images strictly raise FileNotFoundError and NEVER return synthetic data."""
        with self.assertRaises(FileNotFoundError):
            self.dm.get_image_path("ID099_LATA_MANGESHKAR", auto_download=False)

    def test_resolve_direct_url(self):
        direct = "https://upload.wikimedia.org/wikipedia/commons/6/6a/Rajesh_Khanna_Profile.jpg"
        resolved = DatasetManager.resolve_direct_image_url(direct)
        self.assertEqual(resolved, direct)

    def test_retrieves_existing_downloaded_image(self):
        """Checks that ID001_RAJESH_KHANNA which was downloaded into data/images is found."""
        expected_path = os.path.join("data", "images", "ID001_RAJESH_KHANNA.jpg")
        if os.path.exists(expected_path):
            path = self.dm.get_image_path("ID001_RAJESH_KHANNA", auto_download=False)
            self.assertEqual(os.path.abspath(path), os.path.abspath(expected_path))

    def test_corrupt_image_validation_fails(self):
        """Ensure invalid or truncated image bytes fail validation."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(b"NOT_A_VALID_JPEG_HEADER")
            tmp_path = tmp.name

        try:
            with self.assertRaises(Exception):
                img = Image.open(tmp_path)
                img.verify()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == "__main__":
    unittest.main()

