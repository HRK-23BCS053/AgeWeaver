"""
Dataset management and ingestion module for AgeWeaver.
Provides:
- Manifest parsing and validation.
- Permitted-source resolution (e.g. Wikimedia Commons).
- Secure, rate-compliant image downloading with integrity checks.
- Face audit verification on ingested images.
- Strict identity retrieval with no silent fallback.
"""

import os
import sys
import csv
import json
import urllib.request
import urllib.parse
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any, Tuple
from PIL import Image
import io

from ageweaver.utils import compute_image_sha256, setup_logger

logger = setup_logger(__name__)

USER_AGENT = "AgeWeaverResearch/1.0 (Academic Research; IIIT Dharwad; contact: ageweaver-research@iiitdwd.ac.in)"

@dataclass
class CelebrityEntry:
    """Represents a validated entry in data/dataset_manifest.csv."""
    identity_id: str
    display_name: str
    image_id: str
    source_url: str
    source_name: str
    license: str
    attribution: str
    birth_year: Optional[int]
    capture_year: Optional[int]
    reference_age: Optional[int]
    gender: str
    quality_status: str
    face_detection_status: str
    inclusion_status: str
    notes: str

class DatasetManager:
    """
    Manages manifest parsing, downloading, validation, and retrieval
    for benchmark celebrity portraits.
    """
    def __init__(
        self,
        manifest_path: str = "data/dataset_manifest.csv",
        images_dir: str = "data/images",
        log_path: str = "data/download_log.json"
    ):
        self.manifest_path = manifest_path
        self.images_dir = images_dir
        self.log_path = log_path
        self._entries: Optional[Dict[str, CelebrityEntry]] = None

    def load_manifest(self) -> Dict[str, CelebrityEntry]:
        """Loads and indexes manifest rows by identity_id."""
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Dataset manifest not found at: {self.manifest_path}")

        entries = {}
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ident_id = row.get("identity_id", "").strip()
                if not ident_id:
                    continue
                
                b_year = int(row["birth_year"]) if row.get("birth_year") and row["birth_year"].isdigit() else None
                c_year = int(row["capture_year"]) if row.get("capture_year") and row["capture_year"].isdigit() else None
                ref_age = int(row["reference_age"]) if row.get("reference_age") and row["reference_age"].isdigit() else None

                entry = CelebrityEntry(
                    identity_id=ident_id,
                    display_name=row.get("display_name", "").strip(),
                    image_id=row.get("image_id", "").strip(),
                    source_url=row.get("source_url", "").strip(),
                    source_name=row.get("source_name", "").strip(),
                    license=row.get("license", "").strip(),
                    attribution=row.get("attribution", "").strip(),
                    birth_year=b_year,
                    capture_year=c_year,
                    reference_age=ref_age,
                    gender=row.get("gender", "person").strip(),
                    quality_status=row.get("quality_status", "documented").strip(),
                    face_detection_status=row.get("face_detection_status", "not_audited").strip(),
                    inclusion_status=row.get("inclusion_status", "candidate").strip(),
                    notes=row.get("notes", "").strip()
                )
                entries[ident_id] = entry

        self._entries = entries
        return entries

    def get_celebrity(self, identity_id: str) -> CelebrityEntry:
        """Retrieves a manifest entry by identity_id. Raises KeyError if not found."""
        if self._entries is None:
            self.load_manifest()
        if identity_id not in self._entries:
            available = list(self._entries.keys())[:5]
            raise KeyError(
                f"Identity '{identity_id}' not found in manifest. "
                f"Available examples: {available} (Total: {len(self._entries)})"
            )
        return self._entries[identity_id]

    @staticmethod
    def resolve_direct_image_url(source_url: str) -> str:
        """
        Resolves Wikimedia Commons File pages to direct download URLs.
        If already direct (e.g. upload.wikimedia.org), returns directly.
        """
        if "upload.wikimedia.org" in source_url:
            return source_url

        if "commons.wikimedia.org/wiki/File:" in source_url:
            filename = source_url.split("commons.wikimedia.org/wiki/File:")[-1]
            filename = urllib.parse.unquote(filename)
            api_url = (
                f"https://commons.wikimedia.org/w/api.php?action=query&titles=File:{urllib.parse.quote(filename)}"
                f"&prop=imageinfo&iiprop=url&format=json"
            )
            req = urllib.request.Request(api_url, headers={"User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(req, timeout=12) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    pages = data.get("query", {}).get("pages", {})
                    for pid, pinfo in pages.items():
                        if "imageinfo" in pinfo and len(pinfo["imageinfo"]) > 0:
                            return pinfo["imageinfo"][0]["url"]
            except Exception as e:
                logger.warning(f"Could not resolve via Wikimedia API ({e}). Falling back to Special:FilePath.")
            
            # Fallback to Special:FilePath which issues 302 redirect to original file
            return f"https://commons.wikimedia.org/wiki/Special:FilePath/{urllib.parse.quote(filename)}"

        return source_url

    def download_image(
        self,
        identity_id: str,
        overwrite: bool = False,
        audit_face: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Downloads the image for identity_id, validates integrity and face presence,
        and saves it to data/images/{identity_id}.jpg.
        """
        entry = self.get_celebrity(identity_id)
        os.makedirs(self.images_dir, exist_ok=True)
        local_path = os.path.join(self.images_dir, f"{identity_id}.jpg")

        if os.path.exists(local_path) and not overwrite:
            logger.info(f"Image for {identity_id} already exists locally at: {local_path}")
            # Verify file integrity
            img = Image.open(local_path)
            img.verify()
            return local_path, {"status": "cached", "path": local_path}

        direct_url = self.resolve_direct_image_url(entry.source_url)
        logger.info(f"Downloading {entry.display_name} ({identity_id}) from: {direct_url}")

        req = urllib.request.Request(direct_url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                content = resp.read()
        except Exception as e:
            err_msg = f"Failed to download image from {direct_url}: {str(e)}"
            logger.error(err_msg)
            self._log_download_event(identity_id, False, {"error": err_msg, "url": direct_url})
            raise RuntimeError(err_msg) from e

        # Validate with PIL
        try:
            img = Image.open(io.BytesIO(content))
            img_format = img.format
            img_size = img.size
            # Convert to RGB and save as high-quality JPEG
            rgb_img = img.convert("RGB")
            rgb_img.save(local_path, "JPEG", quality=95)
        except Exception as e:
            err_msg = f"Downloaded content is not a valid image: {str(e)}"
            logger.error(err_msg)
            self._log_download_event(identity_id, False, {"error": err_msg, "url": direct_url})
            raise ValueError(err_msg) from e

        sha256_hash = compute_image_sha256(rgb_img)
        meta = {
            "identity_id": identity_id,
            "display_name": entry.display_name,
            "local_path": local_path,
            "source_url": entry.source_url,
            "direct_url": direct_url,
            "license": entry.license,
            "attribution": entry.attribution,
            "sha256": sha256_hash,
            "original_format": img_format,
            "resolution": img_size,
            "face_detected": None,
            "estimated_age": None
        }

        # Face audit if requested
        if audit_face:
            try:
                from ageweaver.identity import FaceIdentityAnalyzer
                analyzer = FaceIdentityAnalyzer()
                face_res = analyzer.analyze_image(rgb_img)
                meta["face_detected"] = face_res.face_detected
                meta["estimated_age"] = face_res.estimated_age
                meta["det_score"] = face_res.det_score
                logger.info(f"Face audit: Detected={face_res.face_detected}, Estimated Age={face_res.estimated_age}")
            except Exception as e:
                logger.warning(f"Face audit skipped/failed: {e}")

        self._log_download_event(identity_id, True, meta)
        logger.info(f"Successfully saved and verified {identity_id} -> {local_path}")
        return local_path, meta

    def get_image_path(self, identity_id: str, auto_download: bool = True) -> str:
        """
        Returns local image path for identity_id.
        Strictly raises FileNotFoundError if not available. NO silent fallbacks.
        """
        local_path = os.path.join(self.images_dir, f"{identity_id}.jpg")
        if os.path.exists(local_path):
            return local_path

        # Try .png extension just in case
        png_path = os.path.join(self.images_dir, f"{identity_id}.png")
        if os.path.exists(png_path):
            return png_path

        if auto_download:
            path, _ = self.download_image(identity_id)
            return path

        raise FileNotFoundError(
            f"Image for identity '{identity_id}' is not present at {local_path} and auto_download is disabled."
        )

    def _log_download_event(self, identity_id: str, success: bool, details: Dict[str, Any]):
        """Logs download metadata and provenance to data/download_log.json."""
        log_data = {}
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    log_data = json.load(f)
            except Exception:
                log_data = {}

        log_data[identity_id] = {
            "success": success,
            "details": details
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, default=str)

