"""
Identity preservation and facial attribute analysis module using InsightFace (ArcFace).
Provides face detection, 512-d normalized embedding extraction,
cosine similarity computation, and age estimation.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2
from PIL import Image

@dataclass
class FaceIdentityResult:
    """Encapsulates the face analysis and identity embedding result."""
    face_detected: bool
    embedding: Optional[np.ndarray] = None
    estimated_age: Optional[float] = None
    gender: Optional[str] = None  # "male" or "female"
    bbox: Optional[Tuple[int, int, int, int]] = None
    det_score: Optional[float] = None
    error_message: Optional[str] = None

class FaceIdentityAnalyzer:
    """
    Wrapper around InsightFace Buffalo_l suite running locally on CPU.
    Provides ArcFace feature extraction, age estimation, and cosine similarity.
    """
    def __init__(self, model_name: str = "buffalo_l", det_size: Tuple[int, int] = (320, 320)):
        self.model_name = model_name
        self.det_size = det_size
        self._app = None

    def _ensure_loaded(self):
        if self._app is None:
            import insightface
            from insightface.app import FaceAnalysis
            # Explicitly force CPUExecutionProvider to match non-GPU environment
            self._app = FaceAnalysis(name=self.model_name, providers=['CPUExecutionProvider'])
            self._app.prepare(ctx_id=0, det_size=self.det_size)

    def analyze_image(self, pil_image: Image.Image) -> FaceIdentityResult:
        """
        Detects primary face, extracts ArcFace 512-d normalized embedding and estimated age.
        """
        try:
            self._ensure_loaded()
            # Convert PIL RGB to OpenCV BGR
            rgb_arr = np.array(pil_image.convert("RGB"))
            bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

            faces = self._app.get(bgr_arr)
            if not faces:
                return FaceIdentityResult(
                    face_detected=False,
                    error_message="No face detected in the image"
                )

            # If multiple faces, pick the one with the largest bounding box area
            primary_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

            # ArcFace embedding extraction and normalization
            raw_emb = primary_face.embedding
            norm = np.linalg.norm(raw_emb)
            normed_emb = raw_emb / norm if norm > 1e-6 else raw_emb

            # Estimated age and gender
            est_age = float(primary_face.age) if hasattr(primary_face, "age") else None
            gender = None
            if hasattr(primary_face, "gender"):
                # InsightFace convention: 1 for male, 0 for female
                gender = "male" if primary_face.gender == 1 else "female"

            bbox = tuple(map(int, primary_face.bbox))
            det_score = float(primary_face.det_score) if hasattr(primary_face, "det_score") else None

            return FaceIdentityResult(
                face_detected=True,
                embedding=normed_emb,
                estimated_age=est_age,
                gender=gender,
                bbox=bbox,
                det_score=det_score
            )

        except Exception as e:
            return FaceIdentityResult(
                face_detected=False,
                error_message=f"Analysis failed with exception: {str(e)}"
            )

    @staticmethod
    def compute_cosine_similarity(emb1: Optional[np.ndarray], emb2: Optional[np.ndarray]) -> Optional[float]:
        """
        Calculates cosine similarity between two 512-dimensional face embeddings.
        Returns float in [-1.0, 1.0], or None if either embedding is missing.
        """
        if emb1 is None or emb2 is None:
            return None
        n1 = np.linalg.norm(emb1)
        n2 = np.linalg.norm(emb2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        sim = float(np.dot(emb1, emb2) / (n1 * n2))
        return max(-1.0, min(1.0, sim))

