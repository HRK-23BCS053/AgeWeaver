"""
Evaluation module for AgeWeaver.
Computes ArcFace identity cosine similarity, estimated age MAE,
face detection success rates, and full cyclic identity consistency.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from PIL import Image
import numpy as np

from ageweaver.identity import FaceIdentityAnalyzer, FaceIdentityResult
from ageweaver.utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class EvaluationMetrics:
    """Encapsulates all quantitative metrics for an age progression experiment."""
    source_face_detected: bool
    generated_face_detected: bool
    identity_similarity: Optional[float]
    source_estimated_age: Optional[float]
    generated_estimated_age: Optional[float]
    target_age: int
    age_mae: Optional[float]
    cyclic_similarity: Optional[float] = None
    cyclic_face_detected: Optional[bool] = None
    cyclic_estimated_age: Optional[float] = None
    notes: str = ""

    @property
    def cyclic_similarity_display(self) -> str:
        """User-facing string for cyclic similarity that avoids raw NaN."""
        if self.cyclic_similarity is None:
            return "N/A (Forward Only)"
        try:
            val = float(self.cyclic_similarity)
            if np.isnan(val):
                return "N/A (Forward Only)"
            return f"{val:.4f}"
        except (ValueError, TypeError):
            return "N/A"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["cyclic_similarity_display"] = self.cyclic_similarity_display
        return d

class AgeWeaverEvaluator:
    """
    Orchestrates quantitative evaluation of age-progressed face images.
    """
    def __init__(self, analyzer: Optional[FaceIdentityAnalyzer] = None):
        self.analyzer = analyzer or FaceIdentityAnalyzer()

    def evaluate(
        self,
        source_image: Image.Image,
        generated_image: Image.Image,
        target_age: int,
        source_age: Optional[int] = None,
        cyclic_reconstructed_image: Optional[Image.Image] = None
    ) -> EvaluationMetrics:
        """
        Computes identity cosine similarity, target-age MAE, and cyclic similarity.
        """
        # Analyze source image
        src_res = self.analyzer.analyze_image(source_image)
        # Analyze generated image
        gen_res = self.analyzer.analyze_image(generated_image)

        # Identity similarity
        identity_sim = None
        if src_res.face_detected and gen_res.face_detected:
            identity_sim = self.analyzer.compute_cosine_similarity(
                src_res.embedding,
                gen_res.embedding
            )
            if identity_sim is not None:
                identity_sim = round(identity_sim, 4)

        # Target-age error (MAE)
        age_mae = None
        if gen_res.face_detected and gen_res.estimated_age is not None:
            age_mae = round(abs(gen_res.estimated_age - target_age), 2)

        # Cyclic identity similarity (if round-trip image provided)
        cyclic_sim = None
        cyclic_det = None
        cyclic_age = None
        notes = []

        if not src_res.face_detected:
            notes.append("Source face not detected.")
        if not gen_res.face_detected:
            notes.append("Generated face not detected.")

        if cyclic_reconstructed_image is not None:
            cyc_res = self.analyzer.analyze_image(cyclic_reconstructed_image)
            cyclic_det = cyc_res.face_detected
            cyclic_age = cyc_res.estimated_age
            if src_res.face_detected and cyc_res.face_detected:
                cyclic_sim = self.analyzer.compute_cosine_similarity(
                    src_res.embedding,
                    cyc_res.embedding
                )
                if cyclic_sim is not None:
                    cyclic_sim = round(cyclic_sim, 4)
            else:
                notes.append("Cyclic face not detected for round-trip identity evaluation.")
        else:
            notes.append("Forward-only evaluation; cyclic reconstruction not executed.")

        return EvaluationMetrics(
            source_face_detected=src_res.face_detected,
            generated_face_detected=gen_res.face_detected,
            identity_similarity=identity_sim,
            source_estimated_age=src_res.estimated_age,
            generated_estimated_age=gen_res.estimated_age,
            target_age=target_age,
            age_mae=age_mae,
            cyclic_similarity=cyclic_sim,
            cyclic_face_detected=cyclic_det,
            cyclic_estimated_age=cyclic_age,
            notes="; ".join(notes)
        )

