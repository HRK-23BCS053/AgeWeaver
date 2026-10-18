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
    reference_identity_similarity: Optional[float] = None
    reference_face_detected: Optional[bool] = None
    reference_estimated_age: Optional[float] = None
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

    @property
    def reference_similarity_display(self) -> str:
        """User-facing string for reference-based identity similarity (ID_sim_ref)."""
        if self.reference_identity_similarity is None:
            return "N/A (No Ground-Truth Pair)"
        try:
            val = float(self.reference_identity_similarity)
            if np.isnan(val):
                return "N/A (No Ground-Truth Pair)"
            return f"{val:.4f}"
        except (ValueError, TypeError):
            return "N/A"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["cyclic_similarity_display"] = self.cyclic_similarity_display
        d["reference_similarity_display"] = self.reference_similarity_display
        return d

class AgeWeaverEvaluator:
    """
    Orchestrates quantitative evaluation of age-progressed face images.
    Implements:
    - Source Identity Cosine Similarity (ID_sim_src)
    - Reference-Based Target Identity Cosine Similarity (ID_sim_ref, FaceTT Section 4.1)
    - Cyclic Identity Similarity (ID_sim_cyc, FaceTT Section 4.1)
    - Apparent Age Estimation and Target MAE
    """
    def __init__(self, analyzer: Optional[FaceIdentityAnalyzer] = None):
        self.analyzer = analyzer or FaceIdentityAnalyzer()

    def compute_reference_identity_similarity(
        self,
        generated_image: Image.Image,
        reference_target_image: Image.Image
    ) -> Optional[float]:
        """
        Computes Reference-Based Identity Similarity (ID_sim_ref) from FaceTT Section 4.1:
            ID_sim_ref = sim(f(I_target_reference), f(I_generated))
        where I_target_reference is a genuine photograph of the SAME person at the target age.
        Returns None if either face is undetected.
        CRITICAL: Never substitutes source similarity when reference pair is unavailable.
        """
        gen_res = self.analyzer.analyze_image(generated_image)
        ref_res = self.analyzer.analyze_image(reference_target_image)
        if gen_res.face_detected and ref_res.face_detected:
            sim = self.analyzer.compute_cosine_similarity(ref_res.embedding, gen_res.embedding)
            return round(sim, 4) if sim is not None else None
        return None

    def evaluate(
        self,
        source_image: Image.Image,
        generated_image: Image.Image,
        target_age: int,
        source_age: Optional[int] = None,
        cyclic_reconstructed_image: Optional[Image.Image] = None,
        reference_target_image: Optional[Image.Image] = None
    ) -> EvaluationMetrics:
        """
        Computes identity cosine similarity, target-age MAE, cyclic similarity,
        and reference-based target similarity (ID_sim_ref).
        """
        # Analyze source image
        src_res = self.analyzer.analyze_image(source_image)
        # Analyze generated image
        gen_res = self.analyzer.analyze_image(generated_image)

        # Identity similarity to source
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

        # Reference-based target identity similarity (FaceTT ID_sim_ref)
        ref_sim = None
        ref_det = None
        ref_age = None
        if reference_target_image is not None:
            ref_res = self.analyzer.analyze_image(reference_target_image)
            ref_det = ref_res.face_detected
            ref_age = ref_res.estimated_age
            if gen_res.face_detected and ref_res.face_detected:
                ref_sim = self.analyzer.compute_cosine_similarity(
                    ref_res.embedding,
                    gen_res.embedding
                )
                if ref_sim is not None:
                    ref_sim = round(ref_sim, 4)
            else:
                notes.append("Reference face not detected for ID_sim_ref evaluation.")
        else:
            notes.append("Reference target image pair unavailable; ID_sim_ref unmeasured.")

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
            reference_identity_similarity=ref_sim,
            reference_face_detected=ref_det,
            reference_estimated_age=ref_age,
            notes="; ".join(notes)
        )

