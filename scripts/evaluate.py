"""
Evaluation script for AgeWeaver.
Computes ArcFace identity preservation, age estimation, and target MAE
between a source image and an age-progressed image.
"""

import os
import sys
import argparse
from PIL import Image

sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("."))

from ageweaver.identity import FaceIdentityAnalyzer
from ageweaver.evaluation import AgeWeaverEvaluator

def main():
    parser = argparse.ArgumentParser(description="AgeWeaver Quantitative Evaluator")
    parser.add_argument("--source", type=str, required=True, help="Path to source ground-truth image")
    parser.add_argument("--generated", type=str, required=True, help="Path to age-progressed image")
    parser.add_argument("--target-age", type=int, required=True, help="Target age requested in progression")
    parser.add_argument("--cyclic", type=str, default=None, help="Optional path to cyclic reconstructed image")

    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"Error: Source image not found: {args.source}")
        sys.exit(1)
    if not os.path.exists(args.generated):
        print(f"Error: Generated image not found: {args.generated}")
        sys.exit(1)

    src_img = Image.open(args.source).convert("RGB")
    gen_img = Image.open(args.generated).convert("RGB")
    cyc_img = Image.open(args.cyclic).convert("RGB") if args.cyclic and os.path.exists(args.cyclic) else None

    analyzer = FaceIdentityAnalyzer()
    evaluator = AgeWeaverEvaluator(analyzer)

    metrics = evaluator.evaluate(
        source_image=src_img,
        generated_image=gen_img,
        target_age=args.target_age,
        cyclic_reconstructed_image=cyc_img
    )

    print("\n" + "=" * 60)
    print(" EVALUATION METRICS REPORT")
    print("=" * 60)
    print(f"  Source Face Detected       : {metrics.source_face_detected}")
    print(f"  Generated Face Detected    : {metrics.generated_face_detected}")
    if metrics.source_estimated_age is not None:
        print(f"  Source Estimated Age       : {metrics.source_estimated_age:.1f}")
    if metrics.generated_estimated_age is not None:
        print(f"  Generated Estimated Age    : {metrics.generated_estimated_age:.1f}")
    print(f"  Target Age Requested       : {metrics.target_age}")
    if metrics.age_mae is not None:
        print(f"  Target Age Error (MAE)     : {metrics.age_mae:.1f}")
    if metrics.identity_similarity is not None:
        print(f"  Identity Cosine Similarity : {metrics.identity_similarity:.4f}")
    if metrics.cyclic_similarity is not None:
        print(f"  Cyclic Round-Trip Sim      : {metrics.cyclic_similarity:.4f}")
    if metrics.notes:
        print(f"  Notes                      : {metrics.notes}")
    print("=" * 60)

if __name__ == "__main__":
    main()

