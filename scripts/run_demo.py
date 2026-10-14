"""
Interactive CLI demonstration script for AgeWeaver.
Runs facial age progression on a provided image or generates a synthetic face benchmark.
Computes and reports live ArcFace identity preservation and target age metrics.
"""

import os
import sys
import argparse
import time
from PIL import Image

sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("."))

from ageweaver.pipeline import AgeWeaverPipeline
from ageweaver.identity import FaceIdentityAnalyzer
from ageweaver.evaluation import AgeWeaverEvaluator
from ageweaver.utils import save_image, setup_logger

logger = setup_logger("ageweaver_demo")

def main():
    parser = argparse.ArgumentParser(description="AgeWeaver Facial Age Progression CLI Demo")
    parser.add_argument("--image", type=str, default=None, help="Path to input facial image (RGB)")
    parser.add_argument("--target-age", type=int, default=65, help="Target age (1-100)")
    parser.add_argument("--mode", type=str, choices=["structured", "generic"], default="structured", help="Prompt refinement mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--steps", type=int, default=15, help="Number of diffusion inference steps")
    parser.add_argument("--output", type=str, default="outputs/demo_output.png", help="Path to save output image")
    parser.add_argument("--cyclic", action="store_true", help="Perform cyclic round-trip reconstruction")

    args = parser.parse_args()

    print("=" * 65)
    print(" AgeWeaver Facial Age Progression Demo")
    print(f" Target Age: {args.target_age} | Mode: {args.mode} | Seed: {args.seed} | Steps: {args.steps}")
    print("=" * 65)

    pipeline = AgeWeaverPipeline()
    pipeline.load_pipeline()
    analyzer = FaceIdentityAnalyzer()
    evaluator = AgeWeaverEvaluator(analyzer)

    # Load or generate source image
    if args.image and os.path.exists(args.image):
        logger.info(f"Loading user image: {args.image}")
        source_img = Image.open(args.image).convert("RGB")
    else:
        logger.info("No input image specified. Generating synthetic benchmark portrait...")
        source_img = pipeline.generate_synthetic_identity(
            prompt="studio photographic portrait of a 25-year-old person, neutral expression, high resolution",
            seed=args.seed,
            steps=18
        )
        os.makedirs("outputs", exist_ok=True)
        save_image(source_img, "outputs/demo_source_image.png")
        print("  [Saved] Source image written to outputs/demo_source_image.png")

    # Analyze source face
    src_res = analyzer.analyze_image(source_img)
    if not src_res.face_detected:
        print("[ERROR] No face detected in source image. Please provide a clear, centered portrait.")
        sys.exit(1)

    source_age = int(round(src_res.estimated_age)) if src_res.estimated_age else 25
    print(f"  [Source Face Detected] Estimated Age: {src_res.estimated_age:.1f} years | Gender: {src_res.gender}")

    # Forward progression
    print(f"\nRunning {args.mode} progression to age {args.target_age}...")
    gen_result = pipeline.progress_age(
        source_image=source_img,
        target_age=args.target_age,
        source_age=source_age,
        mode=args.mode,
        num_inference_steps=args.steps,
        seed=args.seed
    )

    save_image(gen_result.generated_image, args.output)
    print(f"  [Saved] Age-progressed image written to {args.output}")

    # Optional cyclic reconstruction
    cyclic_img = None
    if args.cyclic:
        print(f"\nRunning cyclic reconstruction back to source age {source_age}...")
        cyc_result = pipeline.progress_age(
            source_image=gen_result.generated_image,
            target_age=source_age,
            source_age=args.target_age,
            mode=args.mode,
            num_inference_steps=args.steps,
            seed=args.seed + 1
        )
        cyclic_img = cyc_result.generated_image
        cyc_path = "outputs/demo_cyclic_recon.png"
        save_image(cyclic_img, cyc_path)
        print(f"  [Saved] Cyclic reconstructed image written to {cyc_path}")

    # Evaluate
    print("\nComputing quantitative metrics (InsightFace ArcFace)...")
    metrics = evaluator.evaluate(
        source_image=source_img,
        generated_image=gen_result.generated_image,
        target_age=args.target_age,
        source_age=source_age,
        cyclic_reconstructed_image=cyclic_img
    )

    print("\n" + "=" * 65)
    print(" DEMO EVALUATION SUMMARY")
    print("=" * 65)
    print(f"  Source Face Detected       : {metrics.source_face_detected}")
    print(f"  Generated Face Detected    : {metrics.generated_face_detected}")
    print(f"  Source Estimated Age       : {metrics.source_estimated_age:.1f} yrs")
    print(f"  Generated Estimated Age    : {metrics.generated_estimated_age:.1f} yrs")
    print(f"  Target Age Requested       : {metrics.target_age} yrs")
    print(f"  Age Error (MAE)            : {metrics.age_mae:.1f} yrs")
    print(f"  ArcFace Identity Cosine Sim: {metrics.identity_similarity:.4f}")
    if metrics.cyclic_similarity is not None:
        print(f"  Cyclic Round-Trip Sim      : {metrics.cyclic_similarity:.4f}")
    print(f"  CPU Wall-clock Runtime     : {gen_result.elapsed_time_sec:.2f}s")
    print("=" * 65)

if __name__ == "__main__":
    main()

