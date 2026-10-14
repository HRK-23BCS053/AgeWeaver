"""
Execution script for AgeWeaver comparative research experiments.
Executes:
- Source face preparation (deterministic synthetic identity benchmark).
- Experiment A: Generic prompt baseline.
- Experiment B: Structured age-aware prompt refinement.
- Experiment C: Proposed AgeWeaver adaptive controller (delta-scaled strength/guidance + cyclic round-trip).
Across target age benchmarks (45, 65).
Records live metrics, timings, and outputs.
"""

import os
import sys
import time
from PIL import Image

# Ensure src is on path
sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("."))

from ageweaver.pipeline import AgeWeaverPipeline
from ageweaver.identity import FaceIdentityAnalyzer
from ageweaver.experiment import ExperimentRunner
from ageweaver.utils import save_image, setup_logger

logger = setup_logger("run_experiments")

def main():
    print("=" * 70)
    print(" AGEWEAVER: AUTONOMOUS RESEARCH EXPERIMENTS")
    print(" Device: CPU (Intel Iris Xe) | Model: Stable Diffusion v1-5 (Offline)")
    print(" Evaluator: InsightFace buffalo_l (ArcFace + GenderAge)")
    print("=" * 70)

    output_dir = "outputs/experiments"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Initialize Pipeline & Evaluator
    pipeline = AgeWeaverPipeline()
    pipeline.load_pipeline()
    runner = ExperimentRunner(output_dir=output_dir, pipeline=pipeline)

    # 2. Source Face Preparation
    # Generate high-quality deterministic synthetic face to avoid copyright/privacy issues
    source_img_path = os.path.join(output_dir, "benchmark_source_identity.png")
    
    if os.path.exists(source_img_path):
        logger.info(f"Loading existing benchmark source identity from {source_img_path}")
        source_image = Image.open(source_img_path).convert("RGB")
    else:
        logger.info("Generating deterministic synthetic benchmark identity portrait...")
        source_image = pipeline.generate_synthetic_identity(
            prompt="high quality studio portrait photograph of a 25-year-old young person, neutral expression, looking directly at camera, soft studio lighting, sharp facial details, 8k",
            seed=2024,
            steps=18
        )
        save_image(source_image, source_img_path)

    # Analyze source identity
    src_res = runner.analyzer.analyze_image(source_image)
    if not src_res.face_detected:
        logger.error(f"Source face detection failed: {src_res.error_message}")
        sys.exit(1)

    source_age = int(round(src_res.estimated_age)) if src_res.estimated_age else 25
    gender = src_res.gender or "person"
    logger.info(f"Source Identity Verified: Detected=True, Estimated Age={src_res.estimated_age:.1f} (rounded: {source_age}), Gender={gender}")

    # 3. Controlled Experiment Matrix
    # We test two distinct target ages:
    # Target 45 (mature adult, moderate delta)
    # Target 68 (senior, large delta)
    target_ages = [45, 68]
    conditions = ["generic_baseline", "structured_prompt", "adaptive_control"]

    results = []

    for tgt_age in target_ages:
        for cond in conditions:
            exp_id = f"EXP_{cond}_T{tgt_age}"
            is_cyclic = (cond == "adaptive_control")
            
            logger.info(f"\n>>> Executing {exp_id} (Target Age: {tgt_age}) <<<")
            trial_res = runner.run_single_trial(
                experiment_id=exp_id,
                condition=cond,
                source_image=source_image,
                input_id="SYNTH_ID_2024",
                target_age=tgt_age,
                source_age=source_age,
                seed=42,  # Keep seed fixed across conditions for fair control
                steps=15,
                execute_cyclic=is_cyclic
            )
            results.append(trial_res)
            print(f"Result for {exp_id}:")
            print(f"  Condition           : {trial_res['condition']}")
            print(f"  Target Age          : {trial_res['target_age']}")
            print(f"  Generated Est. Age  : {trial_res['generated_estimated_age']}")
            print(f"  Age MAE             : {trial_res['target_age_mae']}")
            print(f"  Identity Cosine Sim : {trial_res['identity_similarity']}")
            print(f"  Cyclic Sim (if run) : {trial_res['cyclic_similarity']}")
            print(f"  Runtime             : {trial_res['runtime_sec']}s")
            print(f"  Output Image        : {trial_res['output_path']}")

    print("\n" + "=" * 70)
    print(" EXPERIMENT MATRIX SUMMARY")
    print("=" * 70)
    print(f"{'Condition':20} | {'Tgt':4} | {'EstAge':7} | {'Age MAE':8} | {'Ident Sim':10} | {'Cyclic Sim':11} | {'Time (s)':8}")
    print("-" * 75)
    for r in results:
        est_age_str = f"{r['generated_estimated_age']:.1f}" if r['generated_estimated_age'] is not None else "N/A"
        mae_str = f"{r['target_age_mae']:.1f}" if r['target_age_mae'] is not None else "N/A"
        sim_str = f"{r['identity_similarity']:.4f}" if r['identity_similarity'] is not None else "N/A"
        cyc_str = f"{r['cyclic_similarity']:.4f}" if r['cyclic_similarity'] is not None else "N/A"
        print(f"{r['condition']:20} | {r['target_age']:4} | {est_age_str:7} | {mae_str:8} | {sim_str:10} | {cyc_str:11} | {r['runtime_sec']:8.1f}")
    print("=" * 70)
    print(f"Logs written to:\n - {runner.csv_log_path}\n - {runner.jsonl_log_path}")

if __name__ == "__main__":
    main()

