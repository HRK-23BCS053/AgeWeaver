"""
Execution script for AgeWeaver comparative research experiments.
Executes:
- Source face preparation from real celebrity dataset (e.g. ID001_RAJESH_KHANNA) or explicit image path.
- Experiment A: Generic prompt baseline.
- Experiment B: Structured age-aware prompt refinement.
- Experiment C: Proposed AgeWeaver adaptive controller (delta-scaled strength/guidance + cyclic round-trip).
Across target age benchmarks (e.g. 45, 68).
Records live metrics, timings, and outputs with strict no-silent-fallback rules.
"""

import os
import sys
import argparse
import time
from PIL import Image

# Ensure src is on path
sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("."))

from ageweaver.pipeline import AgeWeaverPipeline
from ageweaver.identity import FaceIdentityAnalyzer
from ageweaver.experiment import ExperimentRunner
from ageweaver.dataset import DatasetManager
from ageweaver.utils import save_image, setup_logger

logger = setup_logger("run_experiments")

def main():
    parser = argparse.ArgumentParser(description="AgeWeaver Controlled Comparative Experiments")
    parser.add_argument("--identity-id", type=str, default=None, help="Celebrity identity ID from manifest (e.g. ID001_RAJESH_KHANNA)")
    parser.add_argument("--image", type=str, default=None, help="Explicit path to an image file")
    parser.add_argument("--source-age", type=int, default=None, help="Documented chronological source age of the reference subject (e.g. 28)")
    parser.add_argument("--target-ages", type=str, default="45,60,70", help="Comma-separated older target ages (default: '45,60,70')")
    parser.add_argument("--steps", type=int, default=15, help="Diffusion inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed for controlled comparisons")
    parser.add_argument("--scheduler", type=str, default="dpmsolver++", choices=["dpmsolver++", "euler", "ddim", "pndm"], help="Diffusion scheduler algorithm (default: 'dpmsolver++')")
    parser.add_argument("--allow-synthetic", action="store_true", help="Allow fallback to synthetic identity if no real identity/image specified")
    parser.add_argument("--output-dir", type=str, default="outputs/experiments", help="Output directory for generated results and logs")

    args = parser.parse_args()

    print("=" * 70)
    print(" AGEWEAVER: AUTONOMOUS RESEARCH EXPERIMENTS")
    print(" Device: CPU (Intel Iris Xe) | Model: Stable Diffusion v1-5 (Offline)")
    print(f" Scheduler: {args.scheduler} (Karras Sigmas)")
    print(" Evaluator: InsightFace buffalo_l (ArcFace + GenderAge)")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Initialize Pipeline & Evaluator
    from ageweaver.config import PipelineConfig
    pipe_config = PipelineConfig(scheduler_type=args.scheduler, num_inference_steps=args.steps)
    pipeline = AgeWeaverPipeline(config=pipe_config)
    pipeline.load_pipeline()
    runner = ExperimentRunner(output_dir=args.output_dir, pipeline=pipeline)

    # 2. Source Image Resolution (Strict No Silent Fallback)
    source_image = None
    input_id = None
    manifest_entry = None
    dm = DatasetManager()

    if args.identity_id:
        input_id = args.identity_id
        logger.info(f"Target identity specified from manifest: {args.identity_id}")
        try:
            manifest_entry = dm.get_celebrity(args.identity_id)
            img_path = dm.get_image_path(args.identity_id, auto_download=True)
            logger.info(f"Loaded real celebrity image from: {img_path}")
            source_image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to acquire image for identity '{args.identity_id}': {e}")
            logger.error("Strict Policy: NO silent fallback to synthetic identity permitted when --identity-id is given.")
            sys.exit(1)

    elif args.image:
        if not os.path.exists(args.image):
            logger.error(f"Specified image file not found: {args.image}")
            sys.exit(1)
        input_id = os.path.basename(args.image).split(".")[0]
        logger.info(f"Loaded image from user path: {args.image}")
        source_image = Image.open(args.image).convert("RGB")

    elif args.allow_synthetic:
        input_id = "SYNTH_ID_2024"
        source_img_path = os.path.join(args.output_dir, "benchmark_source_identity.png")
        if os.path.exists(source_img_path):
            logger.info(f"Loading existing benchmark synthetic identity from {source_img_path}")
            source_image = Image.open(source_img_path).convert("RGB")
        else:
            logger.info("Generating deterministic synthetic benchmark identity portrait...")
            source_image = pipeline.generate_synthetic_identity(
                prompt="high quality studio portrait photograph of a 25-year-old young person, neutral expression, looking directly at camera, soft studio lighting, sharp facial details, 8k",
                seed=2024,
                steps=18
            )
            save_image(source_image, source_img_path)
    else:
        logger.error(
            "No source identity provided. Please specify --identity-id <ID> (e.g. --identity-id ID001_RAJESH_KHANNA), "
            "--image <path>, or explicitly pass --allow-synthetic to run the synthetic identity benchmark."
        )
        sys.exit(1)

    # 3. Analyze Source Identity
    src_res = runner.analyzer.analyze_image(source_image)
    if not src_res.face_detected:
        logger.error(f"Source face detection failed on {input_id}: {src_res.error_message}")
        sys.exit(1)

    # Determine chronological vs automated age
    cnn_est_age = src_res.estimated_age
    if args.source_age is not None:
        source_age = args.source_age
        age_source_note = f"Explicit CLI argument: {source_age}"
    elif manifest_entry and manifest_entry.reference_age is not None:
        source_age = manifest_entry.reference_age
        age_source_note = f"Documented historical reference age: {source_age} (Capture Year: {manifest_entry.capture_year})"
    else:
        source_age = int(round(cnn_est_age)) if cnn_est_age else 25
        age_source_note = f"Automated CNN estimate: {source_age}"

    gender = (manifest_entry.gender if manifest_entry and manifest_entry.gender else (src_res.gender or "person"))

    logger.info(f"Source Subject: {input_id}")
    logger.info(f"  Chronological / Reference Age : {source_age} ({age_source_note})")
    logger.info(f"  Automated CNN Estimate        : {cnn_est_age:.1f} yrs (genderage.onnx)")
    logger.info(f"  Gender Hint                   : {gender}")
    logger.info(f"  Detection Confidence          : {src_res.det_score:.3f}")

    # 4. Controlled Experiment Matrix
    target_ages = [int(a.strip()) for a in args.target_ages.split(",") if a.strip()]

    # Validate that every target age is older than input age
    for tgt_age in target_ages:
        if tgt_age <= source_age:
            raise ValueError(
                f"Experimental Design Violation: Target age ({tgt_age}) must be strictly older "
                f"than verified source age ({source_age}) for age-progression research."
            )

    conditions = ["generic_baseline", "structured_prompt", "adaptive_control"]
    results = []

    for tgt_age in target_ages:
        for cond in conditions:
            exp_id = f"EXP_{input_id}_{cond}_T{tgt_age}"
            is_cyclic = (cond == "adaptive_control")

            logger.info(f"\n>>> Executing {exp_id} (Target Age: {tgt_age}) <<<")
            trial_res = runner.run_single_trial(
                experiment_id=exp_id,
                condition=cond,
                source_image=source_image,
                input_id=input_id,
                target_age=tgt_age,
                source_age=source_age,
                gender_hint=gender,
                seed=args.seed,
                steps=args.steps,
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
    print(f" EXPERIMENT MATRIX SUMMARY FOR {input_id}")
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
    print(f"Logs updated at:\n - {runner.csv_log_path}\n - {runner.jsonl_log_path}")

if __name__ == "__main__":
    main()
