"""
AgeWeaver - Self-Contained GPU/CPU Benchmark Runner
Optimized for Google Colab GPU (T4 / A100) with automatic CPU fallback.
Executes multi-age progression and regression benchmark across target ages [2, 12, 35, 50, 70]
on validated reference subjects, custom uploads, and synthetic baselines.
Generates structured CSV/JSONL records and composite comparative visual grids.
"""

import os
import sys
import argparse
import time
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont

# Add project root and src to sys.path
sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("."))

import torch
from ageweaver.config import PipelineConfig, AgeWeaverConfig
from ageweaver.pipeline import AgeWeaverPipeline, classify_transformation
from ageweaver.identity import FaceIdentityAnalyzer
from ageweaver.evaluation import AgeWeaverEvaluator
from ageweaver.experiment import ExperimentRunner
from ageweaver.dataset import DatasetManager
from ageweaver.utils import save_image, setup_logger, compute_image_sha256

logger = setup_logger("colab_benchmark")

def parse_args():
    parser = argparse.ArgumentParser(description="AgeWeaver Colab / High-Speed GPU Benchmark Runner")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Execution device: auto (detects CUDA), cuda, or cpu (default: auto)")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Enable FP16 precision on CUDA devices for 3x speedup and memory efficiency")
    parser.add_argument("--no-fp16", action="store_false", dest="fp16",
                        help="Disable FP16 precision (force FP32)")
    parser.add_argument("--target-ages", type=str, default="2,12,35,50,70",
                        help="Comma-separated target ages (default: '2,12,35,50,70')")
    parser.add_argument("--identities", type=str, default="ID001_RAJESH_KHANNA,ID009_DEEPIKA_PADUKONE",
                        help="Comma-separated manifest IDs or 'all_validated' (default: 'ID001_RAJESH_KHANNA,ID009_DEEPIKA_PADUKONE')")
    parser.add_argument("--custom-images", type=str, default="",
                        help="Comma-separated paths to custom input images (e.g. 'data/uploads/HRK.jpeg')")
    parser.add_argument("--include-synthetic", action="store_true", default=False,
                        help="Include synthetic deterministic reference identity in benchmark")
    parser.add_argument("--cyclic", action="store_true", default=False,
                        help="Execute cyclic round-trip reconstruction (doubles inference steps)")
    parser.add_argument("--conditions", type=str, default="adaptive_control",
                        help="Comma-separated conditions: 'adaptive_control', 'structured_prompt', 'generic_baseline' (default: 'adaptive_control')")
    parser.add_argument("--steps", type=int, default=15,
                        help="Diffusion inference steps (default: 15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic reproducibility (default: 42)")
    parser.add_argument("--scheduler", type=str, default="dpmsolver++",
                        choices=["dpmsolver++", "euler", "ddim", "pndm"],
                        help="Diffusion scheduler algorithm (default: 'dpmsolver++')")
    parser.add_argument("--output-dir", type=str, default="outputs/colab_benchmark",
                        help="Output directory for generated portraits, grids, and metrics")
    parser.add_argument("--save-grids", action="store_true", default=True,
                        help="Generate composite side-by-side progression/regression grids")
    return parser.parse_args()


def create_comparative_grid(
    source_image: Image.Image,
    source_label: str,
    results_by_age: Dict[int, Dict[str, Any]],
    output_grid_path: str,
    title: str = "AgeWeaver Facial Age Transformation Matrix"
) -> Image.Image:
    """
    Constructs a publication-ready horizontal comparative grid:
    [Input Face] | [Age 2] | [Age 12] | [Age 35] | [Age 50] | [Age 70]
    Includes annotations for identity similarity and estimated output age.
    """
    sorted_ages = sorted(results_by_age.keys())
    cells = []

    # 1. Input Image Cell
    src_cell = source_image.copy().convert("RGB").resize((384, 384), Image.Resampling.LANCZOS)
    cells.append({
        "image": src_cell,
        "header": "INPUT SOURCE",
        "sub": source_label,
        "metrics": "Reference Identity"
    })

    # 2. Transformed Age Cells
    for age in sorted_ages:
        res = results_by_age[age]
        gen_img = res.get("image")
        if gen_img is None and "output_path" in res and os.path.exists(res["output_path"]):
            gen_img = Image.open(res["output_path"]).convert("RGB")

        if gen_img is not None:
            c_img = gen_img.copy().convert("RGB").resize((384, 384), Image.Resampling.LANCZOS)
        else:
            c_img = Image.new("RGB", (384, 384), color=(30, 30, 30))

        direction = res.get("direction", "transformation").upper()
        est_age = res.get("generated_estimated_age")
        sim = res.get("identity_similarity")
        mae = res.get("target_age_mae")

        est_str = f"Est: {est_age:.1f}y (MAE {mae:.1f})" if est_age is not None and mae is not None else "Est: N/A"
        sim_str = f"ID-Sim: {sim:.3f}" if sim is not None else "ID-Sim: N/A"

        cells.append({
            "image": c_img,
            "header": f"TARGET {age}",
            "sub": f"[{direction}]",
            "metrics": f"{est_str} | {sim_str}"
        })

    # Build composite canvas
    n_cols = len(cells)
    cell_w, cell_h = 384, 384
    header_h = 60
    footer_h = 50
    total_cell_h = cell_h + header_h + footer_h
    banner_h = 60

    canvas_w = n_cols * cell_w + (n_cols + 1) * 16
    canvas_h = total_cell_h + banner_h + 20

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(15, 17, 26))
    draw = ImageDraw.Draw(canvas)

    # Title Banner
    draw.rectangle([(0, 0), (canvas_w, banner_h)], fill=(22, 27, 46))
    draw.text((24, 18), title, fill=(240, 240, 255))

    for idx, cell in enumerate(cells):
        x = 16 + idx * (cell_w + 16)
        y = banner_h + 16

        # Cell background card
        draw.rectangle([(x - 2, y - 2), (x + cell_w + 2, y + total_cell_h + 2)], fill=(28, 32, 54), outline=(48, 54, 86))

        # Header text
        draw.text((x + 12, y + 8), cell["header"], fill=(139, 92, 246))
        draw.text((x + 12, y + 32), cell["sub"], fill=(160, 170, 190))

        # Paste Image
        canvas.paste(cell["image"], (x, y + header_h))

        # Footer Metrics text
        draw.text((x + 12, y + header_h + cell_h + 14), cell["metrics"], fill=(0, 229, 255))

    os.makedirs(os.path.dirname(output_grid_path), exist_ok=True)
    canvas.save(output_grid_path, quality=95)
    logger.info(f"Composite grid saved: {output_grid_path}")
    return canvas


def main():
    args = parse_args()

    # Determine execution device & dtype
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    use_fp16 = args.fp16 and (device == "cuda")
    torch_dtype = "float16" if use_fp16 else "float32"

    print("=" * 76)
    print(" AGEWEAVER: COMPREHENSIVE MULTI-AGE RESEARCH BENCHMARK")
    print(f" Target Ages      : {args.target_ages}")
    print(f" Device Config    : {device.upper()} (FP16: {use_fp16}, Dtype: {torch_dtype})")
    print(f" Scheduler        : {args.scheduler} (Karras Sigmas, Steps: {args.steps})")
    print(f" Output Directory : {args.output_dir}")
    print("=" * 76)

    os.makedirs(args.output_dir, exist_ok=True)
    grids_dir = os.path.join(args.output_dir, "grids")
    os.makedirs(grids_dir, exist_ok=True)

    # 1. Initialize Pipeline Config
    pipe_config = PipelineConfig(
        device=device,
        torch_dtype=torch_dtype,
        local_files_only=False,  # Allow download on Colab or fresh setups
        num_inference_steps=args.steps,
        scheduler_type=args.scheduler,
        seed=args.seed
    )

    pipeline = AgeWeaverPipeline(config=pipe_config)
    pipeline.load_pipeline()

    evaluator = AgeWeaverEvaluator()
    runner = ExperimentRunner(output_dir=args.output_dir, pipeline=pipeline, evaluator=evaluator)
    dm = DatasetManager()

    # 2. Parse Subjects to Benchmark
    target_ages = [int(a.strip()) for a in args.target_ages.split(",") if a.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    subjects_to_run = []

    # Manifest identities
    if args.identities:
        identities_list = [i.strip() for i in args.identities.split(",") if i.strip()]
        for ident_id in identities_list:
            try:
                m_entry = dm.get_celebrity(ident_id)
                img_path = dm.get_image_path(ident_id, auto_download=True)
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert("RGB")
                    subjects_to_run.append({
                        "id": ident_id,
                        "image": img,
                        "source_age": m_entry.reference_age if m_entry else 28,
                        "gender": m_entry.gender if m_entry else "person",
                        "label": f"{ident_id} (Ref Age: {m_entry.reference_age if m_entry else 'Unknown'})"
                    })
                    logger.info(f"Loaded manifest subject: {ident_id} from {img_path}")
            except Exception as e:
                logger.warning(f"Could not load manifest subject {ident_id}: {e}")

    # Custom upload images
    if args.custom_images:
        for c_path in [p.strip() for p in args.custom_images.split(",") if p.strip()]:
            if os.path.exists(c_path):
                c_img = Image.open(c_path).convert("RGB")
                c_id = os.path.splitext(os.path.basename(c_path))[0]
                subjects_to_run.append({
                    "id": f"UPLOAD_{c_id}",
                    "image": c_img,
                    "source_age": 19,  # Documented young reference age for HRK
                    "gender": "male",
                    "label": f"Custom Upload: {c_id} (Ref Age: 19)"
                })
                logger.info(f"Loaded custom upload: {c_id} from {c_path}")
            else:
                logger.warning(f"Custom image path not found: {c_path}")

    # Synthetic baseline
    if args.include_synthetic:
        synth_cache = os.path.join(args.output_dir, "benchmark_source_identity.png")
        if os.path.exists(synth_cache):
            s_img = Image.open(synth_cache).convert("RGB")
        else:
            logger.info("Generating synthetic 25-year-old baseline identity...")
            s_img = pipeline.generate_synthetic_identity(
                prompt="clean close-up studio portrait photograph of a 25-year-old person, neutral expression, centered face, professional lighting, photorealistic, 8k",
                seed=2024,
                steps=18
            )
            save_image(s_img, synth_cache)

        subjects_to_run.append({
            "id": "SYNTH_ID_2024",
            "image": s_img,
            "source_age": 25,
            "gender": "person",
            "label": "Synthetic Baseline (Ref Age: 25)"
        })
        logger.info("Loaded synthetic baseline identity.")

    if not subjects_to_run:
        logger.error("No valid subjects found to run. Check manifest or image paths.")
        sys.exit(1)

    print(f"\n[INFO] Commencing benchmark across {len(subjects_to_run)} subjects and {len(target_ages)} target ages.")
    total_runs = len(subjects_to_run) * len(target_ages) * len(conditions)
    print(f"[INFO] Total planned forward trials: {total_runs}\n")

    all_benchmark_results = []

    # 3. Main Benchmark Execution Loop
    for subj in subjects_to_run:
        s_id = subj["id"]
        s_img = subj["image"]
        src_age = subj["source_age"]
        gender = subj["gender"]
        label = subj["label"]

        print("=" * 76)
        print(f" PROCESSING SUBJECT: {s_id} (Source Age: {src_age})")
        print("=" * 76)

        results_by_age = {}

        for cond in conditions:
            for tgt_age in target_ages:
                dir_label = classify_transformation(src_age, tgt_age)
                exp_id = f"BM_{s_id}_{cond}_T{tgt_age}"
                is_cyclic = args.cyclic and (cond == "adaptive_control")

                logger.info(f">>> Running {exp_id} | {dir_label.upper()} ({src_age} -> {tgt_age}) | Cond: {cond} <<<")
                trial_res = runner.run_single_trial(
                    experiment_id=exp_id,
                    condition=cond,
                    source_image=s_img,
                    input_id=s_id,
                    target_age=tgt_age,
                    source_age=src_age,
                    gender_hint=gender,
                    seed=args.seed,
                    steps=args.steps,
                    execute_cyclic=is_cyclic
                )
                trial_res["direction"] = dir_label
                all_benchmark_results.append(trial_res)

                # Track primary condition results for the visual grid
                if cond == conditions[0]:
                    # Load generated image for grid
                    out_path = trial_res.get("output_path")
                    gen_img = Image.open(out_path).convert("RGB") if (out_path and os.path.exists(out_path)) else None
                    results_by_age[tgt_age] = {
                        **trial_res,
                        "image": gen_img
                    }

        # 4. Generate Comparative Grid for this Subject
        if args.save_grids and results_by_age:
            grid_filename = f"GRID_{s_id}_{conditions[0]}_MultiAge.png"
            grid_path = os.path.join(grids_dir, grid_filename)
            create_comparative_grid(
                source_image=s_img,
                source_label=label,
                results_by_age=results_by_age,
                output_grid_path=grid_path,
                title=f"AgeWeaver ({conditions[0].upper()}) | Subject: {s_id} (Source Age: {src_age})"
            )

    # 5. Final Reporting
    print("\n" + "=" * 76)
    print(" BENCHMARK COMPLETED SUCCESSFULLY")
    print(f" Total Trials Run : {len(all_benchmark_results)}")
    print(f" CSV Log Path     : {runner.csv_log_path}")
    print(f" JSONL Log Path   : {runner.jsonl_log_path}")
    print(f" Visual Grids     : {grids_dir}")
    print("=" * 76)

if __name__ == "__main__":
    main()
