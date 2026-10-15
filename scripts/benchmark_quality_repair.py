"""
Quality Repair Benchmark Script for AgeWeaver.
Executes controlled quality benchmark on >=3 portraits across >=2 target ages using
the repaired DPMSolverMultistep (Karras) diffusion pipeline.
Measures ArcFace identity preservation, CNN estimated age, target-age MAE, cyclic consistency, and CPU latency.
"""

import os
import sys
import time
from PIL import Image

sys.path.insert(0, os.path.abspath("src"))

from ageweaver.config import AgeWeaverConfig, PipelineConfig
from ageweaver.pipeline import AgeWeaverPipeline
from ageweaver.experiment import ExperimentRunner
from ageweaver.evaluation import AgeWeaverEvaluator
from ageweaver.identity import FaceIdentityAnalyzer
from ageweaver.dataset import DatasetManager

def run_benchmarks():
    print("=" * 70, flush=True)
    print(" AGEWEAVER: SYSTEM-WIDE REPAIR QUALITY BENCHMARK", flush=True)
    print("=" * 70, flush=True)

    config = PipelineConfig(
        scheduler_type="dpmsolver++",
        num_inference_steps=15,
        guidance_scale=7.5,
        default_strength=0.45
    )
    pipeline = AgeWeaverPipeline(config=config)
    pipeline.load_pipeline()

    runner = ExperimentRunner(pipeline=pipeline)

    trials_to_run = [
        # 1. Custom Upload Reference (HRK)
        {
            "id": "UPLOAD_HRK_ea7e3a97",
            "image_path": "outputs/experiments/UPLOAD_HRK_ea7e3a97_source_reference.png",
            "source_age": 19,
            "gender_hint": "male",
            "target_ages": [30, 45],
            "execute_cyclic": True  # Measure repaired cyclic round-trip consistency
        },
        # 2. Celebrity Reference 1 (Rajesh Khanna, young archival)
        {
            "id": "ID001_RAJESH_KHANNA",
            "image_path": "data/images/ID001_RAJESH_KHANNA.jpg",
            "source_age": 28,
            "gender_hint": "male",
            "target_ages": [45, 60],
            "execute_cyclic": False
        },
        # 3. Celebrity Reference 2 (Deepika Padukone, young archival)
        {
            "id": "ID009_DEEPIKA_PADUKONE",
            "image_path": "data/images/ID009_DEEPIKA_PADUKONE.jpg",
            "source_age": 22,
            "gender_hint": "female",
            "target_ages": [35, 50],
            "execute_cyclic": False
        },
        # 4. Synthetic Identity (SYNTH_ID_2024)
        {
            "id": "SYNTH_ID_2024",
            "image_path": "outputs/experiments/benchmark_source_identity.png",
            "source_age": 23,
            "gender_hint": "person",
            "target_ages": [45, 68],
            "execute_cyclic": False
        }
    ]

    all_records = []

    for item in trials_to_run:
        input_id = item["id"]
        img_path = item["image_path"]
        src_age = item["source_age"]
        gender = item["gender_hint"]
        target_ages = item["target_ages"]
        exec_cyclic = item["execute_cyclic"]

        if not os.path.exists(img_path):
            print(f"Skipping {input_id}: File not found at {img_path}", flush=True)
            continue

        source_img = Image.open(img_path).convert("RGB")
        print(f"\n>>> Running Benchmark for Subject: {input_id} (Source Age: {src_age}, Gender: {gender})", flush=True)

        for tgt_age in target_ages:
            exp_id = f"RUN_REPAIRED_{input_id}_adaptive_control_T{tgt_age}"
            print(f"\n--- Trial: {exp_id} (Target Age: {tgt_age}) ---", flush=True)
            t_start = time.time()
            record = runner.run_single_trial(
                experiment_id=exp_id,
                condition="adaptive_control",
                source_image=source_img,
                input_id=input_id,
                target_age=tgt_age,
                source_age=src_age,
                gender_hint=gender,
                seed=42,
                steps=15,
                execute_cyclic=exec_cyclic
            )
            elapsed = time.time() - t_start

            all_records.append(record)
            print(f"Completed in {elapsed:.1f}s | Success: {record['success']}", flush=True)
            print(f"  ArcFace Identity Sim   : {record['identity_similarity']}", flush=True)
            print(f"  Output Estimated Age   : {record['generated_estimated_age']}", flush=True)
            print(f"  Target Age Error (MAE) : {record['target_age_mae']}", flush=True)
            print(f"  Cyclic Similarity      : {record['cyclic_similarity']}", flush=True)
            print(f"  Output Image Saved     : {record['output_path']}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(" ALL QUALITY BENCHMARK TRIALS COMPLETED", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    run_benchmarks()
