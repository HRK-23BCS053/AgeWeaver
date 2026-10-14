"""
Controlled experiment execution harness for AgeWeaver.
Runs controlled trials comparing:
- Experiment A: Generic prompt baseline.
- Experiment B: Structured age-aware prompt refinement.
- Experiment C: Proposed AgeWeaver adaptive controller (adaptive strength + structured prompt + cyclic consistency).
Records auditable machine-readable logs in CSV and JSONL formats.
"""

import os
import json
import csv
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from PIL import Image

from ageweaver.pipeline import AgeWeaverPipeline, GenerationResult
from ageweaver.identity import FaceIdentityAnalyzer
from ageweaver.evaluation import AgeWeaverEvaluator, EvaluationMetrics
from ageweaver.utils import compute_image_sha256, save_image, setup_logger

logger = setup_logger(__name__)

class ExperimentRunner:
    """
    Executes controlled comparative experiments and logs auditable results.
    """
    def __init__(
        self,
        output_dir: str = "outputs/experiments",
        pipeline: Optional[AgeWeaverPipeline] = None,
        evaluator: Optional[AgeWeaverEvaluator] = None
    ):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.pipeline = pipeline or AgeWeaverPipeline()
        self.analyzer = FaceIdentityAnalyzer()
        self.evaluator = evaluator or AgeWeaverEvaluator(self.analyzer)
        self.csv_log_path = os.path.join(self.output_dir, "experiment_log.csv")
        self.jsonl_log_path = os.path.join(self.output_dir, "experiment_log.jsonl")
        self._init_logs()

    def _init_logs(self):
        """Initializes CSV header if file doesn't already exist."""
        if not os.path.exists(self.csv_log_path):
            fieldnames = [
                "experiment_id", "timestamp", "input_id", "input_sha256",
                "condition", "source_age", "target_age", "seed", "steps",
                "strength", "guidance_scale", "runtime_sec", "success",
                "identity_similarity", "source_estimated_age", "generated_estimated_age",
                "target_age_mae", "cyclic_similarity", "output_path", "prompt"
            ]
            with open(self.csv_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def log_record(self, record: Dict[str, Any]):
        """Appends experiment result to CSV and JSONL logs."""
        # JSONL
        with open(self.jsonl_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

        # CSV
        fieldnames = [
            "experiment_id", "timestamp", "input_id", "input_sha256",
            "condition", "source_age", "target_age", "seed", "steps",
            "strength", "guidance_scale", "runtime_sec", "success",
            "identity_similarity", "source_estimated_age", "generated_estimated_age",
            "target_age_mae", "cyclic_similarity", "output_path", "prompt"
        ]
        csv_row = {k: record.get(k, "") for k in fieldnames}
        with open(self.csv_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(csv_row)

    def run_single_trial(
        self,
        experiment_id: str,
        condition: str,  # "generic_baseline", "structured_prompt", "adaptive_control"
        source_image: Image.Image,
        input_id: str,
        target_age: int,
        source_age: Optional[int] = None,
        seed: int = 42,
        steps: int = 15,
        execute_cyclic: bool = False
    ) -> Dict[str, Any]:
        """
        Runs a single controlled trial and computes all metrics.
        """
        logger.info(f"Starting Trial: {experiment_id} | Condition: {condition} | Input: {input_id} | Age: {source_age}->{target_age}")
        timestamp = datetime.now().isoformat()
        input_hash = compute_image_sha256(source_image)

        # Map condition to generation parameters
        if condition == "generic_baseline":
            mode = "generic"
            strength = 0.45
            guidance = 7.5
        elif condition == "structured_prompt":
            mode = "structured"
            strength = 0.45
            guidance = 7.5
        elif condition == "adaptive_control":
            mode = "structured"
            # Pipeline automatically computes adaptive strength & guidance based on delta
            strength = None
            guidance = None
        else:
            raise ValueError(f"Unknown condition: {condition}")

        success = False
        metrics = None
        gen_result = None
        output_rel_path = ""
        cyclic_img = None

        try:
            # 1. Forward progression
            gen_result = self.pipeline.progress_age(
                source_image=source_image,
                target_age=target_age,
                source_age=source_age,
                mode=mode,
                strength=strength,
                guidance_scale=guidance,
                num_inference_steps=steps,
                seed=seed
            )

            # Save generated image
            img_filename = f"{experiment_id}_{condition}_age{target_age}_seed{seed}.png"
            output_abs_path = os.path.join(self.output_dir, img_filename)
            save_image(gen_result.generated_image, output_abs_path)
            output_rel_path = os.path.relpath(output_abs_path, os.getcwd())

            # 2. Optional Cyclic round-trip reconstruction (FaceTT Section 5.4)
            if execute_cyclic and source_age is not None:
                logger.info(f"Executing cyclic round-trip reconstruction back to source age {source_age}...")
                cyc_result = self.pipeline.progress_age(
                    source_image=gen_result.generated_image,
                    target_age=source_age,
                    source_age=target_age,
                    mode=mode,
                    strength=gen_result.strength,
                    guidance_scale=gen_result.guidance_scale,
                    num_inference_steps=steps,
                    seed=seed + 1
                )
                cyclic_img = cyc_result.generated_image
                cyc_filename = f"{experiment_id}_{condition}_cyclic_recon_age{source_age}.png"
                save_image(cyclic_img, os.path.join(self.output_dir, cyc_filename))

            # 3. Quantitative Evaluation
            metrics = self.evaluator.evaluate(
                source_image=source_image,
                generated_image=gen_result.generated_image,
                target_age=target_age,
                source_age=source_age,
                cyclic_reconstructed_image=cyclic_img
            )
            success = True

        except Exception as e:
            logger.error(f"Trial {experiment_id} failed with error: {e}", exc_info=True)
            success = False

        record = {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "input_id": input_id,
            "input_sha256": input_hash,
            "condition": condition,
            "source_age": source_age,
            "target_age": target_age,
            "seed": seed,
            "steps": steps,
            "strength": gen_result.strength if gen_result else strength,
            "guidance_scale": gen_result.guidance_scale if gen_result else guidance,
            "runtime_sec": gen_result.elapsed_time_sec if gen_result else 0.0,
            "success": success,
            "identity_similarity": metrics.identity_similarity if metrics else None,
            "source_estimated_age": metrics.source_estimated_age if metrics else None,
            "generated_estimated_age": metrics.generated_estimated_age if metrics else None,
            "target_age_mae": metrics.age_mae if metrics else None,
            "cyclic_similarity": metrics.cyclic_similarity if metrics else None,
            "output_path": output_rel_path,
            "prompt": gen_result.prompt if gen_result else ""
        }

        self.log_record(record)
        return record

