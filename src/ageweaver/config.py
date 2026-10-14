"""
Configuration module for AgeWeaver.
Provides structured configuration models, validation, and YAML serialization.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import os
import yaml

@dataclass
class PromptConfig:
    """Configuration for prompt construction."""
    mode: str = "structured"  # "generic" or "structured"
    include_hair_markers: bool = True
    include_skin_markers: bool = True
    include_identity_anchors: bool = True
    custom_positive_suffix: str = ""
    negative_prompt: str = (
        "deformed, distorted, disfigured, blurry, bad anatomy, cartoon, 3d render, "
        "illustration, plastic skin, oversmoothed, low quality, unnatural eyes, "
        "different facial structure, changed ethnicity"
    )

@dataclass
class PipelineConfig:
    """Configuration for diffusion inference pipeline."""
    model_id: str = "runwayml/stable-diffusion-v1-5"
    device: str = "cpu"
    local_files_only: bool = True
    num_inference_steps: int = 15  # CPU-friendly default
    guidance_scale: float = 7.5
    default_strength: float = 0.45
    image_size: int = 512
    seed: int = 42
    use_adaptive_control: bool = True
    adaptive_strength_min: float = 0.32
    adaptive_strength_max: float = 0.58
    guidance_min: float = 6.5
    guidance_max: float = 8.5

@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    insightface_model: str = "buffalo_l"
    det_size: int = 320
    compute_cyclic: bool = True
    cyclic_steps: int = 15
    compute_age_mae: bool = True

@dataclass
class AgeWeaverConfig:
    """Master configuration for AgeWeaver experiment runs."""
    project_name: str = "AgeWeaver"
    version: str = "0.1.0"
    output_dir: str = "outputs"
    data_dir: str = "data"
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        """Validate parameter ranges and paths."""
        if self.pipeline.image_size % 8 != 0 or self.pipeline.image_size <= 0:
            raise ValueError(f"image_size must be a positive multiple of 8, got {self.pipeline.image_size}")
        if not (0.0 <= self.pipeline.default_strength <= 1.0):
            raise ValueError(f"default_strength must be in [0.0, 1.0], got {self.pipeline.default_strength}")
        if self.pipeline.num_inference_steps <= 0:
            raise ValueError(f"num_inference_steps must be positive, got {self.pipeline.num_inference_steps}")
        if self.pipeline.guidance_scale <= 0.0:
            raise ValueError(f"guidance_scale must be positive, got {self.pipeline.guidance_scale}")
        if self.prompt.mode not in ("generic", "structured"):
            raise ValueError(f"prompt mode must be 'generic' or 'structured', got '{self.prompt.mode}'")

    @classmethod
    def from_yaml(cls, path: str) -> "AgeWeaverConfig":
        """Load configuration from a YAML file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        pipeline_data = data.get("pipeline", {})
        prompt_data = data.get("prompt", {})
        eval_data = data.get("evaluation", {})

        cfg = cls(
            project_name=data.get("project_name", "AgeWeaver"),
            version=data.get("version", "0.1.0"),
            output_dir=data.get("output_dir", "outputs"),
            data_dir=data.get("data_dir", "data"),
            pipeline=PipelineConfig(**pipeline_data),
            prompt=PromptConfig(**prompt_data),
            evaluation=EvaluationConfig(**eval_data),
        )
        cfg.validate()
        return cfg

    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "project_name": self.project_name,
            "version": self.version,
            "output_dir": self.output_dir,
            "data_dir": self.data_dir,
            "pipeline": asdict(self.pipeline),
            "prompt": asdict(self.prompt),
            "evaluation": asdict(self.evaluation),
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

