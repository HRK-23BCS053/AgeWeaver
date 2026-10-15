"""
Diffusion inference pipeline for AgeWeaver.
Provides offline-capable, CPU-optimized image-to-image facial age progression
with adaptive strength/guidance control and prompt refinement.
"""

import os
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple
from PIL import Image
import torch
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    DDIMScheduler,
    PNDMScheduler
)

from ageweaver.config import PipelineConfig, PromptConfig
from ageweaver.prompts import (
    build_generic_prompt,
    build_structured_prompt,
    build_negative_prompt
)
from ageweaver.utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class GenerationResult:
    """Stores the generated image and all associated experiment metadata."""
    generated_image: Image.Image
    prompt: str
    negative_prompt: str
    strength: float
    guidance_scale: float
    num_inference_steps: int
    seed: int
    elapsed_time_sec: float
    target_age: int
    source_age: Optional[int] = None
    mode: str = "structured"
    scheduler: str = "dpmsolver++"

class AgeWeaverPipeline:
    """
    Age progression pipeline built on Stable Diffusion v1-5.
    Configured for local offline CPU execution.
    """
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._pipe: Optional[StableDiffusionImg2ImgPipeline] = None
        self._text2img_pipe: Optional[StableDiffusionPipeline] = None

    def load_pipeline(self) -> None:
        """Loads the pre-trained diffusion model from local cache."""
        if self._pipe is not None:
            return

        logger.info(f"Loading diffusion model from {self.config.model_id} (local_files_only={self.config.local_files_only})...")
        t0 = time.time()
        
        # Load img2img pipeline
        self._pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.config.model_id,
            dtype=torch.float32,
            local_files_only=self.config.local_files_only,
            safety_checker=None
        )
        self._pipe.to(self.config.device)
        self._pipe.set_progress_bar_config(disable=True)  # Clean logging

        # Configure scheduler for robust artifact-free short-step img2img inference
        sched_type = getattr(self.config, "scheduler_type", "dpmsolver++").lower()
        if sched_type == "dpmsolver++":
            self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self._pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True
            )
        elif sched_type == "euler":
            self._pipe.scheduler = EulerDiscreteScheduler.from_config(
                self._pipe.scheduler.config
            )
        elif sched_type == "ddim":
            self._pipe.scheduler = DDIMScheduler.from_config(
                self._pipe.scheduler.config
            )
        elif sched_type == "pndm":
            self._pipe.scheduler = PNDMScheduler.from_config(
                self._pipe.scheduler.config
            )
        else:
            logger.warning(f"Unrecognized scheduler_type '{sched_type}', defaulting to DPMSolverMultistepScheduler.")
            self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self._pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True
            )
        logger.info(f"Initialized diffusion pipeline with scheduler: {self._pipe.scheduler.__class__.__name__}")
        
        load_time = time.time() - t0
        logger.info(f"Model loaded successfully on {self.config.device} in {load_time:.2f}s")

    def compute_adaptive_parameters(
        self,
        source_age: Optional[int],
        target_age: int
    ) -> Tuple[float, float]:
        """
        Calculates adaptive denoising strength and guidance scale based on the age delta.
        Hypothesis:
        - Small age transitions require low strength (preserving fine identity markers).
        - Large age transitions require higher strength to introduce genuine structural aging.
        """
        if not self.config.use_adaptive_control or source_age is None:
            return self.config.default_strength, self.config.guidance_scale

        delta = abs(target_age - source_age)
        norm_delta = min(1.0, delta / 50.0)  # Max scale over 50 years difference

        strength = self.config.adaptive_strength_min + norm_delta * (
            self.config.adaptive_strength_max - self.config.adaptive_strength_min
        )
        guidance = self.config.guidance_min + norm_delta * (
            self.config.guidance_max - self.config.guidance_min
        )
        return float(round(strength, 3)), float(round(guidance, 2))

    def generate_synthetic_identity(
        self,
        prompt: str = "clean close-up studio portrait of a 25-year-old person, neutral expression, centered face, professional lighting, photorealistic, 8k",
        seed: int = 42,
        steps: int = 20
    ) -> Image.Image:
        """
        Generates a synthetic reference face using text-to-image to provide
        a completely licensed, privacy-safe, reproducible identity baseline.
        """
        logger.info(f"Generating synthetic source identity with seed {seed}...")
        self.load_pipeline()
        
        # Share components with text2img pipeline to conserve RAM
        if self._text2img_pipe is None:
            self._text2img_pipe = StableDiffusionPipeline(
                vae=self._pipe.vae,
                text_encoder=self._pipe.text_encoder,
                tokenizer=self._pipe.tokenizer,
                unet=self._pipe.unet,
                scheduler=self._pipe.scheduler,
                safety_checker=None,
                feature_extractor=None
            )
            self._text2img_pipe.to(self.config.device)
            self._text2img_pipe.set_progress_bar_config(disable=True)

        generator = torch.Generator(device=self.config.device).manual_seed(seed)
        neg_prompt = build_negative_prompt()
        
        with torch.inference_mode():
            out = self._text2img_pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=steps,
                guidance_scale=7.5,
                height=self.config.image_size,
                width=self.config.image_size,
                generator=generator
            )
        img = out.images[0]
        logger.info("Synthetic source identity generated successfully.")
        return img

    def progress_age(
        self,
        source_image: Image.Image,
        target_age: int,
        source_age: Optional[int] = None,
        gender_hint: Optional[str] = None,
        mode: str = "structured",
        strength: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        custom_negative: Optional[str] = None
    ) -> GenerationResult:
        """
        Executes facial age progression on source_image to target_age.
        Supports 'generic' baseline and 'structured' refinement modes.
        """
        self.load_pipeline()
        
        # Resolve parameters
        actual_seed = seed if seed is not None else self.config.seed
        steps = num_inference_steps if num_inference_steps is not None else self.config.num_inference_steps
        
        # Determine strength and guidance
        if strength is None or guidance_scale is None:
            if mode == "structured" and self.config.use_adaptive_control and source_age is not None:
                calc_strength, calc_guidance = self.compute_adaptive_parameters(source_age, target_age)
                actual_strength = strength if strength is not None else calc_strength
                actual_guidance = guidance_scale if guidance_scale is not None else calc_guidance
            else:
                actual_strength = strength if strength is not None else self.config.default_strength
                actual_guidance = guidance_scale if guidance_scale is not None else self.config.guidance_scale
        else:
            actual_strength = strength
            actual_guidance = guidance_scale

        # Build prompts
        if mode == "generic":
            prompt = build_generic_prompt(target_age=target_age, gender_hint=gender_hint)
        else:
            prompt = build_structured_prompt(
                target_age=target_age,
                source_age=source_age,
                gender_hint=gender_hint
            )

        negative_prompt = build_negative_prompt(custom_negative)

        # Prepare generator
        generator = torch.Generator(device=self.config.device).manual_seed(actual_seed)

        # Ensure image is in proper RGB dimensions
        target_size = (self.config.image_size, self.config.image_size)
        if source_image.size != target_size:
            source_image = source_image.resize(target_size, Image.Resampling.LANCZOS)
        source_image = source_image.convert("RGB")

        logger.info(
            f"Running age progression ({mode}) | Age: {source_age} -> {target_age} | "
            f"Steps: {steps} | Strength: {actual_strength:.3f} | Guidance: {actual_guidance:.2f} | Seed: {actual_seed}"
        )
        
        t0 = time.time()
        with torch.inference_mode():
            result = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=source_image,
                strength=actual_strength,
                guidance_scale=actual_guidance,
                num_inference_steps=steps,
                generator=generator
            )
        elapsed = time.time() - t0
        logger.info(f"Inference completed in {elapsed:.2f}s")

        return GenerationResult(
            generated_image=result.images[0],
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=actual_strength,
            guidance_scale=actual_guidance,
            num_inference_steps=steps,
            seed=actual_seed,
            elapsed_time_sec=round(elapsed, 2),
            target_age=target_age,
            source_age=source_age,
            mode=mode,
            scheduler=getattr(self.config, "scheduler_type", "dpmsolver++")
        )

