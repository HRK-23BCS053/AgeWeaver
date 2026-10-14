"""
Diagnostic script to test offline loading of:
1. Stable Diffusion v1-5 from local Hugging Face cache.
2. InsightFace buffalo_l models for face detection, ArcFace embedding, and age estimation.
"""

import os
import sys
import torch

def test_insightface():
    print("Testing InsightFace loading...")
    try:
        import insightface
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(320, 320))
        print("[SUCCESS] InsightFace FaceAnalysis (buffalo_l) loaded successfully on CPU!")
        return True
    except Exception as e:
        print(f"[FAIL] InsightFace loading failed: {e}")
        return False

def test_stable_diffusion():
    print("\nTesting Stable Diffusion v1-5 offline loading...")
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
        
        # Test loading with local_files_only=True
        model_id = "runwayml/stable-diffusion-v1-5"
        print(f"Loading {model_id} with local_files_only=True...")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            local_files_only=True,
            safety_checker=None  # Can disable or keep depending on memory
        )
        print("[SUCCESS] StableDiffusionImg2ImgPipeline loaded successfully offline!")
        print(f"Device: {pipe.device}")
        return True
    except Exception as e:
        print(f"[FAIL] StableDiffusionImg2ImgPipeline loading failed: {e}")
        return False

if __name__ == "__main__":
    if_ok = test_insightface()
    sd_ok = test_stable_diffusion()
    print("\n" + "=" * 50)
    print(f"Summary: InsightFace: {if_ok}, StableDiffusion: {sd_ok}")
    print("=" * 50)

