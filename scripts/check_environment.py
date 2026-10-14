"""
AgeWeaver - Environment and Cached Resources Inspection Script.
Verifies Python version, key ML packages, hardware capabilities,
and inspects the local Hugging Face cache for Stable Diffusion v1-5.
"""

import os
import sys
import platform
import glob

def print_header(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def check_system():
    print_header("SYSTEM & HARDWARE")
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version}")

def check_packages():
    print_header("PACKAGES & DEPENDENCIES")
    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
        ("accelerate", "Accelerate"),
        ("insightface", "InsightFace"),
        ("onnxruntime", "ONNX Runtime"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("pytest", "Pytest"),
        ("yaml", "PyYAML")
    ]
    for pkg_name, label in packages:
        try:
            mod = __import__(pkg_name)
            ver = getattr(mod, "__version__", "Installed (no __version__)")
            print(f"  [OK] {label:15}: {ver}")
        except ImportError as e:
            print(f"  [MISSING] {label:15}: {e}")

    try:
        import torch
        print("\nPyTorch Specifics:")
        print(f"  CUDA Available     : {torch.cuda.is_available()}")
        print(f"  Torch Threads      : {torch.get_num_threads()}")
        print(f"  Default Device     : cpu")
    except Exception as e:
        print(f"Error checking torch specifics: {e}")

def check_hf_cache():
    print_header("HUGGING FACE MODEL CACHE INSPECTION")
    user_home = os.path.expanduser("~")
    hf_cache_dir = os.environ.get("HF_HOME", os.path.join(user_home, ".cache", "huggingface", "hub"))
    print(f"Looking for HF Hub cache at: {hf_cache_dir}")
    
    if not os.path.exists(hf_cache_dir):
        print("  HF cache directory does not exist at default location.")
        return

    # Check for RunwayML Stable Diffusion 1.5
    sd_dirs = [d for d in os.listdir(hf_cache_dir) if "stable-diffusion-v1-5" in d or "runwayml" in d]
    print(f"Found related directories: {sd_dirs}")
    
    sd_dir = os.path.join(hf_cache_dir, "models--runwayml--stable-diffusion-v1-5")
    if os.path.exists(sd_dir):
        print(f"\nTarget cache directory found: {sd_dir}")
        snapshots_dir = os.path.join(sd_dir, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            print(f"Snapshots found: {snapshots}")
            for snap in snapshots:
                snap_path = os.path.join(snapshots_dir, snap)
                print(f"\nInspecting snapshot: {snap}")
                if os.path.isdir(snap_path):
                    items = os.listdir(snap_path)
                    print(f"  Top-level items: {items}")
                    # Check key components of Stable Diffusion
                    expected = ["unet", "vae", "text_encoder", "tokenizer", "scheduler"]
                    for exp in expected:
                        comp_path = os.path.join(snap_path, exp)
                        exists = os.path.exists(comp_path)
                        status = "EXISTS" if exists else "MISSING"
                        print(f"    - {exp:15}: {status}")
        else:
            print("  No snapshots directory found.")
    else:
        print("  models--runwayml--stable-diffusion-v1-5 was not found in standard cache path.")

def check_insightface_cache():
    print_header("INSIGHTFACE CACHE INSPECTION")
    user_home = os.path.expanduser("~")
    insightface_dir = os.path.join(user_home, ".insightface", "models")
    print(f"Looking for InsightFace cache at: {insightface_dir}")
    if os.path.exists(insightface_dir):
        models = os.listdir(insightface_dir)
        print(f"  InsightFace models found: {models}")
        for m in models:
            sub = os.path.join(insightface_dir, m)
            if os.path.isdir(sub):
                print(f"    {m}: {os.listdir(sub)}")
    else:
        print("  No .insightface/models directory found.")

if __name__ == "__main__":
    check_system()
    check_packages()
    check_hf_cache()
    check_insightface_cache()
    print("\n" + "=" * 60)
    print(" Inspection complete.")
    print("=" * 60)

