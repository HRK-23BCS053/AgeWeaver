"""
Utility functions for AgeWeaver:
Image preprocessing, hashing, deterministic seeding, timing, and logging.
"""

import os
import sys
import time
import random
import hashlib
import logging
from typing import Tuple, Optional
import numpy as np
from PIL import Image, ImageOps
import torch

def setup_logger(name: str = "ageweaver", level: int = logging.INFO) -> logging.Logger:
    """Configures a standardized logger with console output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logger()

def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds across Python, NumPy, and PyTorch CPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.debug(f"Deterministic seed set to: {seed}")

def compute_image_sha256(img: Image.Image) -> str:
    """Computes SHA-256 hash of image bytes for auditability and experiment tracking."""
    img_bytes = img.tobytes()
    return hashlib.sha256(img_bytes).hexdigest()[:16]

def load_and_preprocess_image(
    image_path: str,
    target_size: int = 512
) -> Image.Image:
    """
    Loads an image, handles orientation from EXIF, centers and resizes
    to target_size x target_size in RGB mode.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Source image not found: {image_path}")

    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Fit image to square bounding box using high-quality Lanczos resampling
    img = ImageOps.fit(img, (target_size, target_size), Image.Resampling.LANCZOS)
    return img

def save_image(img: Image.Image, output_path: str) -> str:
    """Saves a PIL image ensuring directory structure exists."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img.save(output_path, quality=95)
    return output_path

