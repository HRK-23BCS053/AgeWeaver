"""
Prompt engineering and refinement module for AgeWeaver.
Implements:
1. Generic prompt baseline.
2. Face-attribute-aware structured prompt refinement (incorporating age brackets,
   hair, skin texture, and identity preservation anchors).
"""

from typing import Optional, Dict
from dataclasses import dataclass

AGE_BRACKETS = [
    (0, 19, "adolescent", "youthful smooth skin, soft facial contours", "natural vibrant hair"),
    (20, 34, "young adult", "smooth taut skin, firm jawline, clear complexion", "full natural hair"),
    (35, 49, "mature adult", "subtle laughter lines around eyes, slight nasolabial folds, natural skin texture", "natural hair with slight subtle silver strands at temples"),
    (50, 64, "middle-aged", "visible forehead lines, defined nasolabial folds, fine crow's feet, mature skin texture", "noticeable graying hair, salt and pepper hair"),
    (65, 74, "senior", "pronounced wrinkles, deeper facial creases, visible crow's feet, mature eye contours, textured aging skin", "silver-gray hair, receding hairline or thinning hair"),
    (75, 120, "elderly", "deep prominent wrinkles, pronounced laugh lines, age creases, sagging skin, authentic weathered skin texture", "silver-white hair, thin delicate hair")
]

def get_bracket_info(age: int):
    """Returns (bracket_name, skin_marker, hair_marker) for a given age."""
    for min_age, max_age, name, skin, hair in AGE_BRACKETS:
        if min_age <= age <= max_age:
            return name, skin, hair
    return "adult", "natural mature skin texture", "natural hair"

def build_generic_prompt(target_age: int, gender_hint: Optional[str] = None) -> str:
    """
    Standard generic age-conditioned prompt baseline.
    Represents the naive baseline in literature.
    """
    subject = "person"
    if gender_hint:
        gender_hint_clean = gender_hint.lower().strip()
        if gender_hint_clean in ("male", "man"):
            subject = "man"
        elif gender_hint_clean in ("female", "woman"):
            subject = "woman"

    return f"a portrait photo of a {target_age}-year-old {subject}, realistic photograph, 8k, high quality"

def build_structured_prompt(
    target_age: int,
    source_age: Optional[int] = None,
    gender_hint: Optional[str] = None,
    include_hair: bool = True,
    include_skin: bool = True,
    include_identity_anchor: bool = True,
    custom_suffix: str = ""
) -> str:
    """
    Constructs a structured, face-attribute-aware prompt.
    Incorporates:
    - Target age and age bracket
    - Physiological skin texture markers
    - Hair aging markers
    - Strong identity and geometry preservation anchor
    - High-fidelity photographic context
    """
    subject = "individual"
    if gender_hint:
        gh = gender_hint.lower().strip()
        if gh in ("male", "man"):
            subject = "man"
        elif gh in ("female", "woman"):
            subject = "woman"

    bracket_name, skin_marker, hair_marker = get_bracket_info(target_age)

    parts = []

    # 1. Identity preservation anchor + subject (Priority 1 for CLIP cross-attention)
    if include_identity_anchor:
        parts.append(f"portrait of the exact same {subject}, identical facial bone structure, identical eyes and nose")
    else:
        parts.append(f"portrait of {subject}")

    # 2. Target age and age bracket
    parts.append(f"aged {target_age} years old, {bracket_name}")

    # 3. Skin texture markers
    if include_skin:
        parts.append(skin_marker)

    # 4. Hair markers
    if include_hair:
        parts.append(hair_marker)

    # 5. Photorealism
    parts.append("sharp studio photo, realistic skin texture")

    if custom_suffix:
        parts.append(custom_suffix.strip())

    return ", ".join(parts)

def build_negative_prompt(custom_negative: Optional[str] = None) -> str:
    """Returns a robust negative prompt to prevent identity loss and generation artifacts."""
    if custom_negative:
        return custom_negative
    return (
        "deformed, distorted, disfigured, blurry, low resolution, bad anatomy, bad eyes, "
        "cartoon, 3d render, illustration, anime, drawing, painting, oversmoothed plastic skin, "
        "airbrushed, artificial, changed identity, different person, unnatural smile, double face"
    )

