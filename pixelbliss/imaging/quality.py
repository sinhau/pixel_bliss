"""
Local image quality assessment without ML models.
Provides simple, fast checks for size, sharpness, and exposure.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple


def resize_for_quality(image: Image.Image, long_side: int = 768) -> Image.Image:
    """
    Resize image for quality assessment, maintaining aspect ratio.
    
    Args:
        image: PIL Image to resize
        long_side: Target size for the longer dimension
        
    Returns:
        PIL Image resized for quality assessment
    """
    w, h = image.size
    if max(w, h) <= long_side:
        return image
    
    if w > h:
        new_w, new_h = long_side, int(h * long_side / w)
    else:
        new_w, new_h = int(w * long_side / h), long_side
    
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def check_size_and_aspect(image: Image.Image, min_side: int, ar_min: float, ar_max: float) -> bool:
    """
    Check if image meets minimum size and aspect ratio requirements.
    
    Args:
        image: PIL Image to check
        min_side: Minimum dimension for either width or height
        ar_min: Minimum aspect ratio (width/height)
        ar_max: Maximum aspect ratio (width/height)
        
    Returns:
        bool: True if image passes size and aspect ratio checks
    """
    w, h = image.size
    
    # Check minimum size
    if min(w, h) < min_side:
        return False
    
    # Check aspect ratio
    aspect_ratio = w / h
    if aspect_ratio < ar_min or aspect_ratio > ar_max:
        return False
    
    return True


def sharpness_score(image: Image.Image, sharpness_min: float, sharpness_good: float) -> Tuple[bool, float]:
    """
    Compute sharpness score using Variance of Laplacian.
    
    Args:
        image: PIL Image to assess
        sharpness_min: Minimum VoL threshold (reject if below)
        sharpness_good: VoL value considered "good" (score = 1.0)
        
    Returns:
        Tuple of (passes_floor, sharpness_score)
        - passes_floor: True if VoL >= sharpness_min
        - sharpness_score: Normalized score in [0, 1]
    """
    # Convert to grayscale numpy array
    gray = np.array(image.convert('L'))
    
    # Compute Variance of Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    vol = laplacian.var()
    
    # Check floor
    passes_floor = vol >= sharpness_min
    
    # Compute normalized score
    score = min(vol / sharpness_good, 1.0) if sharpness_good > 0 else 0.0
    
    return passes_floor, score


def exposure_score(image: Image.Image, clip_max: float) -> Tuple[bool, float]:
    """
    Compute exposure score based on shadow/highlight clipping.
    
    Args:
        image: PIL Image to assess
        clip_max: Maximum allowed clipping fraction (reject if above)
        
    Returns:
        Tuple of (passes_floor, exposure_score)
        - passes_floor: True if clipping <= clip_max
        - exposure_score: Normalized score in [0, 1]
    """
    # Convert to grayscale numpy array
    gray = np.array(image.convert('L'))
    
    # Count clipped pixels
    total_pixels = gray.size
    dark_pixels = np.sum(gray <= 3)
    bright_pixels = np.sum(gray >= 252)
    
    clip_frac = (dark_pixels + bright_pixels) / total_pixels
    
    # Check floor
    passes_floor = clip_frac <= clip_max
    
    # Compute normalized score (1.0 = no clipping, 0.0 = max clipping)
    score = max(1.0 - clip_frac / clip_max, 0.0) if clip_max > 0 else 0.0
    
    return passes_floor, score


def evaluate_local(image: Image.Image, cfg) -> Tuple[bool, float]:
    """
    Evaluate local image quality using all checks.
    
    Args:
        image: PIL Image to assess
        cfg: Configuration object with local_quality settings
        
    Returns:
        Tuple of (passes_all_floors, local_quality_score)
        - passes_all_floors: True if image passes all hard floors
        - local_quality_score: Composite score in [0, 1]
    """
    # Get config values
    lq_cfg = cfg.local_quality
    resize_long = lq_cfg.resize_long
    min_side = lq_cfg.min_side
    ar_min = lq_cfg.ar_min
    ar_max = lq_cfg.ar_max
    sharpness_min = lq_cfg.sharpness_min
    sharpness_good = lq_cfg.sharpness_good
    clip_max = lq_cfg.clip_max
    
    # Resize for consistent assessment
    resized_image = resize_for_quality(image, resize_long)
    
    # Check size and aspect ratio (hard floor)
    if not check_size_and_aspect(image, min_side, ar_min, ar_max):
        return False, 0.0
    
    # Check sharpness
    sharp_passes, sharp_score = sharpness_score(resized_image, sharpness_min, sharpness_good)
    if not sharp_passes:
        return False, 0.0
    
    # Check exposure
    exp_passes, exp_score = exposure_score(resized_image, clip_max)
    if not exp_passes:
        return False, 0.0
    
    # Compute composite score (simple average)
    local_quality = 0.5 * sharp_score + 0.5 * exp_score
    
    return True, local_quality
