from PIL import Image
from typing import Dict, List
from ..config import WallpaperVariant

def crop_pad(image: Image.Image, w: int, h: int) -> Image.Image:
    """
    Resize an image to exact dimensions by cropping and scaling.
    
    The image is first cropped to match the target aspect ratio, then resized
    to the exact dimensions using high-quality LANCZOS resampling.
    
    Args:
        image: PIL Image object to resize.
        w: Target width in pixels.
        h: Target height in pixels.
        
    Returns:
        Image.Image: Resized image with exact dimensions.
    """
    # Resize image to fit w x h, cropping or padding as needed
    img_ratio = image.width / image.height
    target_ratio = w / h

    if img_ratio > target_ratio:
        # Image is wider, crop width
        new_width = int(image.height * target_ratio)
        offset = (image.width - new_width) // 2
        image = image.crop((offset, 0, offset + new_width, image.height))
    elif img_ratio < target_ratio:
        # Image is taller, crop height
        new_height = int(image.width / target_ratio)
        offset = (image.height - new_height) // 2
        image = image.crop((0, offset, image.width, offset + new_height))

    # Resize to exact dimensions
    return image.resize((w, h), Image.LANCZOS)

def make_wallpaper_variants(image: Image.Image, variants_cfg: List[WallpaperVariant]) -> Dict[str, Image.Image]:
    """
    Create multiple wallpaper variants from a single image.
    
    Args:
        image: Source PIL Image object to create variants from.
        variants_cfg: List of WallpaperVariant configurations specifying dimensions.
        
    Returns:
        Dict[str, Image.Image]: Dictionary mapping variant names to resized images.
    """
    variants = {}
    for variant in variants_cfg:
        variants[variant.name] = crop_pad(image, variant.w, variant.h)
    return variants
