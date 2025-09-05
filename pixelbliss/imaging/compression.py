"""
Image compression utilities for Twitter upload size limits.
Provides intelligent compression to stay under 5MB while maintaining quality.
"""

import os
import io
from PIL import Image, ImageOps
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Twitter's file size limit
TWITTER_MAX_SIZE_MB = 5
TWITTER_MAX_SIZE_BYTES = TWITTER_MAX_SIZE_MB * 1024 * 1024

# Quality settings for progressive compression
QUALITY_LEVELS = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50]
MIN_QUALITY = 50

# Maximum dimensions to try before reducing quality
MAX_DIMENSION = 4096


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in MB
    """
    if not os.path.exists(file_path):
        return 0.0
    return os.path.getsize(file_path) / (1024 * 1024)


def get_image_size_bytes(image: Image.Image, format: str = 'PNG', quality: int = 95) -> int:
    """
    Get the size in bytes that an image would have when saved.
    
    Args:
        image: PIL Image object
        format: Image format ('PNG', 'JPEG')
        quality: JPEG quality (ignored for PNG)
        
    Returns:
        Size in bytes
    """
    buffer = io.BytesIO()
    save_kwargs = {'format': format}
    if format.upper() == 'JPEG':
        save_kwargs['quality'] = quality
        save_kwargs['optimize'] = True
    elif format.upper() == 'PNG':
        save_kwargs['optimize'] = True
    
    image.save(buffer, **save_kwargs)
    return buffer.tell()


def resize_image_proportionally(image: Image.Image, max_dimension: int) -> Image.Image:
    """
    Resize image maintaining aspect ratio with max dimension limit.
    
    Args:
        image: PIL Image to resize
        max_dimension: Maximum width or height
        
    Returns:
        Resized PIL Image
    """
    w, h = image.size
    if max(w, h) <= max_dimension:
        return image
    
    if w > h:
        new_w, new_h = max_dimension, int(h * max_dimension / w)
    else:
        new_w, new_h = int(w * max_dimension / h), max_dimension
    
    # Use high-quality resampling
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def compress_image_smart(image: Image.Image, target_size_bytes: int = TWITTER_MAX_SIZE_BYTES) -> Tuple[Image.Image, str, int]:
    """
    Intelligently compress image to target size while maintaining quality.
    
    Strategy:
    1. Try PNG first (lossless)
    2. If too large, try progressive JPEG quality reduction
    3. If still too large, reduce dimensions and retry
    
    Args:
        image: PIL Image to compress
        target_size_bytes: Target file size in bytes
        
    Returns:
        Tuple of (compressed_image, format, quality_used)
        - compressed_image: The compressed PIL Image
        - format: Final format used ('PNG' or 'JPEG')
        - quality_used: JPEG quality used (95 for PNG)
    """
    # Ensure image is in RGB mode for consistent processing
    if image.mode not in ('RGB', 'RGBA'):
        if image.mode == 'P' and 'transparency' in image.info:
            image = image.convert('RGBA')
        else:
            image = image.convert('RGB')
    
    # Auto-orient based on EXIF data
    image = ImageOps.exif_transpose(image)
    
    original_size = image.size
    current_image = image.copy()
    
    logger.info(f"Starting compression for {original_size[0]}x{original_size[1]} image, target: {target_size_bytes/1024/1024:.1f}MB")
    
    # Step 1: Try PNG first (lossless)
    png_size = get_image_size_bytes(current_image, 'PNG')
    if png_size <= target_size_bytes:
        logger.info(f"PNG compression successful: {png_size/1024/1024:.2f}MB")
        return current_image, 'PNG', 95
    
    logger.info(f"PNG too large ({png_size/1024/1024:.2f}MB), trying JPEG compression")
    
    # Step 2: Try JPEG with progressive quality reduction
    for quality in QUALITY_LEVELS:
        # Convert to RGB for JPEG if needed
        jpeg_image = current_image
        if jpeg_image.mode == 'RGBA':
            # Create white background for transparency
            background = Image.new('RGB', jpeg_image.size, (255, 255, 255))
            background.paste(jpeg_image, mask=jpeg_image.split()[-1] if jpeg_image.mode == 'RGBA' else None)
            jpeg_image = background
        
        jpeg_size = get_image_size_bytes(jpeg_image, 'JPEG', quality)
        if jpeg_size <= target_size_bytes:
            logger.info(f"JPEG compression successful at quality {quality}: {jpeg_size/1024/1024:.2f}MB")
            return jpeg_image, 'JPEG', quality
    
    logger.info(f"Quality reduction insufficient, trying dimension reduction")
    
    # Step 3: Reduce dimensions and retry
    dimension_steps = [3840, 3200, 2560, 2048, 1920, 1600, 1280, 1024]
    
    for max_dim in dimension_steps:
        if max(current_image.size) <= max_dim:
            continue
            
        resized_image = resize_image_proportionally(current_image, max_dim)
        logger.info(f"Trying {resized_image.size[0]}x{resized_image.size[1]} dimensions")
        
        # Try PNG first for resized image
        png_size = get_image_size_bytes(resized_image, 'PNG')
        if png_size <= target_size_bytes:
            logger.info(f"Resized PNG successful: {png_size/1024/1024:.2f}MB")
            return resized_image, 'PNG', 95
        
        # Try JPEG qualities for resized image
        for quality in QUALITY_LEVELS:
            jpeg_image = resized_image
            if jpeg_image.mode == 'RGBA':
                background = Image.new('RGB', jpeg_image.size, (255, 255, 255))
                background.paste(jpeg_image, mask=jpeg_image.split()[-1])
                jpeg_image = background
            
            jpeg_size = get_image_size_bytes(jpeg_image, 'JPEG', quality)
            if jpeg_size <= target_size_bytes:
                logger.info(f"Resized JPEG successful at {max_dim}px, quality {quality}: {jpeg_size/1024/1024:.2f}MB")
                return jpeg_image, 'JPEG', quality
    
    # Fallback: Use minimum quality JPEG with smallest reasonable size
    final_image = resize_image_proportionally(current_image, 1024)
    if final_image.mode == 'RGBA':
        background = Image.new('RGB', final_image.size, (255, 255, 255))
        background.paste(final_image, mask=final_image.split()[-1])
        final_image = background
    
    final_size = get_image_size_bytes(final_image, 'JPEG', MIN_QUALITY)
    logger.warning(f"Using fallback compression: {final_image.size[0]}x{final_image.size[1]} at quality {MIN_QUALITY}, size: {final_size/1024/1024:.2f}MB")
    
    return final_image, 'JPEG', MIN_QUALITY


def compress_image_file(input_path: str, output_path: Optional[str] = None, target_size_mb: float = TWITTER_MAX_SIZE_MB) -> Tuple[str, float, str, int]:
    """
    Compress an image file to target size.
    
    Args:
        input_path: Path to input image file
        output_path: Path for output file (if None, overwrites input)
        target_size_mb: Target file size in MB
        
    Returns:
        Tuple of (output_path, final_size_mb, format_used, quality_used)
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Check if compression is needed
    current_size_mb = get_file_size_mb(input_path)
    if current_size_mb <= target_size_mb:
        logger.info(f"File already under target size: {current_size_mb:.2f}MB <= {target_size_mb}MB")
        if output_path and output_path != input_path:
            # Copy file if different output path specified
            import shutil
            shutil.copy2(input_path, output_path)
            return output_path, current_size_mb, 'ORIGINAL', 95
        return input_path, current_size_mb, 'ORIGINAL', 95
    
    logger.info(f"Compressing {input_path} from {current_size_mb:.2f}MB to under {target_size_mb}MB")
    
    # Load and compress image
    with Image.open(input_path) as image:
        compressed_image, format_used, quality_used = compress_image_smart(
            image, 
            int(target_size_mb * 1024 * 1024)
        )
        
        # Determine output path
        if output_path is None:
            output_path = input_path
        
        # Save compressed image
        save_kwargs = {'optimize': True}
        if format_used == 'JPEG':
            save_kwargs['quality'] = quality_used
        
        compressed_image.save(output_path, format=format_used, **save_kwargs)
        
        final_size_mb = get_file_size_mb(output_path)
        logger.info(f"Compression complete: {final_size_mb:.2f}MB using {format_used} (quality: {quality_used})")
        
        return output_path, final_size_mb, format_used, quality_used


def prepare_for_twitter_upload(file_paths: list[str], config=None) -> list[str]:
    """
    Prepare a list of image files for Twitter upload by compressing if needed.
    
    Args:
        file_paths: List of image file paths
        config: Configuration object with twitter_compression settings (optional)
        
    Returns:
        List of processed file paths (may be modified in-place)
    """
    processed_paths = []
    
    # Use config settings if provided
    if config and hasattr(config, 'twitter_compression'):
        compression_config = config.twitter_compression
        if not compression_config.enabled:
            logger.info("Twitter compression disabled in config")
            return file_paths
        
        max_size_mb = compression_config.max_size_mb
    else:
        max_size_mb = TWITTER_MAX_SIZE_MB
    
    for file_path in file_paths:
        try:
            output_path, final_size, format_used, quality = compress_image_file(
                file_path, 
                target_size_mb=max_size_mb
            )
            processed_paths.append(output_path)
            
            if format_used != 'ORIGINAL':
                logger.info(f"Compressed {os.path.basename(file_path)}: {final_size:.2f}MB ({format_used}, Q{quality})")
            else:
                logger.info(f"No compression needed for {os.path.basename(file_path)}: {final_size:.2f}MB")
                
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            # Include original file even if compression failed
            processed_paths.append(file_path)
    
    return processed_paths
