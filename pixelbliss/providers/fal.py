import os
from typing import Optional
from PIL import Image
import fal_client
from tenacity import retry, stop_after_attempt, wait_exponential
from pixelbliss.config import load_config
from .base import ImageResult

config = load_config()

@retry(stop=stop_after_attempt(config.image_generation.retries_per_image), wait=wait_exponential(multiplier=1, min=4, max=10))
def _generate_fal_image_with_retry(prompt: str, model: str) -> ImageResult:
    """
    Generate an image using FAL API with automatic retry logic.
    
    Args:
        prompt: Text prompt for image generation.
        model: FAL model identifier to use.
        
    Returns:
        ImageResult: Dictionary containing the generated image and metadata.
        
    Raises:
        ValueError: If no images are returned from the API.
        requests.HTTPError: If image download fails.
    """
    # Use fal_client.run for synchronous image generation
    # Based on FAL API docs for Imagen 4 Ultra
    result = fal_client.run(
        model,
        arguments={
            "prompt": prompt,
        }
    )

    # Extract image URL from response - FAL returns images array with File objects
    if "images" in result and len(result["images"]) > 0:
        image_url = result["images"][0]["url"]
    else:
        raise ValueError("No images returned from FAL API")
    
    # Extract seed - according to docs, seed is at top level of response
    seed = result.get("seed", 12345678)

    # Download image with proper error handling
    import requests
    response = requests.get(image_url, stream=True)
    response.raise_for_status()
    image = Image.open(response.raw)

    return {
        "image": image,
        "provider": "fal",
        "model": model,
        "seed": seed,
        "image_url": image_url
    }

def generate_fal_image(prompt: str, model: str) -> Optional[ImageResult]:
    """
    Generate an image using FAL API with error handling.
    
    Args:
        prompt: Text prompt for image generation.
        model: FAL model identifier to use.
        
    Returns:
        Optional[ImageResult]: Dictionary containing image and metadata, or None if failed.
    """
    try:
        return _generate_fal_image_with_retry(prompt, model)
    except Exception as e:
        from ..logging_config import get_logger
        logger = get_logger('providers.fal')
        logger.error(f"FAL generation failed after retries: {e}")
        return None
