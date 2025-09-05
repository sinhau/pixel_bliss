import os
from typing import Optional
from PIL import Image
import replicate
from tenacity import retry, stop_after_attempt, wait_exponential
from pixelbliss.config import load_config
from .base import ImageResult

config = load_config()

@retry(stop=stop_after_attempt(config.image_generation.retries_per_image), wait=wait_exponential(multiplier=1, min=4, max=10))
def _generate_replicate_image_with_retry(prompt: str, model: str) -> ImageResult:
    """
    Generate an image using Replicate API with automatic retry logic.
    
    Args:
        prompt: Text prompt for image generation.
        model: Replicate model identifier to use.
        
    Returns:
        ImageResult: Dictionary containing the generated image and metadata.
        
    Raises:
        requests.HTTPError: If image download fails.
    """
    # Assuming replicate.run for the model
    output = replicate.run(
        model,
        input={"prompt": prompt}
    )
    # Assume output is a list with image URL
    image_url = output[0] if isinstance(output, list) else output
    seed = 12345678  # Replicate may not provide seed easily

    # Download image
    import requests
    image = Image.open(requests.get(image_url, stream=True).raw)

    return {
        "image": image,
        "provider": "replicate",
        "model": model,
        "seed": seed,
        "image_url": image_url
    }

def generate_replicate_image(prompt: str, model: str) -> Optional[ImageResult]:
    """
    Generate an image using Replicate API with error handling.
    
    Args:
        prompt: Text prompt for image generation.
        model: Replicate model identifier to use.
        
    Returns:
        Optional[ImageResult]: Dictionary containing image and metadata, or None if failed.
    """
    try:
        return _generate_replicate_image_with_retry(prompt, model)
    except Exception as e:
        print(f"Replicate generation failed after retries: {e}")
        return None
