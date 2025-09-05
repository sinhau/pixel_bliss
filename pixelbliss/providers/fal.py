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
    # Use fal_client.run for synchronous image generation
    # Based on FAL API docs for Imagen 4 Ultra
    result = fal_client.run(
        model,
        arguments={
            "prompt": prompt,
            "aspect_ratio": "1:1",        # Default aspect ratio
            "num_images": 1,              # Generate 1 image
            "resolution": "1K",           # Default resolution
            "negative_prompt": ""         # Empty negative prompt by default
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
    try:
        return _generate_fal_image_with_retry(prompt, model)
    except Exception as e:
        print(f"FAL generation failed after retries: {e}")
        return None
