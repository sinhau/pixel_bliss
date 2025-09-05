import os
from typing import Optional
from PIL import Image
import fal_client
from tenacity import retry, stop_after_attempt, wait_exponential
from pixelbliss.config import load_config
from .base import ImageResult

config = load_config()

@retry(stop=stop_after_attempt(config.image_generation.retries_per_image), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_fal_image(prompt: str, model: str) -> Optional[ImageResult]:
    try:
        # Use fal_client.run for synchronous image generation
        result = fal_client.run(
            model,
            arguments={"prompt": prompt}
        )

        # Extract image URL from response
        image_url = result["images"][0]["url"]
        seed = result.get("seed", 12345678)

        # Download image
        import requests
        image = Image.open(requests.get(image_url, stream=True).raw)

        return {
            "image": image,
            "provider": "fal",
            "model": model,
            "seed": seed
        }
    except Exception as e:
        print(f"FAL generation failed: {e}")
        return None
