import os
from typing import Optional
from PIL import Image
import replicate
from tenacity import retry, stop_after_attempt, wait_exponential
from .base import ImageResult

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_replicate_image(prompt: str, model: str, retries: int) -> Optional[ImageResult]:
    try:
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
            "seed": seed
        }
    except Exception as e:
        print(f"Replicate generation failed: {e}")
        if retries > 0:
            return generate_replicate_image(prompt, model, retries - 1)
        return None
