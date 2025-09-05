import os
from typing import Optional
from PIL import Image
import fal_client
from tenacity import retry, stop_after_attempt, wait_exponential
from .base import ImageResult

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_fal_image(prompt: str, model: str, retries: int) -> Optional[ImageResult]:
    try:
        # Assuming fal_client has a function to generate image
        # This is hypothetical; adjust based on actual fal API
        result = fal_client.generate_image(
            model=model,
            prompt=prompt,
            api_key=os.getenv("FAL_API_KEY")
        )
        # Assume result has 'image_url' and 'seed'
        image_url = result['image_url']
        seed = result.get('seed', 12345678)

        # Download image
        import requests
        response = requests.get(image_url)
        image = Image.open(requests.get(image_url, stream=True).raw)

        return {
            "image": image,
            "provider": "fal",
            "model": model,
            "seed": seed
        }
    except Exception as e:
        print(f"FAL generation failed: {e}")
        if retries > 0:
            return generate_fal_image(prompt, model, retries - 1)
        return None
