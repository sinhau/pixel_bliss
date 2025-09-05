from typing import Optional
from PIL import Image
import replicate
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def upscale(image: Image.Image, provider: str, model: str, factor: int) -> Optional[Image.Image]:
    try:
        if provider == "replicate":
            # Assuming model is like "real-esrgan-4x"
            output = replicate.run(
                model,
                input={"image": image, "scale": factor}
            )
            # Assume output is image URL
            import requests
            upscaled = Image.open(requests.get(output, stream=True).raw)
            return upscaled
        else:
            raise ValueError(f"Unsupported upscale provider: {provider}")
    except Exception as e:
        print(f"Upscale failed: {e}")
        return None
