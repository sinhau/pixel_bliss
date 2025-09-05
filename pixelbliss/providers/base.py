from typing import Dict, Any, Optional
from PIL import Image

ImageResult = Dict[str, Any]  # image: PIL.Image, provider: str, model: str, seed: int

def generate_image(prompt: str, provider: str, model: str) -> Optional[ImageResult]:
    if provider == "fal":
        from .fal import generate_fal_image
        return generate_fal_image(prompt, model)
    elif provider == "replicate":
        from .replicate import generate_replicate_image
        return generate_replicate_image(prompt, model, retries)
    else:
        raise ValueError(f"Unknown provider: {provider}")
