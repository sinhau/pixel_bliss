from typing import Dict, Any, Optional
from PIL import Image

ImageResult = Dict[str, Any]  # image: PIL.Image, provider: str, model: str, seed: int

def generate_image(prompt: str, provider: str, model: str) -> Optional[ImageResult]:
    """
    Generate an image using the specified provider and model.
    
    Args:
        prompt: Text prompt for image generation.
        provider: Name of the provider to use ("fal", "replicate", "dummy_local").
        model: Model identifier for the specific provider.
        
    Returns:
        Optional[ImageResult]: Dictionary containing image data and metadata, or None if failed.
        
    Raises:
        ValueError: If the provider is not supported.
    """
    if provider == "fal":
        from .fal import generate_fal_image
        return generate_fal_image(prompt, model)
    elif provider == "replicate":
        from .replicate import generate_replicate_image
        return generate_replicate_image(prompt, model)
    elif provider == "dummy_local":
        from .dummy_local import generate_dummy_local_image
        return generate_dummy_local_image(prompt, model)
    else:
        raise ValueError(f"Unknown provider: {provider}")
