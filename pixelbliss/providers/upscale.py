from typing import Optional
from PIL import Image
import replicate
import fal_client
import io
import base64
from tenacity import retry, stop_after_attempt, wait_exponential

def _image_to_data_uri(image: Image.Image) -> str:
    """
    Convert PIL Image to base64 data URI for FAL API.
    
    Args:
        image: PIL Image object to convert.
        
    Returns:
        str: Base64-encoded data URI string.
    """
    buffer = io.BytesIO()
    # Save as PNG to preserve quality
    image.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Encode to base64
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{image_data}"

def _dummy_local_upscale(image: Image.Image, factor: int) -> Image.Image:
    """
    Dummy upscaler using PIL's built-in resampling for testing.
    
    Uses LANCZOS resampling for better quality than nearest neighbor.
    
    Args:
        image: PIL Image object to upscale.
        factor: Upscaling factor (e.g., 2 for 2x upscaling).
        
    Returns:
        Image.Image: Upscaled image using PIL resampling.
    """
    # Get current dimensions
    width, height = image.size
    
    # Calculate new dimensions
    new_width = width * factor
    new_height = height * factor
    
    # Upscale using LANCZOS resampling (high quality)
    upscaled = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return upscaled

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def upscale(image: Image.Image, provider: str, model: str, factor: int) -> Image.Image:
    """
    Upscale an image using the specified provider and model.
    
    Args:
        image: PIL Image object to upscale.
        provider: Upscaling provider ("replicate", "fal", "dummy_local").
        model: Model identifier for the specific provider.
        factor: Upscaling factor (e.g., 2 for 2x upscaling).
        
    Returns:
        Image.Image: Upscaled image.
        
    Raises:
        ValueError: If provider is unsupported or API returns no image.
        requests.HTTPError: If image download fails.
    """
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
    
    elif provider == "fal":
        # Convert PIL Image to data URI for FAL API
        image_data_uri = _image_to_data_uri(image)
        
        # Use FAL ESRGAN model for upscaling
        # Default to fal-ai/esrgan if no specific model provided
        fal_model = model if model else "fal-ai/esrgan"
        
        # Map factor to FAL scale parameter
        scale = factor if factor else 2
        
        # Call FAL API
        result = fal_client.run(
            fal_model,
            arguments={
                "image_url": image_data_uri,
                "scale": scale,
                "model": "RealESRGAN_x4plus",  # Use the specific model variant
                "output_format": "png"
            }
        )
        
        # Extract upscaled image URL from response
        if "image" in result and "url" in result["image"]:
            image_url = result["image"]["url"]
        else:
            raise ValueError("No upscaled image returned from FAL API")
        
        # Download and return the upscaled image
        import requests
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        upscaled = Image.open(response.raw)
        return upscaled
    
    elif provider == "dummy_local":
        # Use local PIL upscaling - no API calls required
        return _dummy_local_upscale(image, factor)
    
    else:
        raise ValueError(f"Unsupported upscale provider: {provider}")
