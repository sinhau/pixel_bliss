import os
import random
from typing import Optional
from PIL import Image
from .base import ImageResult

class DummyLocalProvider:
    def __init__(self):
        # Path to the example images
        self.example_images_dir = "outputs/2025-09-04/cosmic-minimal_A_beautiful_cosmic-minimal_landscap"
        self.available_images = [
            "square_1x1_2k.png",
            "phone_9x16_2k.png", 
            "phone_20x9_3.2k.png",
            "desktop_16x9_1080p.png",
            "desktop_16x9_1440p.png",
            "desktop_16x9_4k.png",
            "desktop_16x10_1600p.png",
            "ultrawide_21x9.png"
        ]

def generate_dummy_local_image(prompt: str, model: str) -> Optional[ImageResult]:
    """
    Generate a dummy image by returning one of the existing images from outputs directory.
    This is useful for testing without making actual API calls.
    
    Args:
        prompt: Text prompt for image generation (used for seed generation).
        model: Model identifier (unused in dummy implementation).
        
    Returns:
        Optional[ImageResult]: Dictionary containing a randomly selected existing image, or None if failed.
    """
    try:
        provider = DummyLocalProvider()
        
        # Randomly select one of the available images
        selected_image = random.choice(provider.available_images)
        image_path = os.path.join(provider.example_images_dir, selected_image)
        
        # Check if the image file exists
        if not os.path.exists(image_path):
            print(f"Dummy image not found at: {image_path}")
            return None
        
        # Load the image
        image = Image.open(image_path)
        
        # Generate a dummy seed based on prompt hash for consistency
        seed = hash(prompt) % 1000000000
        if seed < 0:
            seed = abs(seed)
        
        return {
            "image": image,
            "provider": "dummy_local",
            "model": model,
            "seed": seed,
            "image_path": image_path,  # Additional info for debugging
            "selected_file": selected_image
        }
        
    except Exception as e:
        print(f"Dummy local generation failed: {e}")
        return None
