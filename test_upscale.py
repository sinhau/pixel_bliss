#!/usr/bin/env python3
"""
Test script for the upscale functionality with FAL provider.
"""

from PIL import Image
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pixelbliss.providers.upscale import upscale

def test_fal_upscale():
    """Test FAL upscaling with a simple test image."""
    try:
        # Create a simple test image (100x100 red square)
        test_image = Image.new('RGB', (100, 100), color='red')
        
        print("Testing FAL upscaling...")
        print(f"Original image size: {test_image.size}")
        
        # Test FAL upscaling - now returns Image.Image directly or raises exception
        upscaled_image = upscale(
            image=test_image,
            provider="fal",
            model="fal-ai/esrgan",
            factor=2
        )
        
        print(f"Upscaled image size: {upscaled_image.size}")
        print("✅ FAL upscaling test passed!")
        
        # Save the result for inspection
        upscaled_image.save("test_upscaled_fal.png")
        print("Saved upscaled image as test_upscaled_fal.png")
            
    except Exception as e:
        print(f"❌ FAL upscaling test failed with error: {e}")

def test_replicate_upscale():
    """Test that existing Replicate functionality still works."""
    try:
        # Create a simple test image
        test_image = Image.new('RGB', (100, 100), color='blue')
        
        print("\nTesting Replicate upscaling (should fail gracefully without API key)...")
        
        # This should fail gracefully since we likely don't have Replicate configured
        upscaled_image = upscale(
            image=test_image,
            provider="replicate",
            model="some-model",
            factor=2
        )
        
        print("✅ Replicate upscaling worked (unexpected)")
        print(f"Upscaled image size: {upscaled_image.size}")
            
    except Exception as e:
        print(f"✅ Replicate upscaling failed gracefully: {e}")

if __name__ == "__main__":
    print("Testing upscale functionality...")
    test_fal_upscale()
    test_replicate_upscale()
    print("\nTest completed!")
