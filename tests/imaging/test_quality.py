"""
Unit tests for local image quality assessment.
"""

import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock
from pixelbliss.imaging import quality


class TestQualityFunctions:
    """Test individual quality assessment functions."""
    
    def test_resize_for_quality(self):
        """Test image resizing for quality assessment."""
        # Create a test image larger than target size
        img = Image.new('RGB', (1200, 800), color='red')
        resized = quality.resize_for_quality(img, long_side=768)
        
        # Should resize to 768x512 (maintaining aspect ratio)
        assert max(resized.size) == 768
        assert resized.size == (768, 512)
        
        # Test image already smaller than target
        small_img = Image.new('RGB', (400, 300), color='blue')
        not_resized = quality.resize_for_quality(small_img, long_side=768)
        assert not_resized.size == (400, 300)
    
    def test_check_size_and_aspect(self):
        """Test size and aspect ratio validation."""
        # Valid image
        img = Image.new('RGB', (1024, 768), color='green')
        assert quality.check_size_and_aspect(img, min_side=512, ar_min=0.5, ar_max=2.0) == True
        
        # Too small
        small_img = Image.new('RGB', (400, 300), color='red')
        assert quality.check_size_and_aspect(small_img, min_side=512, ar_min=0.5, ar_max=2.0) == False
        
        # Aspect ratio too wide
        wide_img = Image.new('RGB', (2000, 600), color='blue')
        assert quality.check_size_and_aspect(wide_img, min_side=512, ar_min=0.5, ar_max=2.0) == False
        
        # Aspect ratio too tall
        tall_img = Image.new('RGB', (600, 2000), color='yellow')
        assert quality.check_size_and_aspect(tall_img, min_side=512, ar_min=0.5, ar_max=2.0) == False
    
    def test_sharpness_score(self):
        """Test sharpness scoring using Variance of Laplacian."""
        # Create a sharp image (checkerboard pattern)
        sharp_array = np.zeros((100, 100), dtype=np.uint8)
        sharp_array[::10, ::10] = 255  # Create high-frequency pattern
        sharp_img = Image.fromarray(sharp_array).convert('RGB')
        
        passes, score = quality.sharpness_score(sharp_img, sharpness_min=50.0, sharpness_good=200.0)
        assert passes == True
        assert score > 0.0
        
        # Create a blurred image (uniform color)
        blurred_img = Image.new('RGB', (100, 100), color=(128, 128, 128))
        passes_blur, score_blur = quality.sharpness_score(blurred_img, sharpness_min=50.0, sharpness_good=200.0)
        assert passes_blur == False
        assert score_blur < score  # Blurred should have lower score
    
    def test_exposure_score(self):
        """Test exposure scoring based on clipping."""
        # Well-exposed image (mid-gray)
        good_img = Image.new('RGB', (100, 100), color=(128, 128, 128))
        passes, score = quality.exposure_score(good_img, clip_max=0.1)
        assert passes == True
        assert score > 0.8  # Should have high score
        
        # Overexposed image (mostly white)
        bright_array = np.full((100, 100, 3), 255, dtype=np.uint8)
        bright_img = Image.fromarray(bright_array)
        passes_bright, score_bright = quality.exposure_score(bright_img, clip_max=0.1)
        assert passes_bright == False  # Should fail clipping test
        
        # Underexposed image (mostly black)
        dark_array = np.full((100, 100, 3), 0, dtype=np.uint8)
        dark_img = Image.fromarray(dark_array)
        passes_dark, score_dark = quality.exposure_score(dark_img, clip_max=0.1)
        assert passes_dark == False  # Should fail clipping test
    
    def test_evaluate_local(self):
        """Test complete local quality evaluation."""
        # Create mock config
        mock_cfg = Mock()
        mock_cfg.local_quality = Mock()
        mock_cfg.local_quality.resize_long = 768
        mock_cfg.local_quality.min_side = 512
        mock_cfg.local_quality.ar_min = 0.5
        mock_cfg.local_quality.ar_max = 2.0
        mock_cfg.local_quality.sharpness_min = 50.0
        mock_cfg.local_quality.sharpness_good = 200.0
        mock_cfg.local_quality.clip_max = 0.2
        
        # Create a good quality image
        good_array = np.random.randint(50, 200, (800, 600, 3), dtype=np.uint8)
        # Add some high-frequency content for sharpness
        good_array[::10, ::10] = 255
        good_img = Image.fromarray(good_array)
        
        passes, score = quality.evaluate_local(good_img, mock_cfg)
        assert passes == True
        assert 0.0 <= score <= 1.0
        
        # Create a bad quality image (too small)
        bad_img = Image.new('RGB', (300, 200), color=(128, 128, 128))
        passes_bad, score_bad = quality.evaluate_local(bad_img, mock_cfg)
        assert passes_bad == False
        assert score_bad == 0.0


class TestQualityIntegration:
    """Test quality assessment integration scenarios."""
    
    def test_quality_score_range(self):
        """Test that quality scores are always in [0, 1] range."""
        mock_cfg = Mock()
        mock_cfg.local_quality = Mock()
        mock_cfg.local_quality.resize_long = 768
        mock_cfg.local_quality.min_side = 512
        mock_cfg.local_quality.ar_min = 0.5
        mock_cfg.local_quality.ar_max = 2.0
        mock_cfg.local_quality.sharpness_min = 50.0
        mock_cfg.local_quality.sharpness_good = 200.0
        mock_cfg.local_quality.clip_max = 0.2
        
        # Test various image types
        test_images = [
            Image.new('RGB', (800, 600), color=(64, 64, 64)),    # Dark
            Image.new('RGB', (800, 600), color=(192, 192, 192)), # Bright
            Image.new('RGB', (800, 600), color=(128, 128, 128)), # Mid-gray
        ]
        
        for img in test_images:
            passes, score = quality.evaluate_local(img, mock_cfg)
            if passes:
                assert 0.0 <= score <= 1.0, f"Score {score} out of range for image"
    
    def test_quality_floors_rejection(self):
        """Test that quality floors properly reject bad images."""
        mock_cfg = Mock()
        mock_cfg.local_quality = Mock()
        mock_cfg.local_quality.resize_long = 768
        mock_cfg.local_quality.min_side = 1000  # Very high minimum
        mock_cfg.local_quality.ar_min = 0.9
        mock_cfg.local_quality.ar_max = 1.1     # Very narrow aspect ratio range
        mock_cfg.local_quality.sharpness_min = 1000.0  # Very high sharpness requirement
        mock_cfg.local_quality.sharpness_good = 2000.0
        mock_cfg.local_quality.clip_max = 0.01  # Very low clipping tolerance
        
        # This image should fail multiple floors
        test_img = Image.new('RGB', (800, 400), color=(255, 255, 255))  # Wrong AR, overexposed
        passes, score = quality.evaluate_local(test_img, mock_cfg)
        assert passes == False
        assert score == 0.0
    
    def test_sharpness_comparison(self):
        """Test that sharper images get higher sharpness scores."""
        # Sharp image with high-frequency content
        sharp_array = np.zeros((200, 200), dtype=np.uint8)
        for i in range(0, 200, 4):
            sharp_array[i:i+2, :] = 255  # Create stripes
        sharp_img = Image.fromarray(sharp_array).convert('RGB')
        
        # Smooth image
        smooth_img = Image.new('RGB', (200, 200), color=(128, 128, 128))
        
        passes_sharp, score_sharp = quality.sharpness_score(sharp_img, 10.0, 100.0)
        passes_smooth, score_smooth = quality.sharpness_score(smooth_img, 10.0, 100.0)
        
        assert score_sharp > score_smooth, "Sharp image should have higher sharpness score"
    
    def test_exposure_comparison(self):
        """Test that well-exposed images get higher exposure scores."""
        # Well-exposed image
        good_img = Image.new('RGB', (100, 100), color=(128, 128, 128))
        
        # Overexposed image (more extreme to trigger clipping)
        bright_img = Image.new('RGB', (100, 100), color=(253, 253, 253))
        
        passes_good, score_good = quality.exposure_score(good_img, 0.2)
        passes_bright, score_bright = quality.exposure_score(bright_img, 0.2)
        
        assert score_good > score_bright, "Well-exposed image should have higher exposure score"
    
    def test_sharpness_score_zero_good(self):
        """Test sharpness score when sharpness_good is 0."""
        image = Image.new('RGB', (100, 100), color=(128, 128, 128))
        passes, score = quality.sharpness_score(image, 0.0, 0.0)
        assert score == 0.0, "Score should be 0.0 when sharpness_good is 0"

    def test_exposure_score_zero_clip_max(self):
        """Test exposure score when clip_max is 0."""
        image = Image.new('RGB', (100, 100), color=(128, 128, 128))
        passes, score = quality.exposure_score(image, 0.0)
        assert score == 0.0, "Score should be 0.0 when clip_max is 0"

    def test_evaluate_local_all_pass(self):
        """Test evaluate_local when all quality checks pass."""
        # Create mock config with very lenient values to ensure all checks pass
        mock_cfg = Mock()
        mock_cfg.local_quality = Mock()
        mock_cfg.local_quality.resize_long = 768
        mock_cfg.local_quality.min_side = 50   # Very low minimum
        mock_cfg.local_quality.ar_min = 0.1    # Very wide range
        mock_cfg.local_quality.ar_max = 10.0   # Very wide range
        mock_cfg.local_quality.sharpness_min = 0.1  # Very low threshold
        mock_cfg.local_quality.sharpness_good = 10.0  # Low good threshold
        mock_cfg.local_quality.clip_max = 0.9  # Very high tolerance
        
        # Create an image that should definitely pass all checks
        # Make it large enough and with good contrast for sharpness
        good_array = np.full((800, 600, 3), 100, dtype=np.uint8)  # Dark gray base
        # Add high contrast pattern for guaranteed sharpness
        good_array[::5, ::5] = 200  # High contrast pattern every 5 pixels
        good_img = Image.fromarray(good_array)
        
        passes, score = quality.evaluate_local(good_img, mock_cfg)
        assert passes == True, f"Image should pass all quality checks, got passes={passes}, score={score}"
        assert score > 0.0, f"Quality score should be positive, got {score}"
        assert score <= 1.0, f"Quality score should not exceed 1.0, got {score}"

    def test_evaluate_local_exposure_fail(self):
        """Test evaluate_local when exposure check fails (line 162 coverage)."""
        # Create mock config that passes size and sharpness but fails exposure
        mock_cfg = Mock()
        mock_cfg.local_quality = Mock()
        mock_cfg.local_quality.resize_long = 768
        mock_cfg.local_quality.min_side = 50   # Low minimum to pass size check
        mock_cfg.local_quality.ar_min = 0.1    # Wide range to pass aspect ratio
        mock_cfg.local_quality.ar_max = 10.0   
        mock_cfg.local_quality.sharpness_min = 0.1  # Low threshold to pass sharpness
        mock_cfg.local_quality.sharpness_good = 10.0
        mock_cfg.local_quality.clip_max = 0.01  # Very strict clipping tolerance to fail exposure
        
        # Create an overexposed image that will fail exposure but pass size/sharpness
        overexposed_array = np.full((800, 600, 3), 255, dtype=np.uint8)  # All white (overexposed)
        # Add some pattern for sharpness
        overexposed_array[::10, ::10] = 200  # Slight pattern for sharpness
        overexposed_img = Image.fromarray(overexposed_array)
        
        passes, score = quality.evaluate_local(overexposed_img, mock_cfg)
        assert passes == False, "Image should fail exposure check"
        assert score == 0.0, "Score should be 0.0 when exposure check fails"
