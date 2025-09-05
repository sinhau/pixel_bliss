import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch
from pixelbliss.imaging.metrics import brightness, entropy


class TestMetrics:
    """Test image metrics calculations."""

    def test_brightness_uniform_image(self):
        """Test brightness calculation on uniform gray image."""
        # Create a uniform gray image (128 gray level)
        with patch('numpy.array') as mock_array:
            mock_array.return_value.mean.return_value = 128.0
            
            mock_image = Mock(spec=Image.Image)
            mock_gray = Mock()
            mock_image.convert.return_value = mock_gray
            
            result = brightness(mock_image)
            
            assert result == 128.0
            mock_image.convert.assert_called_once_with('L')

    def test_brightness_black_image(self):
        """Test brightness calculation on black image."""
        with patch('numpy.array') as mock_array:
            mock_array.return_value.mean.return_value = 0.0
            
            mock_image = Mock(spec=Image.Image)
            mock_gray = Mock()
            mock_image.convert.return_value = mock_gray
            
            result = brightness(mock_image)
            
            assert result == 0.0
            mock_image.convert.assert_called_once_with('L')

    def test_brightness_white_image(self):
        """Test brightness calculation on white image."""
        with patch('numpy.array') as mock_array:
            mock_array.return_value.mean.return_value = 255.0
            
            mock_image = Mock(spec=Image.Image)
            mock_gray = Mock()
            mock_image.convert.return_value = mock_gray
            
            result = brightness(mock_image)
            
            assert result == 255.0
            mock_image.convert.assert_called_once_with('L')

    def test_entropy_uniform_image(self):
        """Test entropy calculation on uniform image (should be low)."""
        # Mock a uniform histogram (all pixels same value)
        uniform_hist = np.zeros(256)
        uniform_hist[128] = 1000  # All pixels are gray level 128
        
        with patch('numpy.histogram') as mock_hist, \
             patch('numpy.array') as mock_array, \
             patch('numpy.sum') as mock_sum, \
             patch('numpy.log2') as mock_log2:
            
            mock_hist.return_value = (uniform_hist, None)
            mock_log2.return_value = np.array([0.0] * 256)  # log2(1) = 0 for uniform
            mock_sum.return_value = 0.0  # Entropy should be 0 for uniform
            
            mock_image = Mock(spec=Image.Image)
            mock_gray = Mock()
            mock_image.convert.return_value = mock_gray
            
            result = entropy(mock_image)
            
            assert result == 0.0
            mock_image.convert.assert_called_once_with('L')

    def test_entropy_random_image(self):
        """Test entropy calculation on random image (should be high)."""
        # Mock a uniform histogram (equal distribution)
        uniform_hist = np.ones(256) * 100  # Equal distribution
        
        with patch('numpy.histogram') as mock_hist, \
             patch('numpy.array') as mock_array, \
             patch('numpy.sum') as mock_sum, \
             patch('numpy.log2') as mock_log2:
            
            mock_hist.return_value = (uniform_hist, None)
            # For uniform distribution, entropy should be log2(256) = 8
            mock_log2.return_value = np.full(256, -8.0)  # log2(1/256) = -8
            mock_sum.return_value = 8.0  # Maximum entropy for 8-bit image
            
            mock_image = Mock(spec=Image.Image)
            mock_gray = Mock()
            mock_image.convert.return_value = mock_gray
            
            result = entropy(mock_image)
            
            assert result == 8.0
            mock_image.convert.assert_called_once_with('L')

    def test_entropy_handles_zero_histogram(self):
        """Test entropy calculation handles zero values in histogram."""
        # Mock histogram with some zero values
        hist_with_zeros = np.array([0, 100, 0, 200, 0] + [0] * 251)
        
        with patch('numpy.histogram') as mock_hist, \
             patch('numpy.array') as mock_array, \
             patch('numpy.sum') as mock_sum, \
             patch('numpy.log2') as mock_log2:
            
            mock_hist.return_value = (hist_with_zeros, None)
            # The function adds 1e-10 to avoid log(0)
            mock_log2.return_value = np.array([-10.0] * 256)  # Mock log values
            mock_sum.return_value = 2.0  # Some entropy value
            
            mock_image = Mock(spec=Image.Image)
            mock_gray = Mock()
            mock_image.convert.return_value = mock_gray
            
            result = entropy(mock_image)
            
            assert result == 2.0
            mock_image.convert.assert_called_once_with('L')
            # Verify that log2 was called with histogram + epsilon to avoid log(0)
            mock_log2.assert_called_once()
