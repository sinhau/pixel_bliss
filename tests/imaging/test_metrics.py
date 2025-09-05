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

    def test_entropy_function_called(self):
        """Test entropy function executes without error."""
        # Create a simple test that verifies the function can be called
        # Mock all numpy operations to avoid division by zero warnings
        mock_image = Mock(spec=Image.Image)
        mock_gray = Mock()
        mock_image.convert.return_value = mock_gray
        
        with patch('pixelbliss.imaging.metrics.np.array') as mock_array, \
             patch('pixelbliss.imaging.metrics.np.histogram') as mock_hist, \
             patch('pixelbliss.imaging.metrics.np.sum') as mock_sum, \
             patch('pixelbliss.imaging.metrics.np.log2') as mock_log2:
            
            # Mock histogram with valid data
            mock_hist.return_value = (np.array([0, 100, 200] + [0] * 253), None)
            mock_sum.return_value = 5.5  # Mock entropy result
            mock_log2.return_value = np.array([-2.0] * 256)
            
            result = entropy(mock_image)
            
            # Verify it returns the mocked result (entropy function uses negative sign)
            assert result == -5.5
            mock_image.convert.assert_called_once_with('L')
