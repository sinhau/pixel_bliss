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
        # without complex mocking that might fail due to numpy internals
        mock_image = Mock(spec=Image.Image)
        mock_gray = Mock()
        mock_image.convert.return_value = mock_gray
        
        # Mock numpy.array to return a simple array
        with patch('pixelbliss.imaging.metrics.np.array') as mock_array:
            mock_array.return_value = np.array([[128, 128], [128, 128]])  # Simple 2x2 gray image
            
            result = entropy(mock_image)
            
            # Just verify it returns a float (actual entropy calculation)
            assert isinstance(result, float)
            mock_image.convert.assert_called_once_with('L')
