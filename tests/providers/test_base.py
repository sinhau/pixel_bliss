import pytest
from unittest.mock import Mock, patch
from pixelbliss.providers.base import generate_image, ImageResult


class TestProviderBase:
    """Test provider base functionality."""

    def test_generate_image_fal_provider(self, mock_pil_image):
        """Test image generation with FAL provider."""
        expected_result = {
            "image": mock_pil_image,
            "provider": "fal",
            "model": "test-model",
            "seed": 12345
        }
        
        with patch('pixelbliss.providers.fal.generate_fal_image') as mock_fal:
            mock_fal.return_value = expected_result
            
            result = generate_image("test prompt", "fal", "test-model")
            
            assert result == expected_result
            mock_fal.assert_called_once_with("test prompt", "test-model")

    def test_generate_image_replicate_provider(self, mock_pil_image):
        """Test image generation with Replicate provider."""
        expected_result = {
            "image": mock_pil_image,
            "provider": "replicate",
            "model": "test-model",
            "seed": 67890
        }
        
        with patch('pixelbliss.providers.replicate.generate_replicate_image') as mock_replicate:
            mock_replicate.return_value = expected_result
            
            result = generate_image("test prompt", "replicate", "test-model")
            
            assert result == expected_result
            mock_replicate.assert_called_once_with("test prompt", "test-model")

    def test_generate_image_dummy_local_provider(self, mock_pil_image):
        """Test image generation with dummy local provider."""
        expected_result = {
            "image": mock_pil_image,
            "provider": "dummy_local",
            "model": "test-model",
            "seed": 11111
        }
        
        with patch('pixelbliss.providers.dummy_local.generate_dummy_local_image') as mock_dummy:
            mock_dummy.return_value = expected_result
            
            result = generate_image("test prompt", "dummy_local", "test-model")
            
            assert result == expected_result
            mock_dummy.assert_called_once_with("test prompt", "test-model")

    def test_generate_image_unknown_provider(self):
        """Test image generation with unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider: unknown"):
            generate_image("test prompt", "unknown", "test-model")

    def test_generate_image_provider_returns_none(self):
        """Test handling when provider returns None."""
        with patch('pixelbliss.providers.fal.generate_fal_image') as mock_fal:
            mock_fal.return_value = None
            
            result = generate_image("test prompt", "fal", "test-model")
            
            assert result is None
            mock_fal.assert_called_once_with("test prompt", "test-model")
