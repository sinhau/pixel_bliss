import pytest
from unittest.mock import Mock, patch
from PIL import Image


class TestReplicateProvider:
    def setup_method(self):
        """Setup method to ensure clean state for each test."""
        # Clear any cached modules
        import sys
        if 'pixelbliss.providers.replicate' in sys.modules:
            del sys.modules['pixelbliss.providers.replicate']

    @patch('pixelbliss.config.load_config')
    @patch('replicate.run')
    @patch('requests.get')
    @patch('PIL.Image.open')
    def test_generate_replicate_image_success_list_output(self, mock_image_open, mock_requests_get, mock_replicate_run, mock_load_config):
        # Mock config to avoid file I/O and retry delays
        mock_config = Mock()
        mock_config.image_generation.retries_per_image = 1  # Minimize retries
        mock_load_config.return_value = mock_config
        
        # Import after mocking config
        from pixelbliss.providers.replicate import generate_replicate_image
        
        mock_replicate_run.return_value = ["http://example.com/image.png"]
        mock_response = Mock()
        mock_requests_get.return_value = mock_response
        mock_image = Mock(spec=Image.Image)
        mock_image_open.return_value = mock_image

        result = generate_replicate_image("test prompt", "replicate/test-model")
        assert result is not None
        assert result["provider"] == "replicate"
        assert result["model"] == "replicate/test-model"
        assert result["seed"] == 12345678

    @patch('pixelbliss.config.load_config')
    @patch('replicate.run')
    @patch('requests.get')
    @patch('PIL.Image.open')
    def test_generate_replicate_image_success_single_output(self, mock_image_open, mock_requests_get, mock_replicate_run, mock_load_config):
        # Mock config to avoid file I/O and retry delays
        mock_config = Mock()
        mock_config.image_generation.retries_per_image = 1  # Minimize retries
        mock_load_config.return_value = mock_config
        
        # Import after mocking config
        from pixelbliss.providers.replicate import generate_replicate_image
        
        mock_replicate_run.return_value = "http://example.com/image.png"
        mock_response = Mock()
        mock_requests_get.return_value = mock_response
        mock_image = Mock(spec=Image.Image)
        mock_image_open.return_value = mock_image

        result = generate_replicate_image("test prompt", "replicate/test-model")
        assert result is not None
        assert result["provider"] == "replicate"

    @patch('pixelbliss.config.load_config')
    @patch('replicate.run')
    def test_generate_replicate_image_exception(self, mock_replicate_run, mock_load_config):
        # Mock config to avoid file I/O and retry delays
        mock_config = Mock()
        mock_config.image_generation.retries_per_image = 1  # Minimize retries
        mock_load_config.return_value = mock_config
        
        # Import after mocking config
        from pixelbliss.providers.replicate import generate_replicate_image
        
        mock_replicate_run.side_effect = Exception("API error")
        result = generate_replicate_image("test prompt", "replicate/test-model")
        assert result is None

    @patch('pixelbliss.config.load_config')
    @patch('replicate.run')
    @patch('requests.get')
    @patch('PIL.Image.open')
    def test_generate_replicate_image_with_retry_success(self, mock_image_open, mock_requests_get, mock_replicate_run, mock_load_config):
        # Mock config to avoid file I/O and retry delays
        mock_config = Mock()
        mock_config.image_generation.retries_per_image = 1  # Minimize retries
        mock_load_config.return_value = mock_config
        
        # Import after mocking config
        from pixelbliss.providers.replicate import _generate_replicate_image_with_retry
        
        mock_replicate_run.return_value = ["http://example.com/image.png"]
        mock_response = Mock()
        mock_requests_get.return_value = mock_response
        mock_image = Mock(spec=Image.Image)
        mock_image_open.return_value = mock_image

        result = _generate_replicate_image_with_retry("test prompt", "replicate/test-model")
        assert result is not None
        assert result["provider"] == "replicate"
