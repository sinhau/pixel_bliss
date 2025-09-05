import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image


class TestFalProvider:
    def setup_method(self):
        """Setup method to ensure clean state for each test."""
        # Clear any cached modules
        import sys
        if 'pixelbliss.providers.fal' in sys.modules:
            del sys.modules['pixelbliss.providers.fal']

    @patch('pixelbliss.providers.fal.retry')
    @patch('pixelbliss.config.load_config')
    @patch('fal_client.run')
    @patch('requests.get')
    @patch('PIL.Image.open')
    def test_generate_fal_image_success(self, mock_image_open, mock_requests_get, mock_fal_run, mock_load_config, mock_retry):
        # Mock retry decorator to avoid wait times
        mock_retry.return_value = lambda func: func
        # Mock config to avoid file I/O
        mock_config = Mock()
        mock_config.image_generation.retries_per_image = 2
        mock_load_config.return_value = mock_config
        
        # Import after mocking config
        from pixelbliss.providers.fal import generate_fal_image
        
        # Setup mocks
        mock_fal_run.return_value = {
            "images": [{"url": "http://example.com/image.png"}],
            "seed": 12345
        }
        mock_response = Mock()
        mock_requests_get.return_value = mock_response
        mock_response.raise_for_status.return_value = None
        mock_response.raw = Mock()
        mock_image = Mock(spec=Image.Image)
        mock_image_open.return_value = mock_image

        result = generate_fal_image("test prompt", "fal/test-model")
        assert result is not None
        assert result["provider"] == "fal"
        assert result["model"] == "fal/test-model"
        assert result["seed"] == 12345

    @patch('pixelbliss.providers.fal.retry')
    @patch('pixelbliss.config.load_config')
    @patch('fal_client.run')
    def test_generate_fal_image_no_images(self, mock_fal_run, mock_load_config, mock_retry):
        # Mock retry decorator to avoid wait times
        mock_retry.return_value = lambda func: func
        # Mock config to avoid file I/O
        mock_config = Mock()
        mock_config.image_generation.retries_per_image = 2
        mock_load_config.return_value = mock_config
        
        # Import after mocking config
        from pixelbliss.providers.fal import generate_fal_image
        
        mock_fal_run.return_value = {"images": []}
        result = generate_fal_image("test prompt", "fal/test-model")
        assert result is None

    @patch('pixelbliss.providers.fal.retry')
    @patch('pixelbliss.config.load_config')
    @patch('fal_client.run')
    @patch('requests.get')
    def test_generate_fal_image_download_fail(self, mock_requests_get, mock_fal_run, mock_load_config, mock_retry):
        # Mock retry decorator to avoid wait times
        mock_retry.return_value = lambda func: func
        # Mock config to avoid file I/O
        mock_config = Mock()
        mock_config.image_generation.retries_per_image = 2
        mock_load_config.return_value = mock_config
        
        # Import after mocking config
        from pixelbliss.providers.fal import generate_fal_image
        
        mock_fal_run.return_value = {
            "images": [{"url": "http://example.com/image.png"}],
            "seed": 12345
        }
        mock_requests_get.side_effect = Exception("Download failed")
        result = generate_fal_image("test prompt", "fal/test-model")
        assert result is None

    @patch('pixelbliss.providers.fal.retry')
    @patch('pixelbliss.config.load_config')
    @patch('fal_client.run')
    @patch('requests.get')
    @patch('PIL.Image.open')
    def test_generate_fal_image_with_retry_success(self, mock_image_open, mock_requests_get, mock_fal_run, mock_load_config, mock_retry):
        # Mock retry decorator to avoid wait times
        mock_retry.return_value = lambda func: func
        # Mock config to avoid file I/O
        mock_config = Mock()
        mock_config.image_generation.retries_per_image = 2
        mock_load_config.return_value = mock_config
        
        # Import after mocking config
        from pixelbliss.providers.fal import _generate_fal_image_with_retry
        
        mock_fal_run.return_value = {
            "images": [{"url": "http://example.com/image.png"}],
            "seed": 12345
        }
        mock_response = Mock()
        mock_requests_get.return_value = mock_response
        mock_response.raise_for_status.return_value = None
        mock_response.raw = Mock()
        mock_image = Mock(spec=Image.Image)
        mock_image_open.return_value = mock_image

        result = _generate_fal_image_with_retry("test prompt", "fal/test-model")
        assert result is not None
        assert result["provider"] == "fal"
