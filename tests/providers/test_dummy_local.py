import pytest
from unittest.mock import Mock, patch
from PIL import Image
from pixelbliss.providers.dummy_local import generate_dummy_local_image, DummyLocalProvider


class TestDummyLocalProvider:
    def test_dummy_local_provider_init(self):
        provider = DummyLocalProvider()
        assert provider.example_images_dir == "outputs/2025-09-04/cosmic-minimal_A_beautiful_cosmic-minimal_landscap"
        assert len(provider.available_images) == 8

    @patch('pixelbliss.providers.dummy_local.os.path.exists')
    @patch('pixelbliss.providers.dummy_local.Image.open')
    @patch('pixelbliss.providers.dummy_local.random.choice')
    def test_generate_dummy_local_image_success(self, mock_random_choice, mock_image_open, mock_exists):
        mock_exists.return_value = True
        mock_random_choice.return_value = "test_image.png"
        mock_image = Mock(spec=Image.Image)
        mock_image_open.return_value = mock_image

        result = generate_dummy_local_image("test prompt", "test model")
        assert result is not None
        assert result["provider"] == "dummy_local"
        assert result["model"] == "test model"
        assert "seed" in result
        assert result["image_path"].endswith("test_image.png")

    @patch('pixelbliss.providers.dummy_local.os.path.exists')
    @patch('pixelbliss.providers.dummy_local.Image.open')
    @patch('pixelbliss.providers.dummy_local.random.choice')
    def test_generate_dummy_local_image_negative_hash(self, mock_random_choice, mock_image_open, mock_exists):
        mock_exists.return_value = True
        mock_random_choice.return_value = "test_image.png"
        mock_image = Mock(spec=Image.Image)
        mock_image_open.return_value = mock_image

        # Use a prompt that will generate a negative hash
        result = generate_dummy_local_image("negative_hash_prompt", "test model")
        assert result is not None
        assert result["seed"] >= 0  # Should be positive after abs()

    @patch('pixelbliss.providers.dummy_local.os.path.exists')
    def test_generate_dummy_local_image_file_not_found(self, mock_exists):
        mock_exists.return_value = False
        result = generate_dummy_local_image("test prompt", "test model")
        assert result is None

    @patch('pixelbliss.providers.dummy_local.os.path.exists')
    @patch('pixelbliss.providers.dummy_local.Image.open')
    def test_generate_dummy_local_image_exception(self, mock_image_open, mock_exists):
        mock_exists.return_value = True
        mock_image_open.side_effect = Exception("Test exception")
        result = generate_dummy_local_image("test prompt", "test model")
        assert result is None
