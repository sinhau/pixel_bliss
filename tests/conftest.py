import pytest
from unittest.mock import Mock, MagicMock
from PIL import Image
import datetime
import pytz
from pixelbliss.config import Config, PromptGeneration, ImageGeneration, Ranking, Upscale, WallpaperVariant, AestheticScoring, Alerts


@pytest.fixture
def sample_config():
    """Fixture providing a sample configuration for testing."""
    return Config(
        timezone="America/Los_Angeles",
        categories=["sci-fi", "tech", "nature"],
        art_styles=["Digital Art", "Realism"],
        category_selection_method="time",
        rotation_minutes=180,
        prompt_generation=PromptGeneration(
            provider="openai",
            model="gpt-5",
            num_prompt_variants=3
        ),
        image_generation=ImageGeneration(
            provider_order=["fal", "replicate"],
            model_fal=["black-forest-labs/flux-1.1"],
            model_replicate=["black-forest-labs/flux"],
            retries_per_image=2
        ),
        ranking=Ranking(
            w_brightness=0.25,
            w_entropy=0.25,
            w_aesthetic=0.50,
            entropy_min=3.5,
            brightness_min=10,
            brightness_max=245,
            phash_distance_min=6
        ),
        aesthetic_scoring=AestheticScoring(
            provider="replicate",
            model="laion/aesthetic-predictor:v2-14",
            score_min=0.0,
            score_max=1.0
        ),
        upscale=Upscale(
            enabled=True,
            provider="replicate",
            model="real-esrgan-4x",
            factor=2
        ),
        wallpaper_variants=[
            WallpaperVariant(name="desktop_16x9_4k", w=3840, h=2160),
            WallpaperVariant(name="phone_9x16_2k", w=1440, h=2560)
        ],
        alerts=Alerts(
            enabled=True,
            webhook_url_env="ALERT_WEBHOOK_URL"
        )
    )


@pytest.fixture
def mock_pil_image():
    """Fixture providing a mock PIL Image."""
    mock_image = Mock(spec=Image.Image)
    mock_image.size = (1024, 1024)
    mock_image.mode = "RGB"
    mock_image.getdata.return_value = [128] * (1024 * 1024 * 3)  # Gray image
    return mock_image


@pytest.fixture
def sample_image_result(mock_pil_image):
    """Fixture providing a sample ImageResult dictionary."""
    return {
        "image": mock_pil_image,
        "provider": "fal",
        "model": "black-forest-labs/flux-1.1",
        "seed": 12345,
        "image_url": "https://example.com/image.jpg"
    }


@pytest.fixture
def fixed_datetime():
    """Fixture providing a fixed datetime for consistent testing."""
    tz = pytz.timezone("America/Los_Angeles")
    return datetime.datetime(2024, 1, 15, 14, 30, 0, tzinfo=tz)


@pytest.fixture
def mock_providers(mocker):
    """Fixture that mocks all external providers."""
    return {
        'fal': mocker.patch('pixelbliss.providers.fal.generate_fal_image'),
        'replicate': mocker.patch('pixelbliss.providers.replicate.generate_replicate_image'),
        'dummy_local': mocker.patch('pixelbliss.providers.dummy_local.generate_dummy_local_image')
    }


@pytest.fixture
def mock_twitter_client(mocker):
    """Fixture that mocks Twitter client operations."""
    return {
        'upload_media': mocker.patch('pixelbliss.twitter.client.upload_media'),
        'set_alt_text': mocker.patch('pixelbliss.twitter.client.set_alt_text'),
        'create_tweet': mocker.patch('pixelbliss.twitter.client.create_tweet')
    }


@pytest.fixture
def mock_file_operations(mocker):
    """Fixture that mocks file system operations."""
    return {
        'save_images': mocker.patch('pixelbliss.storage.fs.save_images'),
        'save_meta': mocker.patch('pixelbliss.storage.fs.save_meta'),
        'load_recent_hashes': mocker.patch('pixelbliss.storage.manifest.load_recent_hashes'),
        'append': mocker.patch('pixelbliss.storage.manifest.append'),
        'update_tweet_id': mocker.patch('pixelbliss.storage.manifest.update_tweet_id')
    }
