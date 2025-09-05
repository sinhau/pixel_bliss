import pytest
import datetime
import pytz
import asyncio
from unittest.mock import Mock, patch, AsyncMock, ANY
from pixelbliss.run_once import (
    generate_theme_hint, normalize_and_rescore, today_local, now_iso, tweet_url, fs_abs,
    try_in_order, post_once, generate_images_sequential
)


class TestThemeGeneration:
    """Test theme generation functions."""

    def test_generate_theme_hint_returns_valid_theme(self):
        """Test generate_theme_hint returns a valid theme."""
        result = generate_theme_hint()
        
        # Should return one of the predefined themes
        expected_themes = [
            "abstract", "nature", "cosmic", "geometric", "atmospheric", 
            "minimal", "organic", "crystalline", "flowing", "luminous"
        ]
        assert result in expected_themes

    def test_generate_theme_hint_multiple_calls(self):
        """Test generate_theme_hint returns valid themes across multiple calls."""
        results = []
        for _ in range(20):
            result = generate_theme_hint()
            results.append(result)
            
        # All results should be valid themes
        expected_themes = [
            "abstract", "nature", "cosmic", "geometric", "atmospheric", 
            "minimal", "organic", "crystalline", "flowing", "luminous"
        ]
        for result in results:
            assert result in expected_themes
        
        # Should have some variety (not all the same)
        assert len(set(results)) > 1


class TestNormalizeAndRescore:
    """Test scoring normalization and rescoring."""

    def test_normalize_and_rescore_basic(self, sample_config):
        """Test basic normalization and rescoring."""
        items = [
            {"brightness": 100, "entropy": 4.0, "aesthetic": 0.8},
            {"brightness": 200, "entropy": 5.0, "aesthetic": 0.6},
            {"brightness": 150, "entropy": 4.5, "aesthetic": 0.7}
        ]
        
        result = normalize_and_rescore(items, sample_config)
        
        # Check that all items have final scores
        for item in result:
            assert "final" in item
            assert isinstance(item["final"], float)
        
        # Verify normalization ranges (0-1 for brightness and entropy)
        # Item with highest brightness (200) should have normalized brightness = 1.0
        # Item with lowest brightness (100) should have normalized brightness = 0.0

    def test_normalize_and_rescore_empty_list(self, sample_config):
        """Test normalization with empty list."""
        items = []
        result = normalize_and_rescore(items, sample_config)
        assert result == []

    def test_normalize_and_rescore_single_item(self, sample_config):
        """Test normalization with single item."""
        items = [{"brightness": 100, "entropy": 4.0, "aesthetic": 0.8}]
        
        result = normalize_and_rescore(items, sample_config)
        
        assert len(result) == 1
        assert "final" in result[0]
        # With single item, normalized brightness and entropy should be 0.5
        # local_quality defaults to 0.5 when not present
        expected_final = (
            sample_config.ranking.w_brightness * 0.5 +
            sample_config.ranking.w_entropy * 0.5 +
            sample_config.ranking.w_aesthetic * 0.8 +
            sample_config.ranking.w_local_quality * 0.5
        )
        assert abs(result[0]["final"] - expected_final) < 0.001

    def test_normalize_and_rescore_identical_values(self, sample_config):
        """Test normalization when all values are identical."""
        items = [
            {"brightness": 100, "entropy": 4.0, "aesthetic": 0.8},
            {"brightness": 100, "entropy": 4.0, "aesthetic": 0.6},
            {"brightness": 100, "entropy": 4.0, "aesthetic": 0.7}
        ]
        
        result = normalize_and_rescore(items, sample_config)
        
        # When min == max, normalized values should be 0.5
        for item in result:
            assert "final" in item
            # local_quality defaults to 0.5 when not present
            expected_final = (
                sample_config.ranking.w_brightness * 0.5 +
                sample_config.ranking.w_entropy * 0.5 +
                sample_config.ranking.w_aesthetic * item["aesthetic"] +
                sample_config.ranking.w_local_quality * 0.5
            )
            assert abs(item["final"] - expected_final) < 0.001


class TestUtilityFunctions:
    """Test utility functions."""

    def test_today_local(self):
        """Test today_local returns correct format."""
        with patch('pixelbliss.run_once.datetime.datetime') as mock_datetime:
            tz = pytz.timezone("America/Los_Angeles")
            mock_now = datetime.datetime(2024, 1, 15, 14, 30, 0, tzinfo=tz)
            mock_datetime.now.return_value = mock_now
            mock_datetime.now.return_value.strftime.return_value = "2024-01-15"
            
            result = today_local()
            assert result == "2024-01-15"

    def test_now_iso(self):
        """Test now_iso returns ISO format."""
        with patch('pixelbliss.run_once.datetime.datetime') as mock_datetime:
            tz = pytz.timezone("America/Los_Angeles")
            mock_now = datetime.datetime(2024, 1, 15, 14, 30, 0, tzinfo=tz)
            mock_datetime.now.return_value = mock_now
            
            result = now_iso()
            assert result == mock_now.isoformat()

    def test_tweet_url(self):
        """Test tweet URL generation."""
        tweet_id = "1234567890"
        result = tweet_url(tweet_id)
        assert result == "https://x.com/user/status/1234567890"

    def test_fs_abs(self):
        """Test filesystem absolute path conversion."""
        path = "relative/path/to/file.jpg"
        result = fs_abs(path)
        assert result == path  # Currently just returns input


class TestTryInOrder:
    """Test try_in_order function."""

    @patch('pixelbliss.run_once.providers.base.generate_image')
    def test_try_in_order_success_first_provider(self, mock_generate):
        """Test try_in_order when first provider succeeds."""
        mock_result = {"image": "mock_image", "provider": "fal"}
        mock_generate.return_value = mock_result
        
        result = try_in_order("test prompt", ["fal", "replicate"], ["model1", "model2"], 3)
        
        assert result == mock_result
        mock_generate.assert_called_once_with("test prompt", "fal", "model1", 3)

    @patch('pixelbliss.run_once.providers.base.generate_image')
    def test_try_in_order_success_second_provider(self, mock_generate):
        """Test try_in_order when first provider fails, second succeeds."""
        mock_result = {"image": "mock_image", "provider": "replicate"}
        mock_generate.side_effect = [None, mock_result]  # First call fails, second succeeds
        
        result = try_in_order("test prompt", ["fal", "replicate"], ["model1", "model2"], 3)
        
        assert result == mock_result
        assert mock_generate.call_count == 2

    @patch('pixelbliss.run_once.providers.base.generate_image')
    def test_try_in_order_all_fail(self, mock_generate):
        """Test try_in_order when all providers fail."""
        mock_generate.return_value = None
        
        result = try_in_order("test prompt", ["fal", "replicate"], ["model1", "model2"], 3)
        
        assert result is None
        assert mock_generate.call_count == 2


class TestPostOnce:
    """Test post_once function."""

    @patch('pixelbliss.run_once.config.load_config')
    @patch('pixelbliss.run_once.generate_theme_hint')
    @patch('pixelbliss.run_once.prompts.make_base')
    @patch('pixelbliss.run_once.prompts.make_variants_from_base')
    @patch('pixelbliss.run_once.providers.base.generate_image')
    @patch('pixelbliss.alerts.webhook.send_failure')
    @pytest.mark.asyncio
    async def test_post_once_no_candidates(self, mock_send_failure, mock_generate, mock_variants, 
                                   mock_base, mock_theme, mock_config):
        """Test post_once when no image candidates are generated."""
        # Setup mocks
        mock_cfg = Mock()
        mock_cfg.image_generation.model_fal = ["model1"]
        mock_cfg.image_generation.provider_order = ["fal", "replicate"]
        mock_cfg.image_generation.model_replicate = ["model2"]
        mock_cfg.image_generation.async_enabled = False  # Disable to avoid asyncio.run() in tests
        mock_cfg.image_generation.max_concurrency = None
        mock_cfg.prompt_generation.num_prompt_variants = 1
        mock_cfg.prompt_generation.async_enabled = False  # Use sync for simplicity
        mock_cfg.prompt_generation.provider = "dummy"  # Set valid provider
        mock_cfg.discord.enabled = False  # Disable Discord for tests
        mock_config.return_value = mock_cfg
        
        mock_theme.return_value = "abstract"
        mock_base.return_value = "base prompt"
        mock_variants.return_value = ["variant1", "variant2"]
        mock_generate.return_value = None  # No images generated
        
        result = await post_once(dry_run=True)
        
        assert result == 1  # Should return error code
        mock_send_failure.assert_called_once()

    @patch('pixelbliss.run_once.config.load_config')
    @patch('pixelbliss.run_once.select_category')
    @patch('pixelbliss.run_once.prompts.make_base')
    @patch('pixelbliss.run_once.prompts.make_variants_from_base')
    @patch('pixelbliss.providers.base.generate_image')
    @patch('pixelbliss.run_once.metrics.brightness')
    @patch('pixelbliss.run_once.metrics.entropy')
    @patch('pixelbliss.run_once.sanity.passes_floors')
    @patch('pixelbliss.run_once.quality.evaluate_local')
    @patch('pixelbliss.run_once.aesthetic.aesthetic')
    @patch('pixelbliss.run_once.normalize_and_rescore')
    @patch('pixelbliss.run_once.today_local')
    @patch('pixelbliss.run_once.storage.paths.make_slug')
    @patch('pixelbliss.run_once.storage.paths.output_dir')
    @patch('pixelbliss.run_once.collage.save_collage')
    @patch('pixelbliss.run_once.manifest.load_recent_hashes')
    @patch('pixelbliss.run_once.phash.is_duplicate')
    @patch('pixelbliss.run_once.phash.phash_hex')
    @patch('pixelbliss.imaging.variants.make_wallpaper_variants')
    @patch('pixelbliss.run_once.prompts.make_alt_text')
    @patch('pixelbliss.run_once.storage.fs.save_images')
    @patch('pixelbliss.run_once.storage.fs.save_meta')
    @patch('pixelbliss.run_once.manifest.append')
    @patch('pixelbliss.run_once.now_iso')
    @pytest.mark.asyncio
    async def test_post_once_dry_run_success(self, mock_iso, mock_append, mock_save_meta, mock_save_images,
                                     mock_alt, mock_variants_wall, mock_phash, mock_duplicate, mock_hashes,
                                     mock_collage, mock_outdir, mock_slug, mock_today, mock_rescore,
                                     mock_aesthetic, mock_quality, mock_floors, mock_entropy, mock_brightness,
                                     mock_generate, mock_variants, mock_base, mock_category, mock_config):
        """Test post_once dry_run success path."""
        # Setup config
        mock_cfg = Mock()
        mock_cfg.categories = ["test"]  # Add categories list
        mock_cfg.image_generation.model_fal = ["model1"]
        mock_cfg.image_generation.provider_order = ["fal", "replicate"]
        mock_cfg.image_generation.model_replicate = ["model2"]
        mock_cfg.image_generation.async_enabled = False  # Disable to avoid asyncio.run() in tests
        mock_cfg.image_generation.max_concurrency = None
        mock_cfg.prompt_generation.num_prompt_variants = 1
        mock_cfg.prompt_generation.async_enabled = False  # Use sync for simplicity
        mock_cfg.prompt_generation.provider = "dummy"  # Set valid provider
        mock_cfg.upscale.enabled = False
        mock_cfg.aesthetic_scoring = Mock()
        mock_cfg.aesthetic_scoring.provider = "dummy_local"
        mock_cfg.discord.enabled = False  # Disable Discord for tests
        mock_config.return_value = mock_cfg
        
        # Setup mocks for successful path
        mock_category.return_value = "test"
        mock_base.return_value = "base prompt"
        mock_variants.return_value = ["variant1"]
        
        # Mock successful image generation
        mock_image = Mock()
        mock_generate.return_value = {"image": mock_image, "provider": "fal", "model": "test", "seed": 123}
        
        # Mock scoring to pass all checks
        mock_brightness.return_value = 150
        mock_entropy.return_value = 4.5
        mock_floors.return_value = True
        mock_quality.return_value = (True, 0.8)
        mock_aesthetic.return_value = 0.7
        
        # Mock rescoring
        mock_rescore.return_value = [{"final": 0.8, "image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1", "aesthetic": 0.7, "brightness": 150, "entropy": 4.5, "local_quality": 0.8}]
        
        # Mock file operations
        mock_today.return_value = "2024-01-01"
        mock_slug.return_value = "test_slug"
        mock_outdir.return_value = "/test/dir"
        mock_collage.return_value = "/test/collage.jpg"
        mock_hashes.return_value = []
        mock_duplicate.return_value = False
        mock_phash.return_value = "abc123"
        mock_variants_wall.return_value = {"desktop": mock_image}
        mock_alt.return_value = "alt text"
        mock_save_images.return_value = {"desktop": "/path/to/desktop.jpg"}
        mock_iso.return_value = "2024-01-01T12:00:00"
        
        result = await post_once(dry_run=True)
        
        assert result == 0  # Should succeed and return 0 for dry_run

    @patch('pixelbliss.run_once.config.load_config')
    @patch('pixelbliss.run_once.select_category')
    @patch('pixelbliss.run_once.prompts.make_base')
    @patch('pixelbliss.run_once.prompts.make_variants_from_base')
    @patch('pixelbliss.run_once.providers.base.generate_image')
    @patch('pixelbliss.run_once.metrics.brightness')
    @patch('pixelbliss.run_once.metrics.entropy')
    @patch('pixelbliss.run_once.sanity.passes_floors')
    @patch('pixelbliss.alerts.webhook.send_failure')
    @pytest.mark.asyncio
    async def test_post_once_sanity_check_fails(self, mock_send_failure, mock_floors, mock_entropy, mock_brightness,
                                        mock_generate, mock_variants, mock_base, mock_category, mock_config):
        """Test post_once when sanity check fails."""
        # Setup config
        mock_cfg = Mock()
        mock_cfg.categories = ["test"]  # Add categories list
        mock_cfg.image_generation.model_fal = ["model1"]
        mock_cfg.image_generation.provider_order = ["fal", "replicate"]
        mock_cfg.image_generation.model_replicate = ["model2"]
        mock_cfg.image_generation.async_enabled = False  # Disable to avoid asyncio.run() in tests
        mock_cfg.image_generation.max_concurrency = None
        mock_cfg.prompt_generation.num_prompt_variants = 1
        mock_cfg.prompt_generation.async_enabled = False  # Use sync for simplicity
        mock_cfg.prompt_generation.provider = "dummy"  # Set valid provider
        mock_cfg.aesthetic_scoring = Mock()
        mock_cfg.aesthetic_scoring.provider = "dummy_local"
        mock_cfg.discord.enabled = False  # Disable Discord for tests
        mock_config.return_value = mock_cfg
        
        # Setup mocks
        mock_category.return_value = "test"
        mock_base.return_value = "base prompt"
        mock_variants.return_value = ["variant1"]
        
        # Mock successful image generation
        mock_image = Mock()
        mock_generate.return_value = {"image": mock_image, "provider": "fal", "model": "test", "seed": 123}
        
        # Mock scoring - sanity check fails
        mock_brightness.return_value = 150
        mock_entropy.return_value = 4.5
        mock_floors.return_value = False  # Sanity check fails
        
        result = await post_once(dry_run=True)
        
        assert result == 1  # Should return error code
        mock_send_failure.assert_called_once_with("all candidates failed sanity/scoring", mock_cfg)

    @patch('pixelbliss.run_once.config.load_config')
    @patch('pixelbliss.run_once.select_category')
    @patch('pixelbliss.run_once.prompts.make_base')
    @patch('pixelbliss.run_once.prompts.make_variants_from_base')
    @patch('pixelbliss.run_once.providers.base.generate_image')
    @patch('pixelbliss.run_once.metrics.brightness')
    @patch('pixelbliss.run_once.metrics.entropy')
    @patch('pixelbliss.run_once.sanity.passes_floors')
    @patch('pixelbliss.run_once.quality.evaluate_local')
    @patch('pixelbliss.alerts.webhook.send_failure')
    @pytest.mark.asyncio
    async def test_post_once_local_quality_fails(self, mock_send_failure, mock_quality, mock_floors, mock_entropy, mock_brightness,
                                         mock_generate, mock_variants, mock_base, mock_category, mock_config):
        """Test post_once when local quality check fails."""
        # Setup config
        mock_cfg = Mock()
        mock_cfg.categories = ["test"]  # Add categories list
        mock_cfg.image_generation.model_fal = ["model1"]
        mock_cfg.image_generation.provider_order = ["fal", "replicate"]
        mock_cfg.image_generation.model_replicate = ["model2"]
        mock_cfg.image_generation.async_enabled = False  # Disable to avoid asyncio.run() in tests
        mock_cfg.image_generation.max_concurrency = None
        mock_cfg.prompt_generation.num_prompt_variants = 1
        mock_cfg.prompt_generation.async_enabled = False  # Use sync for simplicity
        mock_cfg.prompt_generation.provider = "dummy"  # Set valid provider
        mock_cfg.discord.enabled = False  # Disable Discord for tests
        mock_config.return_value = mock_cfg
        
        # Setup mocks
        mock_category.return_value = "test"
        mock_base.return_value = "base prompt"
        mock_variants.return_value = ["variant1"]
        
        # Mock successful image generation
        mock_image = Mock()
        mock_generate.return_value = {"image": mock_image, "provider": "fal", "model": "test", "seed": 123}
        
        # Mock scoring - local quality fails
        mock_brightness.return_value = 150
        mock_entropy.return_value = 4.5
        mock_floors.return_value = True
        mock_quality.return_value = (False, 0.3)  # Local quality fails
        
        result = await post_once(dry_run=True)
        
        assert result == 1  # Should return error code
        mock_send_failure.assert_called_once_with("all candidates failed sanity/scoring", mock_cfg)

    @patch('pixelbliss.run_once.config.load_config')
    @patch('pixelbliss.run_once.select_category')
    @patch('pixelbliss.run_once.prompts.make_base')
    @patch('pixelbliss.run_once.prompts.make_variants_from_base')
    @patch('pixelbliss.run_once.providers.base.generate_image')
    @patch('pixelbliss.run_once.metrics.brightness')
    @patch('pixelbliss.run_once.metrics.entropy')
    @patch('pixelbliss.run_once.sanity.passes_floors')
    @patch('pixelbliss.run_once.quality.evaluate_local')
    @patch('pixelbliss.run_once.aesthetic.aesthetic')
    @patch('pixelbliss.run_once.normalize_and_rescore')
    @patch('pixelbliss.run_once.today_local')
    @patch('pixelbliss.run_once.storage.paths.make_slug')
    @patch('pixelbliss.run_once.storage.paths.output_dir')
    @patch('pixelbliss.run_once.collage.save_collage')
    @patch('pixelbliss.run_once.manifest.load_recent_hashes')
    @patch('pixelbliss.run_once.phash.is_duplicate')
    @patch('pixelbliss.run_once.phash.phash_hex')
    @patch('pixelbliss.alerts.webhook.send_failure')
    @pytest.mark.asyncio
    async def test_post_once_winner_is_duplicate(self, mock_send_failure, mock_phash_hex, mock_duplicate, mock_hashes, mock_collage,
                                         mock_outdir, mock_slug, mock_today, mock_rescore, mock_aesthetic,
                                         mock_quality, mock_floors, mock_entropy, mock_brightness, mock_generate,
                                         mock_variants, mock_base, mock_category, mock_config):
        """Test post_once when winner is duplicate."""
        # Setup config
        mock_cfg = Mock()
        mock_cfg.categories = ["test"]  # Add categories list
        mock_cfg.image_generation.model_fal = ["model1"]
        mock_cfg.image_generation.provider_order = ["fal", "replicate"]
        mock_cfg.image_generation.model_replicate = ["model2"]
        mock_cfg.image_generation.async_enabled = False  # Disable to avoid asyncio.run() in tests
        mock_cfg.image_generation.max_concurrency = None
        mock_cfg.prompt_generation.num_prompt_variants = 1
        mock_cfg.prompt_generation.async_enabled = False  # Use sync for simplicity
        mock_cfg.prompt_generation.provider = "dummy"  # Set valid provider
        mock_cfg.aesthetic_scoring.provider = "dummy_local"
        mock_cfg.aesthetic_scoring.score_min = 0.0
        mock_cfg.aesthetic_scoring.score_max = 1.0
        mock_cfg.discord.enabled = False  # Disable Discord for tests
        mock_config.return_value = mock_cfg
        
        # Setup mocks
        mock_category.return_value = "test"
        mock_base.return_value = "base prompt"
        mock_variants.return_value = ["variant1"]
        
        # Mock successful image generation
        mock_image = Mock()
        mock_generate.return_value = {"image": mock_image, "provider": "fal", "model": "test", "seed": 123, "image_url": "http://example.com/image.jpg"}
        
        # Mock scoring to pass all checks
        mock_brightness.return_value = 150
        mock_entropy.return_value = 4.5
        mock_floors.return_value = True
        mock_quality.return_value = (True, 0.8)
        mock_aesthetic.return_value = 0.7  # This tests the image_url branch
        
        # Mock rescoring
        mock_rescore.return_value = [{"final": 0.8, "image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1", "aesthetic": 0.7, "brightness": 150, "entropy": 4.5, "local_quality": 0.8}]
        
        # Mock file operations
        mock_today.return_value = "2024-01-01"
        mock_slug.return_value = "test_slug"
        mock_outdir.return_value = "/test/dir"
        mock_collage.return_value = "/test/collage.jpg"
        mock_hashes.return_value = ["existing_hash"]
        mock_duplicate.return_value = True  # Winner is duplicate
        
        result = await post_once(dry_run=True)
        
        assert result == 0  # Should return 0 (not fatal)
        mock_send_failure.assert_called_once_with("near-duplicate with manifest history", mock_cfg)

    @patch('pixelbliss.run_once.config.load_config')
    @patch('pixelbliss.run_once.select_category')
    @patch('pixelbliss.run_once.prompts.make_base')
    @patch('pixelbliss.run_once.prompts.make_variants_from_base')
    @patch('pixelbliss.run_once.providers.base.generate_image')
    @patch('pixelbliss.run_once.metrics.brightness')
    @patch('pixelbliss.run_once.metrics.entropy')
    @patch('pixelbliss.run_once.sanity.passes_floors')
    @patch('pixelbliss.run_once.quality.evaluate_local')
    @patch('pixelbliss.run_once.aesthetic.aesthetic')
    @patch('pixelbliss.run_once.normalize_and_rescore')
    @patch('pixelbliss.run_once.today_local')
    @patch('pixelbliss.run_once.storage.paths.make_slug')
    @patch('pixelbliss.run_once.storage.paths.output_dir')
    @patch('pixelbliss.run_once.collage.save_collage')
    @patch('pixelbliss.run_once.manifest.load_recent_hashes')
    @patch('pixelbliss.run_once.phash.is_duplicate')
    @patch('pixelbliss.run_once.phash.phash_hex')
    @patch('pixelbliss.imaging.variants.make_wallpaper_variants')
    @patch('pixelbliss.run_once.prompts.make_alt_text')
    @patch('pixelbliss.run_once.storage.fs.save_images')
    @patch('pixelbliss.run_once.storage.fs.save_meta')
    @patch('pixelbliss.run_once.manifest.append')
    @patch('pixelbliss.run_once.now_iso')
    @patch('pixelbliss.twitter.client.upload_media')
    @patch('pixelbliss.twitter.client.set_alt_text')
    @patch('pixelbliss.twitter.client.create_tweet')
    @patch('pixelbliss.run_once.manifest.update_tweet_id')
    @patch('pixelbliss.alerts.webhook.send_success')
    @pytest.mark.asyncio
    async def test_post_once_full_execution(self, mock_send_success, mock_update_tweet, mock_create_tweet, mock_set_alt,
                                    mock_upload, mock_iso, mock_append, mock_save_meta, mock_save_images, mock_alt,
                                    mock_variants_wall, mock_phash, mock_duplicate, mock_hashes, mock_collage,
                                    mock_outdir, mock_slug, mock_today, mock_rescore, mock_aesthetic, mock_quality,
                                    mock_floors, mock_entropy, mock_brightness, mock_generate, mock_variants,
                                    mock_base, mock_category, mock_config):
        """Test post_once full execution (non-dry-run)."""
        # Setup config
        mock_cfg = Mock()
        mock_cfg.categories = ["test"]  # Add categories list
        mock_cfg.image_generation.model_fal = ["model1"]
        mock_cfg.image_generation.provider_order = ["fal", "replicate"]
        mock_cfg.image_generation.model_replicate = ["model2"]
        mock_cfg.image_generation.async_enabled = False  # Disable to avoid asyncio.run() in tests
        mock_cfg.image_generation.max_concurrency = None
        mock_cfg.prompt_generation.num_prompt_variants = 1
        mock_cfg.prompt_generation.async_enabled = False  # Use sync for simplicity
        mock_cfg.prompt_generation.provider = "dummy"  # Set valid provider
        mock_cfg.upscale.enabled = True
        mock_cfg.upscale.provider = "test_provider"
        mock_cfg.upscale.model = "test_model"
        mock_cfg.upscale.factor = 2
        mock_cfg.discord.enabled = False  # Disable Discord for tests
        mock_config.return_value = mock_cfg
        
        # Setup mocks
        mock_category.return_value = "test"
        mock_base.return_value = "base prompt"
        mock_variants.return_value = ["variant1"]
        
        # Mock successful image generation
        mock_image = Mock()
        mock_generate.return_value = {"image": mock_image, "provider": "fal", "model": "test", "seed": 123}
        
        # Mock scoring to pass all checks
        mock_brightness.return_value = 150
        mock_entropy.return_value = 4.5
        mock_floors.return_value = True
        mock_quality.return_value = (True, 0.8)
        mock_aesthetic.return_value = 0.7
        
        # Mock rescoring
        mock_rescore.return_value = [{"final": 0.8, "image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1", "aesthetic": 0.7, "brightness": 150, "entropy": 4.5, "local_quality": 0.8}]
        
        # Mock file operations
        mock_today.return_value = "2024-01-01"
        mock_slug.return_value = "test_slug"
        mock_outdir.return_value = "/test/dir"
        mock_collage.return_value = "/test/collage.jpg"
        mock_hashes.return_value = []
        mock_duplicate.return_value = False
        mock_phash.return_value = "abc123"
        
        # Mock upscale (will test the upscale failure fallback)
        with patch('pixelbliss.providers.upscale.upscale') as mock_upscale:
            mock_upscale.return_value = None  # Upscale fails, should fallback
            
            mock_variants_wall.return_value = {"desktop": mock_image}
            mock_alt.return_value = "alt text"
            mock_save_images.return_value = {"desktop": "/path/to/desktop.jpg", "base_img": "/path/to/base_img.png"}
            mock_iso.return_value = "2024-01-01T12:00:00"
            
            # Mock Twitter operations
            mock_upload.return_value = ["media_id_123"]
            mock_create_tweet.return_value = "tweet_id_456"
            
            result = await post_once(dry_run=False)  # Non-dry-run
            
            assert result == 0  # Should succeed
            mock_upload.assert_called_once()
            mock_set_alt.assert_called_once_with("media_id_123", "alt text")
            mock_create_tweet.assert_called_once()
            mock_update_tweet.assert_called_once()
            mock_send_success.assert_called_once()


class TestAsyncImageGeneration:
    """Test async image generation functions."""

    @pytest.mark.asyncio
    async def test_generate_for_variant_success_fal(self, sample_config):
        """Test generate_for_variant when FAL succeeds."""
        from pixelbliss.run_once import generate_for_variant
        
        mock_image = Mock()
        mock_result = {"image": mock_image, "provider": "fal", "model": "test_model", "seed": 123}
        
        with patch('pixelbliss.providers.base.generate_image', return_value=mock_result) as mock_generate:
            result = await generate_for_variant("test prompt", sample_config)
            
            assert len(result) == 1
            assert result[0]["prompt"] == "test prompt"
            assert result[0]["provider"] == "fal"
            mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_for_variant_fal_fails_replicate_succeeds(self, sample_config):
        """Test generate_for_variant when FAL fails but Replicate succeeds."""
        from pixelbliss.run_once import generate_for_variant
        
        mock_image = Mock()
        mock_result = {"image": mock_image, "provider": "replicate", "model": "test_model", "seed": 123}
        
        with patch('pixelbliss.providers.base.generate_image', side_effect=[None, mock_result]) as mock_generate:
            result = await generate_for_variant("test prompt", sample_config)
            
            assert len(result) == 1
            assert result[0]["prompt"] == "test prompt"
            assert result[0]["provider"] == "replicate"
            assert mock_generate.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_for_variant_all_fail(self, sample_config):
        """Test generate_for_variant when all providers fail."""
        from pixelbliss.run_once import generate_for_variant
        
        with patch('pixelbliss.providers.base.generate_image', return_value=None) as mock_generate:
            result = await generate_for_variant("test prompt", sample_config)
            
            assert len(result) == 0
            assert mock_generate.call_count == 2  # FAL + Replicate for each model

    @pytest.mark.asyncio
    async def test_generate_for_variant_with_semaphore(self, sample_config):
        """Test generate_for_variant with concurrency semaphore."""
        from pixelbliss.run_once import generate_for_variant
        
        mock_image = Mock()
        mock_result = {"image": mock_image, "provider": "fal", "model": "test_model", "seed": 123}
        semaphore = asyncio.Semaphore(1)
        
        with patch('pixelbliss.providers.base.generate_image', return_value=mock_result) as mock_generate:
            result = await generate_for_variant("test prompt", sample_config, semaphore)
            
            assert len(result) == 1
            assert result[0]["prompt"] == "test prompt"
            mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_for_variant_exception_handling(self, sample_config):
        """Test generate_for_variant handles exceptions gracefully."""
        from pixelbliss.run_once import generate_for_variant
        
        with patch('pixelbliss.providers.base.generate_image', side_effect=Exception("API Error")) as mock_generate:
            result = await generate_for_variant("test prompt", sample_config)
            
            assert len(result) == 0  # Should return empty list on exception

    @pytest.mark.asyncio
    async def test_run_all_variants_success(self, sample_config):
        """Test run_all_variants with successful generation."""
        from pixelbliss.run_once import run_all_variants
        
        mock_image = Mock()
        mock_result = {"image": mock_image, "provider": "fal", "model": "test_model", "seed": 123}
        
        with patch('pixelbliss.providers.base.generate_image', return_value=mock_result) as mock_generate:
            result = await run_all_variants(["prompt1", "prompt2"], sample_config)
            
            assert len(result) == 2  # One result per variant
            assert result[0]["prompt"] == "prompt1"
            assert result[1]["prompt"] == "prompt2"

    @pytest.mark.asyncio
    async def test_run_all_variants_with_concurrency_limit(self, sample_config):
        """Test run_all_variants with concurrency limit."""
        from pixelbliss.run_once import run_all_variants
        
        sample_config.image_generation.max_concurrency = 1
        mock_image = Mock()
        mock_result = {"image": mock_image, "provider": "fal", "model": "test_model", "seed": 123}
        
        with patch('pixelbliss.providers.base.generate_image', return_value=mock_result) as mock_generate:
            result = await run_all_variants(["prompt1", "prompt2"], sample_config)
            
            assert len(result) == 2
            assert result[0]["prompt"] == "prompt1"
            assert result[1]["prompt"] == "prompt2"

    @pytest.mark.asyncio
    async def test_run_all_variants_with_exceptions(self, sample_config):
        """Test run_all_variants handles variant exceptions."""
        from pixelbliss.run_once import run_all_variants
        
        mock_image = Mock()
        mock_result = {"image": mock_image, "provider": "fal", "model": "test_model", "seed": 123}
        
        # First variant succeeds, second fails
        with patch('pixelbliss.providers.base.generate_image', side_effect=[mock_result, Exception("API Error")]) as mock_generate:
            with patch('pixelbliss.alerts.webhook.send_failure') as mock_alert:
                result = await run_all_variants(["prompt1", "prompt2"], sample_config)
                
                assert len(result) == 1  # Only successful variant
                assert result[0]["prompt"] == "prompt1"

    @pytest.mark.asyncio
    async def test_run_all_variants_with_alert_failure(self, sample_config):
        """Test run_all_variants handles alert webhook failures gracefully."""
        from pixelbliss.run_once import run_all_variants
        
        # Mock generate_for_variant to raise an exception
        with patch('pixelbliss.run_once.generate_for_variant', side_effect=Exception("Variant Error")) as mock_generate:
            # Mock alerts.webhook.send_failure to also raise an exception
            with patch('pixelbliss.alerts.webhook.send_failure', side_effect=Exception("Alert Error")) as mock_alert:
                result = await run_all_variants(["prompt1"], sample_config)
                
                assert len(result) == 0  # No successful variants
                mock_alert.assert_called_once()  # Alert was attempted

    @pytest.mark.asyncio
    async def test_run_all_variants_no_concurrency_limit(self, sample_config):
        """Test run_all_variants with no concurrency limit (None)."""
        from pixelbliss.run_once import run_all_variants
        
        sample_config.image_generation.max_concurrency = None
        mock_image = Mock()
        mock_result = {"image": mock_image, "provider": "fal", "model": "test_model", "seed": 123}
        
        with patch('pixelbliss.providers.base.generate_image', return_value=mock_result) as mock_generate:
            result = await run_all_variants(["prompt1", "prompt2", "prompt3"], sample_config)
            
            assert len(result) == 3
            for i, res in enumerate(result):
                assert res["prompt"] == f"prompt{i+1}"

    def test_generate_images_sequential_success(self, sample_config):
        """Test generate_images_sequential with successful generation."""
        mock_image = Mock()
        mock_result = {"image": mock_image, "provider": "fal", "model": "test_model", "seed": 123}
        
        with patch('pixelbliss.providers.base.generate_image', return_value=mock_result) as mock_generate:
            result = generate_images_sequential(["prompt1", "prompt2"], sample_config)
            
            assert len(result) == 2
            assert result[0]["prompt"] == "prompt1"
            assert result[1]["prompt"] == "prompt2"

    def test_generate_images_sequential_fal_fails_replicate_succeeds(self, sample_config):
        """Test generate_images_sequential when FAL fails but Replicate succeeds."""
        mock_image = Mock()
        mock_result = {"image": mock_image, "provider": "replicate", "model": "test_model", "seed": 123}
        
        with patch('pixelbliss.providers.base.generate_image', side_effect=[None, mock_result]) as mock_generate:
            result = generate_images_sequential(["prompt1"], sample_config)
            
            assert len(result) == 1
            assert result[0]["prompt"] == "prompt1"
            assert result[0]["provider"] == "replicate"
            assert mock_generate.call_count == 2  # FAL then Replicate

    def test_generate_images_sequential_all_fail(self, sample_config):
        """Test generate_images_sequential when all providers fail."""
        with patch('pixelbliss.providers.base.generate_image', return_value=None) as mock_generate:
            result = generate_images_sequential(["prompt1"], sample_config)
            
            assert len(result) == 0
            assert mock_generate.call_count == 2  # FAL + Replicate for each model


class TestAsyncIntegration:
    """Test async integration with post_once function."""

    @patch('pixelbliss.run_once.config.load_config')
    @patch('pixelbliss.run_once.select_category')
    @patch('pixelbliss.run_once.prompts.make_base')
    @patch('pixelbliss.run_once.prompts.make_variants_from_base')
    @patch('pixelbliss.run_once.run_all_variants')
    @patch('pixelbliss.run_once.metrics.brightness')
    @patch('pixelbliss.run_once.metrics.entropy')
    @patch('pixelbliss.run_once.sanity.passes_floors')
    @patch('pixelbliss.run_once.quality.evaluate_local')
    @patch('pixelbliss.run_once.aesthetic.score_candidates_parallel')
    @patch('pixelbliss.run_once.normalize_and_rescore')
    @patch('pixelbliss.run_once.today_local')
    @patch('pixelbliss.run_once.storage.paths.make_slug')
    @patch('pixelbliss.run_once.storage.paths.output_dir')
    @patch('pixelbliss.run_once.collage.save_collage')
    @patch('pixelbliss.run_once.manifest.load_recent_hashes')
    @patch('pixelbliss.run_once.phash.is_duplicate')
    @patch('pixelbliss.run_once.phash.phash_hex')
    @patch('pixelbliss.imaging.variants.make_wallpaper_variants')
    @patch('pixelbliss.run_once.prompts.make_alt_text')
    @patch('pixelbliss.run_once.storage.fs.save_images')
    @patch('pixelbliss.run_once.storage.fs.save_meta')
    @patch('pixelbliss.run_once.manifest.append')
    @patch('pixelbliss.run_once.now_iso')
    @pytest.mark.asyncio
    async def test_post_once_async_enabled(self, mock_iso, mock_append, mock_save_meta, mock_save_images,
                                   mock_alt, mock_variants_wall, mock_phash, mock_duplicate, mock_hashes,
                                   mock_collage, mock_outdir, mock_slug, mock_today, mock_rescore,
                                   mock_score_parallel, mock_quality, mock_floors, mock_entropy, mock_brightness,
                                   mock_run_variants, mock_variants, mock_base, mock_category, mock_config):
        """Test post_once uses async generation when enabled."""
        # Setup config with async enabled
        mock_cfg = Mock()
        mock_cfg.categories = ["test"]  # Add categories list
        mock_cfg.image_generation.async_enabled = True
        mock_cfg.image_generation.model_fal = ["model1"]
        mock_cfg.image_generation.provider_order = ["fal", "replicate"]
        mock_cfg.image_generation.model_replicate = ["model2"]
        mock_cfg.image_generation.max_concurrency = 2  # Set to actual integer
        mock_cfg.prompt_generation.num_prompt_variants = 2
        mock_cfg.prompt_generation.async_enabled = False  # Use sync for simplicity
        mock_cfg.prompt_generation.provider = "dummy"  # Set valid provider
        mock_cfg.upscale.enabled = False
        mock_cfg.alerts.webhook_url_env = "WEBHOOK_URL"  # Set to actual string
        mock_cfg.discord.enabled = False  # Disable Discord for tests
        mock_config.return_value = mock_cfg
        
        # Setup mocks
        mock_category.return_value = "test"
        mock_base.return_value = "base prompt"
        mock_variants.return_value = ["variant1", "variant2"]
        
        # Mock async generation result
        mock_image = Mock()
        mock_candidates = [
            {"image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1", "brightness": 150, "entropy": 4.5, "local_quality": 0.8},
            {"image": mock_image, "provider": "fal", "model": "test", "seed": 124, "prompt": "variant2", "brightness": 150, "entropy": 4.5, "local_quality": 0.8}
        ]
        
        # Mock the async functions
        mock_run_variants.return_value = mock_candidates
        mock_score_parallel.return_value = [
            {**c, "aesthetic": 0.7} for c in mock_candidates
        ]
        
        # Mock scoring to pass all checks
        mock_brightness.return_value = 150
        mock_entropy.return_value = 4.5
        mock_floors.return_value = True
        mock_quality.return_value = (True, 0.8)
        
        # Mock rescoring
        mock_rescore.return_value = [{"final": 0.8, "image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1", "aesthetic": 0.7, "brightness": 150, "entropy": 4.5, "local_quality": 0.8}]
        
        # Mock file operations
        mock_today.return_value = "2024-01-01"
        mock_slug.return_value = "test_slug"
        mock_outdir.return_value = "/test/dir"
        mock_collage.return_value = "/test/collage.jpg"
        mock_hashes.return_value = []
        mock_duplicate.return_value = False
        mock_phash.return_value = "abc123"
        mock_variants_wall.return_value = {"desktop": mock_image}
        mock_alt.return_value = "alt text"
        mock_save_images.return_value = {"desktop": "/path/to/desktop.jpg"}
        mock_iso.return_value = "2024-01-01T12:00:00"
        
        result = await post_once(dry_run=True)
        
        assert result == 0
        # Verify async path was used
        mock_run_variants.assert_called_once()
        mock_score_parallel.assert_called_once()

    @patch('pixelbliss.run_once.config.load_config')
    @patch('pixelbliss.run_once.select_category')
    @patch('pixelbliss.run_once.prompts.make_base')
    @patch('pixelbliss.run_once.prompts.make_variants_from_base')
    @patch('pixelbliss.run_once.generate_images_sequential')
    @patch('pixelbliss.run_once.asyncio.run')  # Mock asyncio.run to prevent warnings
    @patch('pixelbliss.run_once.metrics.brightness')
    @patch('pixelbliss.run_once.metrics.entropy')
    @patch('pixelbliss.run_once.sanity.passes_floors')
    @patch('pixelbliss.run_once.quality.evaluate_local')
    @patch('pixelbliss.run_once.aesthetic.aesthetic')
    @patch('pixelbliss.run_once.normalize_and_rescore')
    @patch('pixelbliss.run_once.today_local')
    @patch('pixelbliss.run_once.storage.paths.make_slug')
    @patch('pixelbliss.run_once.storage.paths.output_dir')
    @patch('pixelbliss.run_once.collage.save_collage')
    @patch('pixelbliss.run_once.manifest.load_recent_hashes')
    @patch('pixelbliss.run_once.phash.is_duplicate')
    @patch('pixelbliss.run_once.phash.phash_hex')
    @patch('pixelbliss.imaging.variants.make_wallpaper_variants')
    @patch('pixelbliss.run_once.prompts.make_alt_text')
    @patch('pixelbliss.run_once.storage.fs.save_images')
    @patch('pixelbliss.run_once.storage.fs.save_meta')
    @patch('pixelbliss.run_once.manifest.append')
    @patch('pixelbliss.run_once.now_iso')
    @pytest.mark.asyncio
    async def test_post_once_async_disabled(self, mock_iso, mock_append, mock_save_meta, mock_save_images,
                                    mock_alt, mock_variants_wall, mock_phash, mock_duplicate, mock_hashes,
                                    mock_collage, mock_outdir, mock_slug, mock_today, mock_rescore,
                                    mock_aesthetic, mock_quality, mock_floors, mock_entropy, mock_brightness,
                                    mock_asyncio_run, mock_sequential, mock_variants, mock_base, mock_category, mock_config):
        """Test post_once uses sequential generation when async disabled."""
        # Setup config with async disabled
        mock_cfg = Mock()
        mock_cfg.categories = ["test"]  # Add categories list
        mock_cfg.image_generation.async_enabled = False
        mock_cfg.image_generation.model_fal = ["model1"]
        mock_cfg.image_generation.provider_order = ["fal", "replicate"]
        mock_cfg.image_generation.model_replicate = ["model2"]
        mock_cfg.prompt_generation.num_prompt_variants = 2
        mock_cfg.prompt_generation.async_enabled = False  # Use sync for simplicity
        mock_cfg.prompt_generation.provider = "dummy"  # Set valid provider
        mock_cfg.upscale.enabled = False
        mock_cfg.discord.enabled = False  # Disable Discord for tests
        mock_config.return_value = mock_cfg
        
        # Setup mocks
        mock_category.return_value = "test"
        mock_base.return_value = "base prompt"
        mock_variants.return_value = ["variant1", "variant2"]
        
        # Mock sequential generation result
        mock_image = Mock()
        mock_candidates = [
            {"image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1"}
        ]
        mock_sequential.return_value = mock_candidates
        
        # Mock scoring to pass all checks
        mock_brightness.return_value = 150
        mock_entropy.return_value = 4.5
        mock_floors.return_value = True
        mock_quality.return_value = (True, 0.8)
        mock_aesthetic.return_value = 0.7
        
        # Mock rescoring
        mock_rescore.return_value = [{"final": 0.8, "image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1", "aesthetic": 0.7, "brightness": 150, "entropy": 4.5, "local_quality": 0.8}]
        
        # Mock file operations
        mock_today.return_value = "2024-01-01"
        mock_slug.return_value = "test_slug"
        mock_outdir.return_value = "/test/dir"
        mock_collage.return_value = "/test/collage.jpg"
        mock_hashes.return_value = []
        mock_duplicate.return_value = False
        mock_phash.return_value = "abc123"
        mock_variants_wall.return_value = {"desktop": mock_image}
        mock_alt.return_value = "alt text"
        mock_save_images.return_value = {"desktop": "/path/to/desktop.jpg"}
        mock_iso.return_value = "2024-01-01T12:00:00"
        
        result = await post_once(dry_run=True)
        
        assert result == 0
        mock_sequential.assert_called_once()  # Verify sequential path was used
        mock_asyncio_run.assert_not_called()  # Verify async path was not used


class TestAsyncPromptGeneration:
    """Test async prompt generation functions."""


    @pytest.mark.asyncio
    async def test_dummy_provider_knobs_async_method(self):
        """Test dummy provider async knobs method."""
        from pixelbliss.prompt_engine.dummy_local import DummyLocalProvider
        
        provider = DummyLocalProvider()
        
        base_prompt = "A beautiful landscape"
        variant_knobs_list = [
            {"tone_curve": "high-key airy matte", "color_grade": "warm gold + muted", "surface_fx": "soft matte"},
            {"tone_curve": "low-key velvet deep", "color_grade": "cool silver + desaturated", "surface_fx": "pearl glow"}
        ]
        
        result = await provider.make_variants_with_knobs_async(
            base_prompt, 
            2, 
            variant_knobs_list
        )
        
        assert len(result) == 2
        assert "Variation 1" in result[0]
        assert "tone_curve: high-key airy matte" in result[0]
        assert "Variation 2" in result[1]
        assert "tone_curve: low-key velvet deep" in result[1]

    @patch('pixelbliss.run_once.config.load_config')
    @patch('pixelbliss.run_once.select_category')
    @patch('pixelbliss.run_once.prompts.make_base')
    @patch('pixelbliss.run_once.prompts.make_variants_from_base')
    @patch('pixelbliss.run_once.generate_images_sequential')
    @patch('pixelbliss.run_once.asyncio.run')  # Mock asyncio.run to prevent warnings
    @patch('pixelbliss.run_once.metrics.brightness')
    @patch('pixelbliss.run_once.metrics.entropy')
    @patch('pixelbliss.run_once.sanity.passes_floors')
    @patch('pixelbliss.run_once.quality.evaluate_local')
    @patch('pixelbliss.run_once.normalize_and_rescore')
    @patch('pixelbliss.run_once.today_local')
    @patch('pixelbliss.run_once.storage.paths.make_slug')
    @patch('pixelbliss.run_once.storage.paths.output_dir')
    @patch('pixelbliss.run_once.collage.save_collage')
    @patch('pixelbliss.run_once.manifest.load_recent_hashes')
    @patch('pixelbliss.run_once.phash.is_duplicate')
    @patch('pixelbliss.run_once.phash.phash_hex')
    @patch('pixelbliss.imaging.variants.make_wallpaper_variants')
    @patch('pixelbliss.run_once.prompts.make_alt_text')
    @patch('pixelbliss.run_once.storage.fs.save_images')
    @patch('pixelbliss.run_once.storage.fs.save_meta')
    @patch('pixelbliss.run_once.manifest.append')
    @patch('pixelbliss.run_once.now_iso')
    @pytest.mark.asyncio
    async def test_post_once_async_disabled_no_image_url(self, mock_iso, mock_append, mock_save_meta, mock_save_images,
                                                  mock_alt, mock_variants_wall, mock_phash, mock_duplicate, mock_hashes,
                                                  mock_collage, mock_outdir, mock_slug, mock_today, mock_rescore,
                                                  mock_quality, mock_floors, mock_entropy, mock_brightness,
                                                  mock_asyncio_run, mock_sequential, mock_variants, mock_base, mock_category, mock_config):
        """Test post_once uses sequential generation when async disabled and no image_url available."""
        # Setup config with async disabled
        mock_cfg = Mock()
        mock_cfg.categories = ["test"]  # Add categories list
        mock_cfg.image_generation.async_enabled = False
        mock_cfg.image_generation.model_fal = ["model1"]
        mock_cfg.image_generation.provider_order = ["fal", "replicate"]
        mock_cfg.image_generation.model_replicate = ["model2"]
        mock_cfg.prompt_generation.num_prompt_variants = 1
        mock_cfg.prompt_generation.async_enabled = False  # Explicitly disable async prompt generation
        mock_cfg.prompt_generation.provider = "dummy"  # Set valid provider
        mock_cfg.upscale.enabled = False
        mock_cfg.discord.enabled = False  # Disable Discord for tests
        mock_config.return_value = mock_cfg
        
        # Setup mocks
        mock_category.return_value = "test"
        mock_base.return_value = "base prompt"
        mock_variants.return_value = ["variant1"]
        
        # Mock sequential generation result WITHOUT image_url
        mock_image = Mock()
        mock_candidates = [
            {"image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1"}
            # Note: no "image_url" key - this will trigger the fallback a = 0.5 line
        ]
        mock_sequential.return_value = mock_candidates
        
        # Mock scoring to pass all checks
        mock_brightness.return_value = 150
        mock_entropy.return_value = 4.5
        mock_floors.return_value = True
        mock_quality.return_value = (True, 0.8)
        # Don't mock aesthetic.aesthetic - let it execute the real code path
        
        # Mock rescoring - need to include aesthetic score that will be set to 0.5
        mock_rescore.return_value = [{"final": 0.8, "image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1", "aesthetic": 0.5, "brightness": 150, "entropy": 4.5, "local_quality": 0.8}]
        
        # Mock file operations
        mock_today.return_value = "2024-01-01"
        mock_slug.return_value = "test_slug"
        mock_outdir.return_value = "/test/dir"
        mock_collage.return_value = "/test/collage.jpg"
        mock_hashes.return_value = []
        mock_duplicate.return_value = False
        mock_phash.return_value = "abc123"
        mock_variants_wall.return_value = {"desktop": mock_image}
        mock_alt.return_value = "alt text"
        mock_save_images.return_value = {"desktop": "/path/to/desktop.jpg"}
        mock_iso.return_value = "2024-01-01T12:00:00"
        
        result = await post_once(dry_run=True)
        
        assert result == 0
        mock_sequential.assert_called_once()  # Verify sequential path was used
        mock_asyncio_run.assert_not_called()  # Verify async path was not used

    @pytest.mark.asyncio
    async def test_generate_for_variant_with_progress_logger_and_semaphore(self, sample_config):
        """Test generate_for_variant with progress logger and semaphore to cover missing lines."""
        from pixelbliss.run_once import generate_for_variant
        
        mock_image = Mock()
        mock_result = {"image": mock_image, "provider": "fal", "model": "test_model", "seed": 123}
        mock_progress_logger = Mock()
        semaphore = asyncio.Semaphore(1)
        
        with patch('pixelbliss.providers.base.generate_image', return_value=mock_result) as mock_generate:
            result = await generate_for_variant("test prompt", sample_config, semaphore, mock_progress_logger, 0)
            
            assert len(result) == 1
            assert result[0]["prompt"] == "test prompt"
            assert result[0]["provider"] == "fal"
            mock_progress_logger.update_operation_progress.assert_called_once_with("image_generation")

    @pytest.mark.asyncio
    async def test_generate_for_variant_exception_with_progress_logger(self, sample_config):
        """Test generate_for_variant exception handling with progress logger."""
        from pixelbliss.run_once import generate_for_variant
        
        mock_progress_logger = Mock()
        
        with patch('pixelbliss.providers.base.generate_image', side_effect=Exception("API Error")) as mock_generate:
            # generate_for_variant catches exceptions and returns empty list
            result = await generate_for_variant("test prompt", sample_config, None, mock_progress_logger, 0)
            
            assert len(result) == 0  # Should return empty list on exception
            # Should still update progress even on exception
            mock_progress_logger.update_operation_progress.assert_called_once_with("image_generation")

    @pytest.mark.asyncio
    async def test_run_all_variants_with_progress_logger(self, sample_config):
        """Test run_all_variants with progress logger to cover missing lines."""
        from pixelbliss.run_once import run_all_variants
        
        mock_image = Mock()
        mock_result = {"image": mock_image, "provider": "fal", "model": "test_model", "seed": 123}
        mock_progress_logger = Mock()
        
        with patch('pixelbliss.providers.base.generate_image', return_value=mock_result) as mock_generate:
            result = await run_all_variants(["prompt1", "prompt2"], sample_config, mock_progress_logger)
            
            assert len(result) == 2
            mock_progress_logger.start_operation.assert_called_once_with("image_generation", 2, "parallel image generation")
            mock_progress_logger.finish_operation.assert_called_once_with("image_generation", True)

    @pytest.mark.asyncio
    async def test_run_all_variants_with_failures_and_progress_logger(self, sample_config):
        """Test run_all_variants with failures and progress logger to cover warning lines."""
        from pixelbliss.run_once import run_all_variants
        
        mock_progress_logger = Mock()
        
        # Mock generate_for_variant to raise exceptions
        with patch('pixelbliss.run_once.generate_for_variant', side_effect=Exception("Variant Error")) as mock_generate:
            with patch('pixelbliss.alerts.webhook.send_failure') as mock_alert:
                result = await run_all_variants(["prompt1", "prompt2"], sample_config, mock_progress_logger)
                
                assert len(result) == 0  # No successful variants
                mock_progress_logger.start_operation.assert_called_once_with("image_generation", 2, "parallel image generation")
                mock_progress_logger.finish_operation.assert_called_once_with("image_generation", False)
                mock_progress_logger.warning.assert_called_once_with("2 image generation variants failed")

    @patch('pixelbliss.run_once.config.load_config')
    @patch('pixelbliss.run_once.select_category')
    @patch('pixelbliss.run_once.prompts.make_base')
    @patch('pixelbliss.run_once.prompts.make_variants_from_base_async')
    @patch('pixelbliss.run_once.generate_images_sequential')
    @patch('pixelbliss.run_once.metrics.brightness')
    @patch('pixelbliss.run_once.metrics.entropy')
    @patch('pixelbliss.run_once.sanity.passes_floors')
    @patch('pixelbliss.run_once.quality.evaluate_local')
    @patch('pixelbliss.run_once.normalize_and_rescore')
    @patch('pixelbliss.run_once.today_local')
    @patch('pixelbliss.run_once.storage.paths.make_slug')
    @patch('pixelbliss.run_once.storage.paths.output_dir')
    @patch('pixelbliss.run_once.collage.save_collage')
    @patch('pixelbliss.run_once.manifest.load_recent_hashes')
    @patch('pixelbliss.run_once.phash.is_duplicate')
    @patch('pixelbliss.run_once.phash.phash_hex')
    @patch('pixelbliss.imaging.variants.make_wallpaper_variants')
    @patch('pixelbliss.run_once.prompts.make_alt_text')
    @patch('pixelbliss.run_once.storage.fs.save_images')
    @patch('pixelbliss.run_once.storage.fs.save_meta')
    @patch('pixelbliss.run_once.manifest.append')
    @patch('pixelbliss.run_once.now_iso')
    @pytest.mark.asyncio
    async def test_post_once_async_prompt_generation(self, mock_iso, mock_append, mock_save_meta, mock_save_images,
                                             mock_alt, mock_variants_wall, mock_phash, mock_duplicate, mock_hashes,
                                             mock_collage, mock_outdir, mock_slug, mock_today, mock_rescore,
                                             mock_quality, mock_floors, mock_entropy, mock_brightness,
                                             mock_sequential, mock_variants_async, mock_base, mock_category, mock_config):
        """Test post_once with async prompt generation enabled."""
        # Setup config with async prompt generation enabled
        mock_cfg = Mock()
        mock_cfg.categories = ["test"]
        mock_cfg.image_generation.async_enabled = False
        mock_cfg.image_generation.model_fal = ["model1"]
        mock_cfg.image_generation.provider_order = ["fal", "replicate"]
        mock_cfg.image_generation.model_replicate = ["model2"]
        mock_cfg.prompt_generation.num_prompt_variants = 2
        mock_cfg.prompt_generation.async_enabled = True  # Enable async prompt generation
        mock_cfg.prompt_generation.provider = "openai"
        mock_cfg.upscale.enabled = False
        mock_cfg.discord.enabled = False  # Disable Discord for tests
        mock_config.return_value = mock_cfg
        
        # Setup mocks
        mock_category.return_value = "test"
        mock_base.return_value = "base prompt"
        mock_variants_async.return_value = ["variant1", "variant2"]  # Async variant generation
        
        # Mock sequential generation result
        mock_image = Mock()
        mock_candidates = [
            {"image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1"}
        ]
        mock_sequential.return_value = mock_candidates
        
        # Mock scoring to pass all checks
        mock_brightness.return_value = 150
        mock_entropy.return_value = 4.5
        mock_floors.return_value = True
        mock_quality.return_value = (True, 0.8)
        
        # Mock rescoring
        mock_rescore.return_value = [{"final": 0.8, "image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1", "aesthetic": 0.5, "brightness": 150, "entropy": 4.5, "local_quality": 0.8}]
        
        # Mock file operations
        mock_today.return_value = "2024-01-01"
        mock_slug.return_value = "test_slug"
        mock_outdir.return_value = "/test/dir"
        mock_collage.return_value = "/test/collage.jpg"
        mock_hashes.return_value = []
        mock_duplicate.return_value = False
        mock_phash.return_value = "abc123"
        mock_variants_wall.return_value = {"desktop": mock_image}
        mock_alt.return_value = "alt text"
        mock_save_images.return_value = {"desktop": "/path/to/desktop.jpg"}
        mock_iso.return_value = "2024-01-01T12:00:00"
        
        result = await post_once(dry_run=True)
        
        assert result == 0
        mock_variants_async.assert_called_once()  # Verify async prompt generation was used

    @patch('pixelbliss.run_once.config.load_config')
    @patch('pixelbliss.run_once.select_category')
    @patch('pixelbliss.run_once.prompts.make_base')
    @patch('pixelbliss.run_once.prompts.make_variants_from_base')
    @patch('pixelbliss.run_once.generate_images_sequential')
    @patch('pixelbliss.run_once.metrics.brightness')
    @patch('pixelbliss.run_once.metrics.entropy')
    @patch('pixelbliss.run_once.sanity.passes_floors')
    @patch('pixelbliss.run_once.quality.evaluate_local')
    @patch('pixelbliss.run_once.aesthetic.aesthetic')
    @patch('pixelbliss.run_once.normalize_and_rescore')
    @patch('pixelbliss.run_once.today_local')
    @patch('pixelbliss.run_once.storage.paths.make_slug')
    @patch('pixelbliss.run_once.storage.paths.output_dir')
    @patch('pixelbliss.run_once.collage.save_collage')
    @patch('pixelbliss.run_once.manifest.load_recent_hashes')
    @patch('pixelbliss.run_once.phash.is_duplicate')
    @patch('pixelbliss.run_once.phash.phash_hex')
    @patch('pixelbliss.imaging.variants.make_wallpaper_variants')
    @patch('pixelbliss.run_once.prompts.make_alt_text')
    @patch('pixelbliss.run_once.storage.fs.save_images')
    @patch('pixelbliss.run_once.storage.fs.save_meta')
    @patch('pixelbliss.run_once.manifest.append')
    @patch('pixelbliss.run_once.now_iso')
    @patch('pixelbliss.twitter.client.upload_media')
    @patch('pixelbliss.twitter.client.set_alt_text')
    @patch('pixelbliss.twitter.client.create_tweet')
    @patch('pixelbliss.run_once.manifest.update_tweet_id')
    @patch('pixelbliss.alerts.webhook.send_success')
    @pytest.mark.asyncio
    async def test_post_once_upscale_success(self, mock_send_success, mock_update_tweet, mock_create_tweet, mock_set_alt,
                                     mock_upload, mock_iso, mock_append, mock_save_meta, mock_save_images, mock_alt,
                                     mock_variants_wall, mock_phash, mock_duplicate, mock_hashes, mock_collage,
                                     mock_outdir, mock_slug, mock_today, mock_rescore, mock_aesthetic, mock_quality,
                                     mock_floors, mock_entropy, mock_brightness, mock_sequential, mock_variants,
                                     mock_base, mock_category, mock_config):
        """Test post_once with successful upscaling."""
        # Setup config
        mock_cfg = Mock()
        mock_cfg.categories = ["test"]
        mock_cfg.image_generation.async_enabled = False
        mock_cfg.image_generation.model_fal = ["model1"]
        mock_cfg.image_generation.provider_order = ["fal", "replicate"]
        mock_cfg.image_generation.model_replicate = ["model2"]
        mock_cfg.prompt_generation.num_prompt_variants = 1
        mock_cfg.prompt_generation.async_enabled = False
        mock_cfg.prompt_generation.provider = "dummy"
        mock_cfg.upscale.enabled = True
        mock_cfg.upscale.provider = "test_provider"
        mock_cfg.upscale.model = "test_model"
        mock_cfg.upscale.factor = 2
        mock_cfg.discord.enabled = False  # Disable Discord for tests
        mock_config.return_value = mock_cfg
        
        # Setup mocks
        mock_category.return_value = "test"
        mock_base.return_value = "base prompt"
        mock_variants.return_value = ["variant1"]
        
        # Mock successful image generation
        mock_image = Mock()
        mock_upscaled_image = Mock()
        mock_sequential.return_value = [{"image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1"}]
        
        # Mock scoring to pass all checks
        mock_brightness.return_value = 150
        mock_entropy.return_value = 4.5
        mock_floors.return_value = True
        mock_quality.return_value = (True, 0.8)
        mock_aesthetic.return_value = 0.7
        
        # Mock rescoring
        mock_rescore.return_value = [{"final": 0.8, "image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1", "aesthetic": 0.7, "brightness": 150, "entropy": 4.5, "local_quality": 0.8}]
        
        # Mock file operations
        mock_today.return_value = "2024-01-01"
        mock_slug.return_value = "test_slug"
        mock_outdir.return_value = "/test/dir"
        mock_collage.return_value = "/test/collage.jpg"
        mock_hashes.return_value = []
        mock_duplicate.return_value = False
        mock_phash.return_value = "abc123"
        
        # Mock successful upscale
        with patch('pixelbliss.providers.upscale.upscale') as mock_upscale:
            mock_upscale.return_value = mock_upscaled_image  # Upscale succeeds
            
            mock_variants_wall.return_value = {"desktop": mock_upscaled_image}
            mock_alt.return_value = "alt text"
            mock_save_images.return_value = {"desktop": "/path/to/desktop.jpg", "base_img": "/path/to/base_img.png"}
            mock_iso.return_value = "2024-01-01T12:00:00"
            
            # Mock Twitter operations
            mock_upload.return_value = ["media_id_123"]
            mock_create_tweet.return_value = "tweet_id_456"
            
            result = await post_once(dry_run=False)
            
            assert result == 0
            mock_upscale.assert_called_once_with(mock_image, "test_provider", "test_model", 2)

    @patch('pixelbliss.run_once.config.load_config')
    @patch('pixelbliss.run_once.select_category')
    @patch('pixelbliss.run_once.prompts.make_base')
    @patch('pixelbliss.run_once.prompts.make_variants_from_base')
    @patch('pixelbliss.run_once.generate_images_sequential')
    @patch('pixelbliss.run_once.metrics.brightness')
    @patch('pixelbliss.run_once.metrics.entropy')
    @patch('pixelbliss.run_once.sanity.passes_floors')
    @patch('pixelbliss.run_once.quality.evaluate_local')
    @patch('pixelbliss.run_once.aesthetic.aesthetic')
    @patch('pixelbliss.run_once.normalize_and_rescore')
    @patch('pixelbliss.run_once.today_local')
    @patch('pixelbliss.run_once.storage.paths.make_slug')
    @patch('pixelbliss.run_once.storage.paths.output_dir')
    @patch('pixelbliss.run_once.collage.save_collage')
    @patch('pixelbliss.run_once.manifest.load_recent_hashes')
    @patch('pixelbliss.run_once.phash.is_duplicate')
    @patch('pixelbliss.run_once.phash.phash_hex')
    @patch('pixelbliss.imaging.variants.make_wallpaper_variants')
    @patch('pixelbliss.run_once.prompts.make_alt_text')
    @patch('pixelbliss.run_once.storage.fs.save_images')
    @patch('pixelbliss.run_once.storage.fs.save_meta')
    @patch('pixelbliss.run_once.manifest.append')
    @patch('pixelbliss.run_once.now_iso')
    @patch('pixelbliss.twitter.client.upload_media')
    @patch('pixelbliss.twitter.client.set_alt_text')
    @patch('pixelbliss.twitter.client.create_tweet')
    @pytest.mark.asyncio
    async def test_post_once_social_media_exception(self, mock_create_tweet, mock_set_alt, mock_upload, mock_iso,
                                            mock_append, mock_save_meta, mock_save_images, mock_alt, mock_variants_wall,
                                            mock_phash, mock_duplicate, mock_hashes, mock_collage, mock_outdir,
                                            mock_slug, mock_today, mock_rescore, mock_aesthetic, mock_quality,
                                            mock_floors, mock_entropy, mock_brightness, mock_sequential, mock_variants,
                                            mock_base, mock_category, mock_config):
        """Test post_once with social media posting exception."""
        # Setup config
        mock_cfg = Mock()
        mock_cfg.categories = ["test"]
        mock_cfg.image_generation.async_enabled = False
        mock_cfg.image_generation.model_fal = ["model1"]
        mock_cfg.image_generation.provider_order = ["fal", "replicate"]
        mock_cfg.image_generation.model_replicate = ["model2"]
        mock_cfg.prompt_generation.num_prompt_variants = 1
        mock_cfg.prompt_generation.async_enabled = False
        mock_cfg.prompt_generation.provider = "dummy"
        mock_cfg.upscale.enabled = False
        mock_cfg.discord.enabled = False  # Disable Discord for tests
        mock_config.return_value = mock_cfg
        
        # Setup mocks
        mock_category.return_value = "test"
        mock_base.return_value = "base prompt"
        mock_variants.return_value = ["variant1"]
        
        # Mock successful image generation
        mock_image = Mock()
        mock_sequential.return_value = [{"image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1"}]
        
        # Mock scoring to pass all checks
        mock_brightness.return_value = 150
        mock_entropy.return_value = 4.5
        mock_floors.return_value = True
        mock_quality.return_value = (True, 0.8)
        mock_aesthetic.return_value = 0.7
        
        # Mock rescoring
        mock_rescore.return_value = [{"final": 0.8, "image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1", "aesthetic": 0.7, "brightness": 150, "entropy": 4.5, "local_quality": 0.8}]
        
        # Mock file operations
        mock_today.return_value = "2024-01-01"
        mock_slug.return_value = "test_slug"
        mock_outdir.return_value = "/test/dir"
        mock_collage.return_value = "/test/collage.jpg"
        mock_hashes.return_value = []
        mock_duplicate.return_value = False
        mock_phash.return_value = "abc123"
        mock_variants_wall.return_value = {"desktop": mock_image}
        mock_alt.return_value = "alt text"
        mock_save_images.return_value = {"desktop": "/path/to/desktop.jpg", "base_img": "/path/to/base_img.png"}
        mock_iso.return_value = "2024-01-01T12:00:00"
        
        # Mock Twitter operations to fail
        mock_upload.side_effect = Exception("Twitter API Error")
        
        result = await post_once(dry_run=False)
        
        assert result == 1  # Should return error code
        mock_upload.assert_called_once()

    @patch('pixelbliss.run_once.config.load_config')
    @pytest.mark.asyncio
    async def test_post_once_pipeline_exception(self, mock_config):
        """Test post_once with pipeline exception."""
        # Mock config loading to raise an exception
        mock_config.side_effect = Exception("Config loading failed")
        
        result = await post_once(dry_run=True)
        
        assert result == 1  # Should return error code

    @patch('pixelbliss.run_once.config.load_config')
    @patch('pixelbliss.run_once.select_category')
    @patch('pixelbliss.run_once.prompts.make_base')
    @patch('pixelbliss.run_once.prompts.make_variants_from_base')
    @patch('pixelbliss.run_once.generate_images_sequential')
    @patch('pixelbliss.run_once.metrics.brightness')
    @patch('pixelbliss.run_once.metrics.entropy')
    @patch('pixelbliss.run_once.sanity.passes_floors')
    @patch('pixelbliss.run_once.quality.evaluate_local')
    @patch('pixelbliss.run_once.aesthetic.aesthetic')
    @patch('pixelbliss.run_once.normalize_and_rescore')
    @patch('pixelbliss.run_once.today_local')
    @patch('pixelbliss.run_once.storage.paths.make_slug')
    @patch('pixelbliss.run_once.storage.paths.output_dir')
    @patch('pixelbliss.run_once.collage.save_collage')
    @patch('pixelbliss.run_once.manifest.load_recent_hashes')
    @patch('pixelbliss.run_once.phash.is_duplicate')
    @patch('pixelbliss.run_once.phash.phash_hex')
    @patch('pixelbliss.imaging.variants.make_wallpaper_variants')
    @patch('pixelbliss.run_once.prompts.make_alt_text')
    @patch('pixelbliss.run_once.storage.fs.save_images')
    @patch('pixelbliss.run_once.storage.fs.save_meta')
    @patch('pixelbliss.run_once.manifest.append')
    @patch('pixelbliss.run_once.now_iso')
    @patch('pixelbliss.alerts.discord_select.ask_user_to_select_raw')
    @pytest.mark.asyncio
    async def test_post_once_discord_human_selection_success(self, mock_discord_select, mock_iso, mock_append, mock_save_meta, mock_save_images,
                                                     mock_alt, mock_variants_wall, mock_phash, mock_duplicate, mock_hashes,
                                                     mock_collage, mock_outdir, mock_slug, mock_today, mock_rescore,
                                                     mock_aesthetic, mock_quality, mock_floors, mock_entropy, mock_brightness,
                                                     mock_sequential, mock_variants, mock_base, mock_category, mock_config):
        """Test post_once with Discord human selection enabled and successful selection."""
        # Setup config with Discord enabled
        mock_cfg = Mock()
        mock_cfg.categories = ["test"]
        mock_cfg.image_generation.async_enabled = False
        mock_cfg.image_generation.model_fal = ["model1"]
        mock_cfg.image_generation.provider_order = ["fal", "replicate"]
        mock_cfg.image_generation.model_replicate = ["model2"]
        mock_cfg.prompt_generation.num_prompt_variants = 1
        mock_cfg.prompt_generation.async_enabled = False
        mock_cfg.prompt_generation.provider = "dummy"
        mock_cfg.upscale.enabled = False
        mock_cfg.discord.enabled = True  # Enable Discord for this test
        mock_config.return_value = mock_cfg
        
        # Setup mocks
        mock_category.return_value = "test"
        mock_base.return_value = "base prompt"
        mock_variants.return_value = ["variant1"]
        
        # Mock successful image generation
        mock_image = Mock()
        mock_candidates = [
            {"image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1"},
            {"image": mock_image, "provider": "replicate", "model": "test2", "seed": 124, "prompt": "variant1"}
        ]
        mock_sequential.return_value = mock_candidates
        
        # Mock successful Discord selection (user selects second candidate)
        mock_discord_select.return_value = mock_candidates[1]  # User selects second candidate
        
        # Mock file operations
        mock_today.return_value = "2024-01-01"
        mock_slug.return_value = "test_slug"
        mock_outdir.return_value = "/test/dir"
        mock_phash.return_value = "abc123"
        mock_variants_wall.return_value = {"desktop": mock_image}
        mock_alt.return_value = "alt text"
        mock_save_images.return_value = {"desktop": "/path/to/desktop.jpg"}
        mock_iso.return_value = "2024-01-01T12:00:00"
        
        result = await post_once(dry_run=True)
        
        assert result == 0
        # Verify Discord selection was called
        mock_discord_select.assert_called_once_with(mock_candidates, mock_cfg, ANY)
        # Verify metadata includes human selection info
        mock_save_meta.assert_called_once()
        saved_meta = mock_save_meta.call_args[0][1]
        assert saved_meta["human_selection"]["enabled"] is True
        assert saved_meta["human_selection"]["selected_rank_raw"] == 2  # Second candidate
        assert saved_meta["human_selection"]["timeout_fallback"] is False

    @patch('pixelbliss.run_once.config.load_config')
    @patch('pixelbliss.run_once.select_category')
    @patch('pixelbliss.run_once.prompts.make_base')
    @patch('pixelbliss.run_once.prompts.make_variants_from_base')
    @patch('pixelbliss.run_once.generate_images_sequential')
    @patch('pixelbliss.run_once.metrics.brightness')
    @patch('pixelbliss.run_once.metrics.entropy')
    @patch('pixelbliss.run_once.sanity.passes_floors')
    @patch('pixelbliss.run_once.quality.evaluate_local')
    @patch('pixelbliss.run_once.aesthetic.aesthetic')
    @patch('pixelbliss.run_once.normalize_and_rescore')
    @patch('pixelbliss.run_once.today_local')
    @patch('pixelbliss.run_once.storage.paths.make_slug')
    @patch('pixelbliss.run_once.storage.paths.output_dir')
    @patch('pixelbliss.run_once.collage.save_collage')
    @patch('pixelbliss.run_once.manifest.load_recent_hashes')
    @patch('pixelbliss.run_once.phash.is_duplicate')
    @patch('pixelbliss.run_once.phash.phash_hex')
    @patch('pixelbliss.imaging.variants.make_wallpaper_variants')
    @patch('pixelbliss.run_once.prompts.make_alt_text')
    @patch('pixelbliss.run_once.storage.fs.save_images')
    @patch('pixelbliss.run_once.storage.fs.save_meta')
    @patch('pixelbliss.run_once.manifest.append')
    @patch('pixelbliss.run_once.now_iso')
    @patch('pixelbliss.alerts.discord_select.ask_user_to_select_raw')
    @pytest.mark.asyncio
    async def test_post_once_discord_human_selection_timeout(self, mock_discord_select, mock_iso, mock_append, mock_save_meta, mock_save_images,
                                                     mock_alt, mock_variants_wall, mock_phash, mock_duplicate, mock_hashes,
                                                     mock_collage, mock_outdir, mock_slug, mock_today, mock_rescore,
                                                     mock_aesthetic, mock_quality, mock_floors, mock_entropy, mock_brightness,
                                                     mock_sequential, mock_variants, mock_base, mock_category, mock_config):
        """Test post_once with Discord human selection timeout fallback."""
        # Setup config with Discord enabled
        mock_cfg = Mock()
        mock_cfg.categories = ["test"]
        mock_cfg.image_generation.async_enabled = False
        mock_cfg.image_generation.model_fal = ["model1"]
        mock_cfg.image_generation.provider_order = ["fal", "replicate"]
        mock_cfg.image_generation.model_replicate = ["model2"]
        mock_cfg.prompt_generation.num_prompt_variants = 1
        mock_cfg.prompt_generation.async_enabled = False
        mock_cfg.prompt_generation.provider = "dummy"
        mock_cfg.upscale.enabled = False
        mock_cfg.discord.enabled = True  # Enable Discord for this test
        mock_config.return_value = mock_cfg
        
        # Setup mocks
        mock_category.return_value = "test"
        mock_base.return_value = "base prompt"
        mock_variants.return_value = ["variant1"]
        
        # Mock successful image generation
        mock_image = Mock()
        mock_candidates = [
            {"image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1"}
        ]
        mock_sequential.return_value = mock_candidates
        
        # Mock Discord selection timeout (returns None)
        mock_discord_select.return_value = None
        
        # Mock file operations
        mock_today.return_value = "2024-01-01"
        mock_slug.return_value = "test_slug"
        mock_outdir.return_value = "/test/dir"
        mock_phash.return_value = "abc123"
        mock_variants_wall.return_value = {"desktop": mock_image}
        mock_alt.return_value = "alt text"
        mock_save_images.return_value = {"desktop": "/path/to/desktop.jpg"}
        mock_iso.return_value = "2024-01-01T12:00:00"
        
        result = await post_once(dry_run=True)
        
        assert result == 0
        # Verify Discord selection was called
        mock_discord_select.assert_called_once_with(mock_candidates, mock_cfg, ANY)
        # Verify metadata includes timeout fallback info
        mock_save_meta.assert_called_once()
        saved_meta = mock_save_meta.call_args[0][1]
        assert saved_meta["human_selection"]["enabled"] is True
        assert saved_meta["human_selection"]["selected_rank_raw"] == 1  # First candidate (fallback)
        assert saved_meta["human_selection"]["timeout_fallback"] is True
