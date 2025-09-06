import pytest
import datetime
import pytz
import asyncio
from unittest.mock import Mock, patch, AsyncMock, ANY
from pixelbliss.run_once import (
    normalize_and_rescore, today_local, now_iso, tweet_url, fs_abs,
    try_in_order, post_once
)


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


class TestPostOnceIntegration:
    """Test post_once integration with simplified mocking."""

    @patch('pixelbliss.run_once.config.load_config')
    @patch('pixelbliss.run_once.generate_theme_hint_async')
    @patch('pixelbliss.run_once.prompts.make_base')
    @patch('pixelbliss.run_once.prompts.make_variants_from_base')
    @patch('pixelbliss.run_once.run_all_variants')
    @patch('pixelbliss.alerts.webhook.send_failure')
    @pytest.mark.asyncio
    async def test_post_once_no_candidates(self, mock_send_failure, mock_run_variants, mock_variants, 
                                   mock_base, mock_theme, mock_config):
        """Test post_once when no image candidates are generated."""
        # Setup mocks
        mock_cfg = Mock()
        mock_cfg.trending_themes.enabled = True
        mock_cfg.image_generation.model_fal = ["model1"]
        mock_cfg.image_generation.provider_order = ["fal", "replicate"]
        mock_cfg.image_generation.model_replicate = ["model2"]
        mock_cfg.image_generation.max_concurrency = None
        mock_cfg.prompt_generation.num_prompt_variants = 1
        mock_cfg.prompt_generation.async_enabled = False  # Use sync for simplicity
        mock_cfg.prompt_generation.provider = "dummy"  # Set valid provider
        mock_cfg.discord.enabled = False  # Disable Discord for tests
        mock_config.return_value = mock_cfg
        
        mock_theme.return_value = "abstract"
        mock_base.return_value = "base prompt"
        mock_variants.return_value = ["variant1", "variant2"]
        mock_run_variants.return_value = []  # No images generated
        
        result = await post_once(dry_run=True)
        
        assert result == 1  # Should return error code
        mock_send_failure.assert_called_once()

    @patch('pixelbliss.run_once.config.load_config')
    @pytest.mark.asyncio
    async def test_post_once_pipeline_exception(self, mock_config):
        """Test post_once with pipeline exception."""
        # Mock config loading to raise an exception
        mock_config.side_effect = Exception("Config loading failed")
        
        result = await post_once(dry_run=True)
        
        assert result == 1  # Should return error code

    @patch('pixelbliss.run_once.config.load_config')
    @patch('pixelbliss.run_once.generate_theme_hint_async')
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
    async def test_post_once_success_path(self, mock_iso, mock_append, mock_save_meta, mock_save_images,
                                 mock_alt, mock_variants_wall, mock_phash, mock_duplicate, mock_hashes,
                                 mock_collage, mock_outdir, mock_slug, mock_today, mock_rescore,
                                 mock_score_parallel, mock_quality, mock_floors, mock_entropy, mock_brightness,
                                 mock_run_variants, mock_variants, mock_base, mock_theme, mock_config):
        """Test post_once success path with full pipeline."""
        # Setup mocks
        mock_cfg = Mock()
        mock_cfg.trending_themes.enabled = True
        mock_cfg.image_generation.model_fal = ["model1"]
        mock_cfg.image_generation.provider_order = ["fal", "replicate"]
        mock_cfg.image_generation.model_replicate = ["model2"]
        mock_cfg.image_generation.max_concurrency = None
        mock_cfg.prompt_generation.num_prompt_variants = 1
        mock_cfg.prompt_generation.async_enabled = False
        mock_cfg.prompt_generation.provider = "dummy"
        mock_cfg.discord.enabled = False
        mock_cfg.upscale.enabled = False
        mock_config.return_value = mock_cfg
        
        # Mock successful pipeline
        mock_theme.return_value = "abstract"
        mock_base.return_value = "base prompt"
        mock_variants.return_value = ["variant1"]
        
        mock_image = Mock()
        mock_candidates = [{"image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1"}]
        mock_run_variants.return_value = mock_candidates
        
        # Mock scoring
        mock_brightness.return_value = 150
        mock_entropy.return_value = 4.5
        mock_floors.return_value = True
        mock_quality.return_value = (True, 0.8)
        mock_score_parallel.return_value = [{"image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1", "aesthetic": 0.7, "brightness": 150, "entropy": 4.5, "local_quality": 0.8}]
        
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

    @patch('pixelbliss.run_once.config.load_config')
    @patch('pixelbliss.run_once.generate_theme_hint_async')
    @patch('pixelbliss.run_once.prompts.make_base')
    @patch('pixelbliss.run_once.prompts.make_variants_from_base_async')
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
    async def test_post_once_async_prompt_generation(self, mock_iso, mock_append, mock_save_meta, mock_save_images,
                                             mock_alt, mock_variants_wall, mock_phash, mock_duplicate, mock_hashes,
                                             mock_collage, mock_outdir, mock_slug, mock_today, mock_rescore,
                                             mock_score_parallel, mock_quality, mock_floors, mock_entropy, mock_brightness,
                                             mock_run_variants, mock_variants_async, mock_base, mock_theme, mock_config):
        """Test post_once with async prompt generation enabled."""
        # Setup mocks
        mock_cfg = Mock()
        mock_cfg.trending_themes.enabled = True
        mock_cfg.image_generation.model_fal = ["model1"]
        mock_cfg.image_generation.provider_order = ["fal", "replicate"]
        mock_cfg.image_generation.model_replicate = ["model2"]
        mock_cfg.image_generation.max_concurrency = None
        mock_cfg.prompt_generation.num_prompt_variants = 1
        mock_cfg.prompt_generation.async_enabled = True  # Enable async prompt generation
        mock_cfg.prompt_generation.provider = "dummy"
        mock_cfg.discord.enabled = False
        mock_cfg.upscale.enabled = False
        mock_config.return_value = mock_cfg
        
        # Mock successful pipeline
        mock_theme.return_value = "abstract"
        mock_base.return_value = "base prompt"
        mock_variants_async.return_value = ["variant1"]  # Async variant generation
        
        mock_image = Mock()
        mock_candidates = [{"image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1"}]
        mock_run_variants.return_value = mock_candidates
        
        # Mock scoring
        mock_brightness.return_value = 150
        mock_entropy.return_value = 4.5
        mock_floors.return_value = True
        mock_quality.return_value = (True, 0.8)
        mock_score_parallel.return_value = [{"image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1", "aesthetic": 0.7, "brightness": 150, "entropy": 4.5, "local_quality": 0.8}]
        
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
        mock_variants_async.assert_called_once()  # Verify async prompt generation was used

    @patch('pixelbliss.run_once.config.load_config')
    @patch('pixelbliss.run_once.generate_theme_hint_async')
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
    @patch('pixelbliss.alerts.discord_select.ask_user_to_select_raw')
    @pytest.mark.asyncio
    async def test_post_once_discord_selection_success(self, mock_discord_select, mock_iso, mock_append, mock_save_meta, mock_save_images,
                                               mock_alt, mock_variants_wall, mock_phash, mock_duplicate, mock_hashes,
                                               mock_collage, mock_outdir, mock_slug, mock_today, mock_rescore,
                                               mock_score_parallel, mock_quality, mock_floors, mock_entropy, mock_brightness,
                                               mock_run_variants, mock_variants, mock_base, mock_theme, mock_config):
        """Test post_once with Discord human selection."""
        # Setup mocks
        mock_cfg = Mock()
        mock_cfg.trending_themes.enabled = True
        mock_cfg.image_generation.model_fal = ["model1"]
        mock_cfg.image_generation.provider_order = ["fal", "replicate"]
        mock_cfg.image_generation.model_replicate = ["model2"]
        mock_cfg.image_generation.max_concurrency = None
        mock_cfg.prompt_generation.num_prompt_variants = 1
        mock_cfg.prompt_generation.async_enabled = False
        mock_cfg.prompt_generation.provider = "dummy"
        mock_cfg.discord.enabled = True  # Enable Discord
        mock_cfg.upscale.enabled = False
        mock_config.return_value = mock_cfg
        
        # Mock successful pipeline
        mock_theme.return_value = "abstract"
        mock_base.return_value = "base prompt"
        mock_variants.return_value = ["variant1"]
        
        mock_image = Mock()
        mock_candidates = [{"image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1"}]
        mock_run_variants.return_value = mock_candidates
        
        # Mock Discord selection
        mock_discord_select.return_value = mock_candidates[0]
        
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
        mock_discord_select.assert_called_once()

    @patch('pixelbliss.run_once.config.load_config')
    @patch('pixelbliss.run_once.generate_theme_hint_async')
    @patch('pixelbliss.run_once.prompts.make_base')
    @patch('pixelbliss.run_once.prompts.make_variants_from_base')
    @patch('pixelbliss.run_once.run_all_variants')
    @patch('pixelbliss.alerts.discord_select.ask_user_to_select_raw')
    @pytest.mark.asyncio
    async def test_post_once_discord_selection_timeout(self, mock_discord_select, mock_run_variants, mock_variants, 
                                               mock_base, mock_theme, mock_config):
        """Test post_once with Discord selection timeout."""
        # Setup mocks
        mock_cfg = Mock()
        mock_cfg.trending_themes.enabled = True
        mock_cfg.image_generation.model_fal = ["model1"]
        mock_cfg.image_generation.provider_order = ["fal", "replicate"]
        mock_cfg.image_generation.model_replicate = ["model2"]
        mock_cfg.image_generation.max_concurrency = None
        mock_cfg.prompt_generation.num_prompt_variants = 1
        mock_cfg.prompt_generation.async_enabled = False
        mock_cfg.prompt_generation.provider = "dummy"
        mock_cfg.discord.enabled = True  # Enable Discord
        mock_config.return_value = mock_cfg
        
        # Mock successful pipeline
        mock_theme.return_value = "abstract"
        mock_base.return_value = "base prompt"
        mock_variants.return_value = ["variant1"]
        
        mock_image = Mock()
        mock_candidates = [{"image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1"}]
        mock_run_variants.return_value = mock_candidates
        
        # Mock Discord selection timeout
        mock_discord_select.return_value = None
        
        result = await post_once(dry_run=True)
        
        assert result == 0  # Should end cleanly
        mock_discord_select.assert_called_once()

    @patch('pixelbliss.run_once.config.load_config')
    @patch('pixelbliss.run_once.generate_theme_hint_async')
    @patch('pixelbliss.run_once.prompts.make_base')
    @patch('pixelbliss.run_once.prompts.make_variants_from_base')
    @patch('pixelbliss.run_once.run_all_variants')
    @patch('pixelbliss.alerts.discord_select.ask_user_to_select_raw')
    @pytest.mark.asyncio
    async def test_post_once_discord_selection_none(self, mock_discord_select, mock_run_variants, mock_variants, 
                                            mock_base, mock_theme, mock_config):
        """Test post_once with Discord selection 'none'."""
        # Setup mocks
        mock_cfg = Mock()
        mock_cfg.trending_themes.enabled = True
        mock_cfg.image_generation.model_fal = ["model1"]
        mock_cfg.image_generation.provider_order = ["fal", "replicate"]
        mock_cfg.image_generation.model_replicate = ["model2"]
        mock_cfg.image_generation.max_concurrency = None
        mock_cfg.prompt_generation.num_prompt_variants = 1
        mock_cfg.prompt_generation.async_enabled = False
        mock_cfg.prompt_generation.provider = "dummy"
        mock_cfg.discord.enabled = True  # Enable Discord
        mock_config.return_value = mock_cfg
        
        # Mock successful pipeline
        mock_theme.return_value = "abstract"
        mock_base.return_value = "base prompt"
        mock_variants.return_value = ["variant1"]
        
        mock_image = Mock()
        mock_candidates = [{"image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1"}]
        mock_run_variants.return_value = mock_candidates
        
        # Mock Discord selection 'none'
        mock_discord_select.return_value = "none"
        
        result = await post_once(dry_run=True)
        
        assert result == 0  # Should end cleanly
        mock_discord_select.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_theme_hint_async_trending_disabled(self):
        """Test generate_theme_hint_async with trending disabled."""
        from pixelbliss.run_once import generate_theme_hint_async
        from pixelbliss.config import Config
        
        cfg = Config()
        cfg.trending_themes.enabled = False
        
        result = await generate_theme_hint_async(cfg)
        
        # Should return one of the fallback themes
        expected_themes = [
            "abstract", "nature", "cosmic", "geometric", "organic", "crystalline", "flow",
            "balance", "harmony", "unity", "duality", "symmetry", "asymmetry",
            "cycles", "growth", "renewal", "emergence", "evolution",
            "interconnection", "networks", "continuum", "wholeness", "infinity",
            "order and randomness", "pattern", "repetition", "rhythm",
            "fractal", "spirals", "tessellation", "lattice", "grid",
            "waveforms", "fields", "orbits", "constellations", "topography", "cartography",
            "elemental", "terrestrial", "celestial", "aquatic", "mineral",
            "botanical", "aerial", "seasonal", "weather",
            "journey", "thresholds", "liminality", "sanctuary", "play",
            "curiosity", "wonder", "stillness", "openness", "simplicity",
            "order and flow", "cause and effect", "microcosm and macrocosm"
        ]
        assert result in expected_themes

    @patch('pixelbliss.run_once.config.load_config')
    @patch('pixelbliss.run_once.generate_theme_hint_async')
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
    @patch('pixelbliss.twitter.client.upload_media')
    @patch('pixelbliss.twitter.client.set_alt_text')
    @patch('pixelbliss.twitter.client.create_tweet')
    @patch('pixelbliss.run_once.manifest.update_tweet_id')
    @patch('pixelbliss.alerts.webhook.send_success')
    @pytest.mark.asyncio
    async def test_post_once_full_execution_with_twitter(self, mock_send_success, mock_update_tweet, mock_create_tweet, mock_set_alt,
                                                mock_upload, mock_iso, mock_append, mock_save_meta, mock_save_images,
                                                mock_alt, mock_variants_wall, mock_phash, mock_duplicate, mock_hashes,
                                                mock_collage, mock_outdir, mock_slug, mock_today, mock_rescore,
                                                mock_score_parallel, mock_quality, mock_floors, mock_entropy, mock_brightness,
                                                mock_run_variants, mock_variants, mock_base, mock_theme, mock_config):
        """Test post_once full execution with Twitter posting."""
        # Setup mocks
        mock_cfg = Mock()
        mock_cfg.trending_themes.enabled = True
        mock_cfg.image_generation.model_fal = ["model1"]
        mock_cfg.image_generation.provider_order = ["fal", "replicate"]
        mock_cfg.image_generation.model_replicate = ["model2"]
        mock_cfg.image_generation.max_concurrency = None
        mock_cfg.prompt_generation.num_prompt_variants = 1
        mock_cfg.prompt_generation.async_enabled = False
        mock_cfg.prompt_generation.provider = "dummy"
        mock_cfg.discord.enabled = False
        mock_cfg.upscale.enabled = True
        mock_cfg.upscale.provider = "test_provider"
        mock_cfg.upscale.model = "test_model"
        mock_cfg.upscale.factor = 2
        mock_config.return_value = mock_cfg
        
        # Mock successful pipeline
        mock_theme.return_value = "abstract"
        mock_base.return_value = "base prompt"
        mock_variants.return_value = ["variant1"]
        
        mock_image = Mock()
        mock_upscaled_image = Mock()
        mock_candidates = [{"image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1"}]
        mock_run_variants.return_value = mock_candidates
        
        # Mock scoring
        mock_brightness.return_value = 150
        mock_entropy.return_value = 4.5
        mock_floors.return_value = True
        mock_quality.return_value = (True, 0.8)
        mock_score_parallel.return_value = [{"image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1", "aesthetic": 0.7, "brightness": 150, "entropy": 4.5, "local_quality": 0.8}]
        
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
        
        # Mock upscaling
        with patch('pixelbliss.providers.upscale.upscale') as mock_upscale:
            mock_upscale.return_value = mock_upscaled_image
            
            mock_variants_wall.return_value = {"desktop": mock_upscaled_image}
            mock_alt.return_value = "alt text"
            mock_save_images.return_value = {"desktop": "/path/to/desktop.jpg", "base_img": "/path/to/base_img.png"}
            mock_iso.return_value = "2024-01-01T12:00:00"
            
            # Mock Twitter operations
            mock_upload.return_value = ["media_id_123"]
            mock_create_tweet.return_value = "tweet_id_456"
            
            result = await post_once(dry_run=False)  # Non-dry-run to test Twitter path
            
            assert result == 0
            mock_upscale.assert_called_once()
            mock_upload.assert_called_once()
            mock_create_tweet.assert_called_once()
            mock_update_tweet.assert_called_once()
            mock_send_success.assert_called_once()

    @patch('pixelbliss.run_once.config.load_config')
    @patch('pixelbliss.run_once.generate_theme_hint_async')
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
    @patch('pixelbliss.alerts.webhook.send_failure')
    @pytest.mark.asyncio
    async def test_post_once_all_candidates_duplicate(self, mock_send_failure, mock_phash, mock_duplicate, mock_hashes,
                                              mock_collage, mock_outdir, mock_slug, mock_today, mock_rescore,
                                              mock_score_parallel, mock_quality, mock_floors, mock_entropy, mock_brightness,
                                              mock_run_variants, mock_variants, mock_base, mock_theme, mock_config):
        """Test post_once when all candidates are duplicates."""
        # Setup mocks
        mock_cfg = Mock()
        mock_cfg.trending_themes.enabled = True
        mock_cfg.image_generation.model_fal = ["model1"]
        mock_cfg.image_generation.provider_order = ["fal", "replicate"]
        mock_cfg.image_generation.model_replicate = ["model2"]
        mock_cfg.image_generation.max_concurrency = None
        mock_cfg.prompt_generation.num_prompt_variants = 1
        mock_cfg.prompt_generation.async_enabled = False
        mock_cfg.prompt_generation.provider = "dummy"
        mock_cfg.discord.enabled = False
        mock_cfg.upscale.enabled = False
        mock_cfg.ranking.phash_distance_min = 6
        mock_config.return_value = mock_cfg
        
        # Mock successful pipeline
        mock_theme.return_value = "abstract"
        mock_base.return_value = "base prompt"
        mock_variants.return_value = ["variant1"]
        
        mock_image = Mock()
        mock_candidates = [{"image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1"}]
        mock_run_variants.return_value = mock_candidates
        
        # Mock scoring
        mock_brightness.return_value = 150
        mock_entropy.return_value = 4.5
        mock_floors.return_value = True
        mock_quality.return_value = (True, 0.8)
        mock_score_parallel.return_value = [{"image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1", "aesthetic": 0.7, "brightness": 150, "entropy": 4.5, "local_quality": 0.8}]
        
        # Mock rescoring
        mock_rescore.return_value = [{"final": 0.8, "image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1", "aesthetic": 0.7, "brightness": 150, "entropy": 4.5, "local_quality": 0.8}]
        
        # Mock file operations
        mock_today.return_value = "2024-01-01"
        mock_slug.return_value = "test_slug"
        mock_outdir.return_value = "/test/dir"
        mock_collage.return_value = "/test/collage.jpg"
        mock_hashes.return_value = ["existing_hash"]
        mock_duplicate.return_value = True  # All candidates are duplicates
        mock_phash.return_value = "abc123"
        
        result = await post_once(dry_run=True)
        
        assert result == 0  # Should return 0 (not fatal)
        mock_send_failure.assert_called_once_with("near-duplicate with manifest history", mock_cfg)

    @patch('pixelbliss.run_once.config.load_config')
    @patch('pixelbliss.run_once.generate_theme_hint_async')
    @patch('pixelbliss.run_once.prompts.make_base')
    @patch('pixelbliss.run_once.prompts.make_variants_from_base')
    @patch('pixelbliss.run_once.run_all_variants')
    @patch('pixelbliss.run_once.metrics.brightness')
    @patch('pixelbliss.run_once.metrics.entropy')
    @patch('pixelbliss.run_once.sanity.passes_floors')
    @patch('pixelbliss.alerts.webhook.send_failure')
    @pytest.mark.asyncio
    async def test_post_once_all_candidates_fail_sanity(self, mock_send_failure, mock_floors, mock_entropy, mock_brightness,
                                                mock_run_variants, mock_variants, mock_base, mock_theme, mock_config):
        """Test post_once when all candidates fail sanity checks."""
        # Setup mocks
        mock_cfg = Mock()
        mock_cfg.trending_themes.enabled = True
        mock_cfg.image_generation.model_fal = ["model1"]
        mock_cfg.image_generation.provider_order = ["fal", "replicate"]
        mock_cfg.image_generation.model_replicate = ["model2"]
        mock_cfg.image_generation.max_concurrency = None
        mock_cfg.prompt_generation.num_prompt_variants = 1
        mock_cfg.prompt_generation.async_enabled = False
        mock_cfg.prompt_generation.provider = "dummy"
        mock_cfg.discord.enabled = False
        mock_config.return_value = mock_cfg
        
        # Mock successful pipeline
        mock_theme.return_value = "abstract"
        mock_base.return_value = "base prompt"
        mock_variants.return_value = ["variant1"]
        
        mock_image = Mock()
        mock_candidates = [{"image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1"}]
        mock_run_variants.return_value = mock_candidates
        
        # Mock scoring - all fail sanity
        mock_brightness.return_value = 150
        mock_entropy.return_value = 4.5
        mock_floors.return_value = False  # Sanity check fails
        
        result = await post_once(dry_run=True)
        
        assert result == 1  # Should return error code
        mock_send_failure.assert_called_once_with("all candidates failed sanity/scoring", mock_cfg)

    @patch('pixelbliss.run_once.config.load_config')
    @patch('pixelbliss.run_once.generate_theme_hint_async')
    @patch('pixelbliss.run_once.prompts.make_base')
    @patch('pixelbliss.run_once.prompts.make_variants_from_base')
    @patch('pixelbliss.run_once.run_all_variants')
    @patch('pixelbliss.run_once.metrics.brightness')
    @patch('pixelbliss.run_once.metrics.entropy')
    @patch('pixelbliss.run_once.sanity.passes_floors')
    @patch('pixelbliss.run_once.quality.evaluate_local')
    @patch('pixelbliss.alerts.webhook.send_failure')
    @pytest.mark.asyncio
    async def test_post_once_all_candidates_fail_quality(self, mock_send_failure, mock_quality, mock_floors, mock_entropy, mock_brightness,
                                                 mock_run_variants, mock_variants, mock_base, mock_theme, mock_config):
        """Test post_once when all candidates fail quality checks."""
        # Setup mocks
        mock_cfg = Mock()
        mock_cfg.trending_themes.enabled = True
        mock_cfg.image_generation.model_fal = ["model1"]
        mock_cfg.image_generation.provider_order = ["fal", "replicate"]
        mock_cfg.image_generation.model_replicate = ["model2"]
        mock_cfg.image_generation.max_concurrency = None
        mock_cfg.prompt_generation.num_prompt_variants = 1
        mock_cfg.prompt_generation.async_enabled = False
        mock_cfg.prompt_generation.provider = "dummy"
        mock_cfg.discord.enabled = False
        mock_config.return_value = mock_cfg
        
        # Mock successful pipeline
        mock_theme.return_value = "abstract"
        mock_base.return_value = "base prompt"
        mock_variants.return_value = ["variant1"]
        
        mock_image = Mock()
        mock_candidates = [{"image": mock_image, "provider": "fal", "model": "test", "seed": 123, "prompt": "variant1"}]
        mock_run_variants.return_value = mock_candidates
        
        # Mock scoring - pass sanity but fail quality
        mock_brightness.return_value = 150
        mock_entropy.return_value = 4.5
        mock_floors.return_value = True
        mock_quality.return_value = (False, 0.3)  # Quality check fails
        
        result = await post_once(dry_run=True)
        
        assert result == 1  # Should return error code
        mock_send_failure.assert_called_once_with("all candidates failed sanity/scoring", mock_cfg)
