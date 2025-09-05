import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pixelbliss.prompts import get_provider, make_base, make_variants_from_base, make_alt_text, make_variants_from_base_async, make_variants_with_knobs
from pixelbliss.prompt_engine.openai_gpt5 import OpenAIGPT5Provider
from pixelbliss.prompt_engine.dummy_local import DummyLocalProvider


class TestPrompts:
    @patch('pixelbliss.prompts.OpenAIGPT5Provider')
    def test_get_provider_openai(self, mock_openai_provider):
        cfg = Mock()
        cfg.prompt_generation.provider = "openai"
        cfg.prompt_generation.model = "gpt-4"
        provider = get_provider(cfg)
        mock_openai_provider.assert_called_once_with(model="gpt-4")

    def test_get_provider_dummy(self):
        cfg = Mock()
        cfg.prompt_generation.provider = "dummy"
        provider = get_provider(cfg)
        assert isinstance(provider, DummyLocalProvider)

    def test_get_provider_unknown(self):
        cfg = Mock()
        cfg.prompt_generation.provider = "unknown"
        with pytest.raises(ValueError):
            get_provider(cfg)

    @patch('pixelbliss.prompts.get_provider')
    @patch('pixelbliss.prompts.KnobSelector.select_base_knobs')
    @patch('pixelbliss.prompts.KnobSelector.get_avoid_list')
    @patch('pixelbliss.prompts.time.time')
    def test_make_base_with_progress_logger(self, mock_time, mock_avoid_list, mock_base_knobs, mock_get_provider):
        # Setup mocks
        mock_time.side_effect = [0.0, 2.5]  # start and end times
        mock_provider = Mock()
        mock_provider.make_base_with_knobs.return_value = "base prompt"
        mock_get_provider.return_value = mock_provider
        mock_base_knobs.return_value = {"vibe": "dreamlike", "palette": "warm"}
        mock_avoid_list.return_value = ["harsh", "neon"]
        mock_progress_logger = Mock()
        
        cfg = Mock()
        cfg.prompt_generation.provider = "openai"
        cfg.prompt_generation.model = "gpt-5"
        
        result = make_base("sci-fi", cfg, mock_progress_logger)
        
        assert result == "base prompt"
        mock_provider.make_base_with_knobs.assert_called_once_with({"vibe": "dreamlike", "palette": "warm"}, ["harsh", "neon"])
        mock_progress_logger.log_base_prompt_generation.assert_called_once_with("knobs:sci-fi", "openai", "gpt-5")
        mock_progress_logger.log_base_prompt_success.assert_called_once_with("base prompt", 2.5)

    @patch('pixelbliss.prompts.get_provider')
    def test_make_alt_text(self, mock_get_provider):
        mock_provider = Mock()
        mock_provider.make_alt_text.return_value = "alt text"
        mock_get_provider.return_value = mock_provider
        cfg = Mock()
        result = make_alt_text("base", "variant", cfg)
        assert result == "alt text"
        mock_provider.make_alt_text.assert_called_once_with("base", "variant")

    @patch('pixelbliss.prompts.get_provider')
    @patch('pixelbliss.prompts.KnobSelector.select_base_knobs')
    @patch('pixelbliss.prompts.KnobSelector.get_avoid_list')
    @patch('pixelbliss.prompts.time.time')
    def test_make_base_without_progress_logger(self, mock_time, mock_avoid_list, mock_base_knobs, mock_get_provider):
        # Setup mocks
        mock_time.side_effect = [0.0, 1.5]  # start and end times
        mock_provider = Mock()
        mock_provider.make_base_with_knobs.return_value = "base prompt"
        mock_get_provider.return_value = mock_provider
        mock_base_knobs.return_value = {"vibe": "dreamlike", "palette": "warm"}
        mock_avoid_list.return_value = ["harsh", "neon"]
        
        cfg = Mock()
        cfg.prompt_generation.provider = "dummy"
        cfg.prompt_generation.model = "local"
        
        with patch('pixelbliss.prompts.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            result = make_base("nature", cfg)
            
            assert result == "base prompt"
            mock_provider.make_base_with_knobs.assert_called_once_with({"vibe": "dreamlike", "palette": "warm"}, ["harsh", "neon"])
            # Now we expect 2 info calls: one for knobs selection, one for success
            assert mock_logger.info.call_count == 2
            mock_logger.debug.assert_called_once()

    @patch('pixelbliss.prompts.get_provider')
    @patch('pixelbliss.prompts.KnobSelector.select_base_knobs')
    @patch('pixelbliss.prompts.KnobSelector.get_avoid_list')
    @patch('pixelbliss.prompts.time.time')
    def test_make_base_error_handling(self, mock_time, mock_avoid_list, mock_base_knobs, mock_get_provider):
        # Setup mocks
        mock_time.side_effect = [0.0, 1.0]  # start and end times
        mock_provider = Mock()
        mock_provider.make_base_with_knobs.side_effect = Exception("API Error")
        mock_get_provider.return_value = mock_provider
        mock_base_knobs.return_value = {"vibe": "dreamlike"}
        mock_avoid_list.return_value = ["harsh"]
        
        cfg = Mock()
        
        with patch('pixelbliss.prompts.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with pytest.raises(Exception, match="API Error"):
                make_base("category", cfg)
            
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args[0][0]
            assert "Knobs-based base prompt generation failed after 1.00s: API Error" in call_args

    @patch('pixelbliss.prompts.get_provider')
    @patch('pixelbliss.prompts.KnobSelector.select_single_variant_knob')
    @patch('pixelbliss.prompts.KnobSelector.get_avoid_list')
    @patch('pixelbliss.prompts.time.time')
    def test_make_variants_with_knobs_without_progress_logger(self, mock_time, mock_avoid_list, mock_variant_knobs, mock_get_provider):
        # Setup mocks
        mock_time.side_effect = [0.0, 2.0]  # start and end times
        mock_provider = Mock()
        mock_provider.make_variants_with_knobs.return_value = ["variant1", "variant2"]
        mock_get_provider.return_value = mock_provider
        mock_variant_knobs.side_effect = [
            {"tone_curve": "high-key", "color_grade": "warm"},
            {"tone_curve": "low-key", "color_grade": "cool"}
        ]
        mock_avoid_list.return_value = ["harsh"]
        
        cfg = Mock()
        cfg.prompt_generation.provider = "dummy"
        cfg.prompt_generation.model = "local"
        cfg.prompt_generation.variant_strategy = "single"
        
        with patch('pixelbliss.prompts.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            result = make_variants_with_knobs("base prompt", 2, cfg)
            
            assert result == ["variant1", "variant2"]
            mock_provider.make_variants_with_knobs.assert_called_once()
            # Now we expect 4 info calls: strategy header + 2 variants + success
            assert mock_logger.info.call_count == 4
            mock_logger.debug.assert_called()  # Should be called for each variant
            assert mock_logger.debug.call_count == 2

    @patch('pixelbliss.prompts.get_provider')
    @patch('pixelbliss.prompts.KnobSelector.select_variant_knobs')
    @patch('pixelbliss.prompts.KnobSelector.get_avoid_list')
    @patch('pixelbliss.prompts.time.time')
    def test_make_variants_with_knobs_multiple_strategy(self, mock_time, mock_avoid_list, mock_variant_knobs, mock_get_provider):
        # Setup mocks
        mock_time.side_effect = [0.0, 2.0]  # start and end times
        mock_provider = Mock()
        mock_provider.make_variants_with_knobs.return_value = ["variant1", "variant2"]
        mock_get_provider.return_value = mock_provider
        mock_variant_knobs.side_effect = [
            {"tone_curve": "high-key", "color_grade": "warm", "surface_fx": "matte"},
            {"tone_curve": "low-key", "color_grade": "cool", "surface_fx": "glossy"}
        ]
        mock_avoid_list.return_value = ["harsh"]
        
        cfg = Mock()
        cfg.prompt_generation.provider = "dummy"
        cfg.prompt_generation.model = "local"
        cfg.prompt_generation.variant_strategy = "multiple"  # Test multiple strategy
        
        with patch('pixelbliss.prompts.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            result = make_variants_with_knobs("base prompt", 2, cfg)
            
            assert result == ["variant1", "variant2"]
            mock_provider.make_variants_with_knobs.assert_called_once()
            # Now we expect 4 info calls: strategy header + 2 variants + success
            assert mock_logger.info.call_count == 4

    @patch('pixelbliss.prompts.get_provider')
    @patch('pixelbliss.prompts.KnobSelector.select_single_variant_knob')
    @patch('pixelbliss.prompts.KnobSelector.get_avoid_list')
    @patch('pixelbliss.prompts.time.time')
    def test_make_variants_with_knobs_error_handling(self, mock_time, mock_avoid_list, mock_variant_knobs, mock_get_provider):
        # Setup mocks
        mock_time.side_effect = [0.0, 1.5]  # start and end times
        mock_provider = Mock()
        mock_provider.make_variants_with_knobs.side_effect = Exception("Rate limit exceeded")
        mock_get_provider.return_value = mock_provider
        mock_variant_knobs.return_value = {"tone_curve": "high-key"}
        mock_avoid_list.return_value = ["harsh"]
        
        cfg = Mock()
        cfg.prompt_generation.provider = "dummy"
        cfg.prompt_generation.variant_strategy = "single"
        
        with patch('pixelbliss.prompts.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with pytest.raises(Exception, match="Rate limit exceeded"):
                make_variants_with_knobs("base prompt", 1, cfg)
            
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args[0][0]
            assert "Knobs-based variant prompt generation failed after 1.50s: Rate limit exceeded" in call_args

    @pytest.mark.asyncio
    @patch('pixelbliss.prompts.get_provider')
    @patch('pixelbliss.prompts.KnobSelector.select_single_variant_knob')
    @patch('pixelbliss.prompts.KnobSelector.get_avoid_list')
    @patch('pixelbliss.prompts.time.time')
    async def test_make_variants_from_base_async_with_async_provider(self, mock_time, mock_avoid_list, mock_variant_knobs, mock_get_provider):
        # Setup mocks
        mock_time.side_effect = [0.0, 4.1]  # start and end times
        mock_provider = Mock()
        mock_provider.make_variants_with_knobs_async = AsyncMock(return_value=["async_variant1", "async_variant2"])
        mock_get_provider.return_value = mock_provider
        mock_variant_knobs.side_effect = [
            {"tone_curve": "high-key"},
            {"tone_curve": "low-key"}
        ]
        mock_avoid_list.return_value = ["harsh"]
        
        cfg = Mock()
        cfg.prompt_generation.provider = "openai"
        cfg.prompt_generation.model = "gpt-5"
        cfg.prompt_generation.max_concurrency = 5
        cfg.prompt_generation.variant_strategy = "single"
        
        with patch('pixelbliss.prompts.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            result = await make_variants_from_base_async("base prompt", 2, cfg)
            
            assert result == ["async_variant1", "async_variant2"]
            mock_provider.make_variants_with_knobs_async.assert_called_once()
            # Now we expect 4 info calls: strategy header + 2 variants + success
            assert mock_logger.info.call_count == 4

    @pytest.mark.asyncio
    @patch('pixelbliss.prompts.get_provider')
    @patch('pixelbliss.prompts.KnobSelector.select_single_variant_knob')
    @patch('pixelbliss.prompts.KnobSelector.get_avoid_list')
    @patch('pixelbliss.prompts.time.time')
    @patch('pixelbliss.prompts.asyncio.to_thread')
    async def test_make_variants_from_base_async_fallback_to_sync(self, mock_to_thread, mock_time, mock_avoid_list, mock_variant_knobs, mock_get_provider):
        # Setup mocks
        mock_time.side_effect = [0.0, 2.8]  # start and end times
        mock_provider = Mock()
        # Provider doesn't have async method, so it will fallback to sync
        # Explicitly delete the async method to ensure hasattr returns False
        if hasattr(mock_provider, 'make_variants_with_knobs_async'):
            delattr(mock_provider, 'make_variants_with_knobs_async')
        mock_provider.make_variants_with_knobs.return_value = ["sync_variant1", "sync_variant2"]
        mock_get_provider.return_value = mock_provider
        mock_variant_knobs.side_effect = [
            {"tone_curve": "high-key"},
            {"tone_curve": "low-key"}
        ]
        mock_avoid_list.return_value = ["harsh"]
        mock_to_thread.return_value = ["sync_variant1", "sync_variant2"]
        
        cfg = Mock()
        cfg.prompt_generation.provider = "dummy"
        cfg.prompt_generation.model = "local"
        cfg.prompt_generation.variant_strategy = "single"
        
        with patch('pixelbliss.prompts.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            result = await make_variants_from_base_async("base prompt", 2, cfg)
            
            assert result == ["sync_variant1", "sync_variant2"]
            mock_to_thread.assert_called_once()
            # Now we expect 4 info calls: strategy header + 2 variants + success
            assert mock_logger.info.call_count == 4

    @pytest.mark.asyncio
    @patch('pixelbliss.prompts.get_provider')
    @patch('pixelbliss.prompts.KnobSelector.select_single_variant_knob')
    @patch('pixelbliss.prompts.KnobSelector.get_avoid_list')
    @patch('pixelbliss.prompts.time.time')
    async def test_make_variants_from_base_async_error_handling(self, mock_time, mock_avoid_list, mock_variant_knobs, mock_get_provider):
        # Setup mocks
        mock_time.side_effect = [0.0, 1.5]  # start and end times
        mock_provider = Mock()
        mock_provider.make_variants_with_knobs_async = AsyncMock(side_effect=Exception("Connection timeout"))
        mock_get_provider.return_value = mock_provider
        mock_variant_knobs.return_value = {"tone_curve": "high-key"}
        mock_avoid_list.return_value = ["harsh"]
        
        cfg = Mock()
        cfg.prompt_generation.provider = "openai"
        cfg.prompt_generation.model = "gpt-5"
        cfg.prompt_generation.max_concurrency = 3
        cfg.prompt_generation.variant_strategy = "single"
        
        with patch('pixelbliss.prompts.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with pytest.raises(Exception, match="Connection timeout"):
                await make_variants_from_base_async("base prompt", 1, cfg)
            
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args[0][0]
            assert "Async knobs-based variant prompt generation failed after 1.50s: Connection timeout" in call_args
