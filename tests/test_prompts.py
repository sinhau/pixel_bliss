import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pixelbliss.prompts import get_provider, make_base, make_variants_from_base, make_alt_text, make_variants_from_base_async
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
    @patch('pixelbliss.prompts.time.time')
    def test_make_base_with_progress_logger(self, mock_time, mock_get_provider):
        # Setup mocks
        mock_time.side_effect = [0.0, 2.5]  # start and end times
        mock_provider = Mock()
        mock_provider.make_base.return_value = "base prompt"
        mock_get_provider.return_value = mock_provider
        mock_progress_logger = Mock()
        
        cfg = Mock()
        cfg.prompt_generation.provider = "openai"
        cfg.prompt_generation.model = "gpt-5"
        
        result = make_base("sci-fi", cfg, mock_progress_logger)
        
        assert result == "base prompt"
        mock_provider.make_base.assert_called_once_with("sci-fi")
        mock_progress_logger.log_base_prompt_generation.assert_called_once_with("sci-fi", "openai", "gpt-5")
        mock_progress_logger.log_base_prompt_success.assert_called_once_with("base prompt", 2.5)

    @patch('pixelbliss.prompts.get_provider')
    @patch('pixelbliss.prompts.time.time')
    def test_make_base_without_progress_logger(self, mock_time, mock_get_provider):
        # Setup mocks
        mock_time.side_effect = [0.0, 1.5]  # start and end times
        mock_provider = Mock()
        mock_provider.make_base.return_value = "base prompt"
        mock_get_provider.return_value = mock_provider
        
        cfg = Mock()
        cfg.prompt_generation.provider = "dummy"
        cfg.prompt_generation.model = "local"
        
        with patch('pixelbliss.prompts.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            result = make_base("nature", cfg)
            
            assert result == "base prompt"
            mock_provider.make_base.assert_called_once_with("nature")
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            assert "Base prompt generated for 'nature' in 1.50s" in call_args

    @patch('pixelbliss.prompts.get_provider')
    @patch('pixelbliss.prompts.time.time')
    def test_make_base_error_handling(self, mock_time, mock_get_provider):
        # Setup mocks
        mock_time.side_effect = [0.0, 1.0]  # start and end times
        mock_provider = Mock()
        mock_provider.make_base.side_effect = Exception("API Error")
        mock_get_provider.return_value = mock_provider
        
        cfg = Mock()
        
        with patch('pixelbliss.prompts.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with pytest.raises(Exception, match="API Error"):
                make_base("category", cfg)
            
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args[0][0]
            assert "Base prompt generation failed for 'category' after 1.00s" in call_args

    @patch('pixelbliss.prompts.get_provider')
    @patch('pixelbliss.prompts.time.time')
    def test_make_variants_from_base_with_progress_logger(self, mock_time, mock_get_provider):
        # Setup mocks
        mock_time.side_effect = [0.0, 3.2]  # start and end times
        mock_provider = Mock()
        mock_provider.make_variants_from_base.return_value = ["variant1", "variant2"]
        mock_get_provider.return_value = mock_provider
        mock_progress_logger = Mock()
        
        cfg = Mock()
        cfg.prompt_generation.provider = "openai"
        cfg.prompt_generation.model = "gpt-5"
        cfg.art_styles = ["style1", "style2"]
        
        result = make_variants_from_base("base prompt", 2, cfg, mock_progress_logger)
        
        assert result == ["variant1", "variant2"]
        mock_provider.make_variants_from_base.assert_called_once_with("base prompt", 2, ["style1", "style2"])
        mock_progress_logger.log_variant_prompt_generation_start.assert_called_once_with(2, "openai", "gpt-5", False)
        mock_progress_logger.log_variant_prompt_success.assert_called_once_with(["variant1", "variant2"], 3.2)

    @patch('pixelbliss.prompts.get_provider')
    @patch('pixelbliss.prompts.time.time')
    def test_make_variants_from_base_error_handling(self, mock_time, mock_get_provider):
        # Setup mocks
        mock_time.side_effect = [0.0, 2.0]  # start and end times
        mock_provider = Mock()
        mock_provider.make_variants_from_base.side_effect = Exception("Rate limit exceeded")
        mock_get_provider.return_value = mock_provider
        mock_progress_logger = Mock()
        
        cfg = Mock()
        cfg.prompt_generation.provider = "openai"
        cfg.prompt_generation.model = "gpt-5"
        cfg.art_styles = ["style1"]
        
        with pytest.raises(Exception, match="Rate limit exceeded"):
            make_variants_from_base("base prompt", 2, cfg, mock_progress_logger)
        
        mock_progress_logger.log_variant_prompt_error.assert_called_once_with("Rate limit exceeded", 2.0)

    @pytest.mark.asyncio
    @patch('pixelbliss.prompts.get_provider')
    @patch('pixelbliss.prompts.time.time')
    async def test_make_variants_from_base_async_with_async_provider(self, mock_time, mock_get_provider):
        # Setup mocks
        mock_time.side_effect = [0.0, 4.1]  # start and end times
        mock_provider = Mock()
        mock_provider.make_variants_from_base_async = AsyncMock(return_value=["async_variant1", "async_variant2"])
        mock_get_provider.return_value = mock_provider
        mock_progress_logger = Mock()
        
        cfg = Mock()
        cfg.prompt_generation.provider = "openai"
        cfg.prompt_generation.model = "gpt-5"
        cfg.art_styles = ["style1", "style2"]
        cfg.prompt_generation.max_concurrency = 5
        
        result = await make_variants_from_base_async("base prompt", 2, cfg, mock_progress_logger)
        
        assert result == ["async_variant1", "async_variant2"]
        mock_provider.make_variants_from_base_async.assert_called_once_with(
            "base prompt", 2, ["style1", "style2"], 5, mock_progress_logger
        )
        mock_progress_logger.log_variant_prompt_generation_start.assert_called_once_with(2, "openai", "gpt-5", True)
        mock_progress_logger.log_variant_prompt_success.assert_called_once_with(["async_variant1", "async_variant2"], 4.1)

    @pytest.mark.asyncio
    @patch('pixelbliss.prompts.get_provider')
    @patch('pixelbliss.prompts.time.time')
    @patch('pixelbliss.prompts.asyncio.to_thread')
    async def test_make_variants_from_base_async_fallback_to_sync(self, mock_to_thread, mock_time, mock_get_provider):
        # Setup mocks
        mock_time.side_effect = [0.0, 2.8]  # start and end times
        mock_provider = Mock()
        # Provider doesn't have async method, so it will fallback to sync
        # Remove the async method attribute to trigger fallback
        del mock_provider.make_variants_from_base_async
        mock_provider.make_variants_from_base.return_value = ["sync_variant1", "sync_variant2"]
        mock_get_provider.return_value = mock_provider
        mock_progress_logger = Mock()
        mock_to_thread.return_value = ["sync_variant1", "sync_variant2"]
        
        cfg = Mock()
        cfg.prompt_generation.provider = "dummy"
        cfg.prompt_generation.model = "local"
        cfg.art_styles = ["style1"]
        
        result = await make_variants_from_base_async("base prompt", 2, cfg, mock_progress_logger)
        
        assert result == ["sync_variant1", "sync_variant2"]
        mock_progress_logger.start_operation.assert_called_once_with("prompt_generation", 2, "sequential prompt generation")
        mock_progress_logger.update_operation_progress.assert_called_once_with("prompt_generation", completed=2)
        mock_progress_logger.finish_operation.assert_called_once_with("prompt_generation", success=True)
        mock_progress_logger.log_variant_prompt_success.assert_called_once_with(["sync_variant1", "sync_variant2"], 2.8)

    @pytest.mark.asyncio
    @patch('pixelbliss.prompts.get_provider')
    @patch('pixelbliss.prompts.time.time')
    async def test_make_variants_from_base_async_error_handling(self, mock_time, mock_get_provider):
        # Setup mocks
        mock_time.side_effect = [0.0, 1.5]  # start and end times
        mock_provider = Mock()
        mock_provider.make_variants_from_base_async = AsyncMock(side_effect=Exception("Connection timeout"))
        mock_get_provider.return_value = mock_provider
        mock_progress_logger = Mock()
        
        cfg = Mock()
        cfg.prompt_generation.provider = "openai"
        cfg.prompt_generation.model = "gpt-5"
        cfg.art_styles = ["style1"]
        cfg.prompt_generation.max_concurrency = 3
        
        with pytest.raises(Exception, match="Connection timeout"):
            await make_variants_from_base_async("base prompt", 2, cfg, mock_progress_logger)
        
        mock_progress_logger.log_variant_prompt_error.assert_called_once_with("Connection timeout", 1.5)

    @patch('pixelbliss.prompts.get_provider')
    def test_make_alt_text(self, mock_get_provider):
        mock_provider = Mock()
        mock_provider.make_alt_text.return_value = "alt text"
        mock_get_provider.return_value = mock_provider
        cfg = Mock()
        result = make_alt_text("base", "variant", cfg)
        assert result == "alt text"
        mock_provider.make_alt_text.assert_called_once_with("base", "variant")

    # Legacy tests for backward compatibility
    @patch('pixelbliss.prompts.get_provider')
    def test_make_base_legacy(self, mock_get_provider):
        mock_provider = Mock()
        mock_provider.make_base.return_value = "base prompt"
        mock_get_provider.return_value = mock_provider
        cfg = Mock()
        result = make_base("category", cfg)
        assert result == "base prompt"
        mock_provider.make_base.assert_called_once_with("category")

    @patch('pixelbliss.prompts.get_provider')
    def test_make_variants_from_base_legacy(self, mock_get_provider):
        mock_provider = Mock()
        mock_provider.make_variants_from_base.return_value = ["variant1", "variant2"]
        mock_get_provider.return_value = mock_provider
        cfg = Mock()
        cfg.art_styles = ["style1"]
        result = make_variants_from_base("base", 2, cfg)
        assert result == ["variant1", "variant2"]
        mock_provider.make_variants_from_base.assert_called_once_with("base", 2, ["style1"])

    @patch('pixelbliss.prompts.get_provider')
    @patch('pixelbliss.prompts.time.time')
    def test_make_variants_from_base_without_progress_logger(self, mock_time, mock_get_provider):
        # Setup mocks
        mock_time.side_effect = [0.0, 2.0]  # start and end times
        mock_provider = Mock()
        mock_provider.make_variants_from_base.return_value = ["variant1", "variant2"]
        mock_get_provider.return_value = mock_provider
        
        cfg = Mock()
        cfg.prompt_generation.provider = "dummy"
        cfg.prompt_generation.model = "local"
        cfg.art_styles = ["style1", "style2"]
        
        with patch('pixelbliss.prompts.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            result = make_variants_from_base("base prompt", 2, cfg)
            
            assert result == ["variant1", "variant2"]
            mock_provider.make_variants_from_base.assert_called_once_with("base prompt", 2, ["style1", "style2"])
            mock_logger.info.assert_called_once()
            mock_logger.debug.assert_called()  # Should be called for each variant
            assert mock_logger.debug.call_count == 2

    @patch('pixelbliss.prompts.get_provider')
    @patch('pixelbliss.prompts.time.time')
    def test_make_variants_from_base_error_without_progress_logger(self, mock_time, mock_get_provider):
        # Setup mocks
        mock_time.side_effect = [0.0, 1.5]  # start and end times
        mock_provider = Mock()
        mock_provider.make_variants_from_base.side_effect = Exception("API Error")
        mock_get_provider.return_value = mock_provider
        
        cfg = Mock()
        cfg.prompt_generation.provider = "dummy"
        cfg.prompt_generation.model = "local"
        cfg.art_styles = ["style1", "style2"]
        
        with patch('pixelbliss.prompts.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with pytest.raises(Exception, match="API Error"):
                make_variants_from_base("base prompt", 2, cfg)  # No progress logger
            
            mock_provider.make_variants_from_base.assert_called_once_with("base prompt", 2, ["style1", "style2"])
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args[0][0]
            assert "Variant prompt generation failed after 1.50s: API Error" in call_args

    @pytest.mark.asyncio
    @patch('pixelbliss.prompts.get_provider')
    @patch('pixelbliss.prompts.time.time')
    async def test_make_variants_from_base_async_error_without_progress_logger(self, mock_time, mock_get_provider):
        # Setup mocks
        mock_time.side_effect = [0.0, 2.5]  # start and end times
        mock_provider = Mock()
        mock_provider.make_variants_from_base_async = AsyncMock(side_effect=Exception("Connection timeout"))
        mock_get_provider.return_value = mock_provider
        
        cfg = Mock()
        cfg.prompt_generation.provider = "openai"
        cfg.prompt_generation.model = "gpt-5"
        cfg.art_styles = ["style1", "style2"]
        cfg.prompt_generation.max_concurrency = 5
        
        with patch('pixelbliss.prompts.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with pytest.raises(Exception, match="Connection timeout"):
                await make_variants_from_base_async("base prompt", 2, cfg)  # No progress logger
            
            mock_provider.make_variants_from_base_async.assert_called_once_with(
                "base prompt", 2, ["style1", "style2"], 5, None
            )
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args[0][0]
            assert "Async variant prompt generation failed after 2.50s: Connection timeout" in call_args

    @pytest.mark.asyncio
    @patch('pixelbliss.prompts.get_provider')
    @patch('pixelbliss.prompts.time.time')
    async def test_make_variants_from_base_async_without_progress_logger(self, mock_time, mock_get_provider):
        # Setup mocks
        mock_time.side_effect = [0.0, 3.0]  # start and end times
        mock_provider = Mock()
        mock_provider.make_variants_from_base_async = AsyncMock(return_value=["async_variant1", "async_variant2"])
        mock_get_provider.return_value = mock_provider
        
        cfg = Mock()
        cfg.prompt_generation.provider = "openai"
        cfg.prompt_generation.model = "gpt-5"
        cfg.art_styles = ["style1", "style2"]
        cfg.prompt_generation.max_concurrency = 5
        
        with patch('pixelbliss.prompts.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            result = await make_variants_from_base_async("base prompt", 2, cfg)
            
            assert result == ["async_variant1", "async_variant2"]
            mock_provider.make_variants_from_base_async.assert_called_once_with(
                "base prompt", 2, ["style1", "style2"], 5, None
            )
            mock_logger.info.assert_called_once()
            mock_logger.debug.assert_called()  # Should be called for each variant
            assert mock_logger.debug.call_count == 2
