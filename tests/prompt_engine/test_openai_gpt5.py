import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pixelbliss.prompt_engine.openai_gpt5 import OpenAIGPT5Provider


class TestOpenAIGPT5Provider:
    @patch('pixelbliss.prompt_engine.openai_gpt5.AsyncOpenAI')
    @patch('pixelbliss.prompt_engine.openai_gpt5.OpenAI')
    def test_init(self, mock_openai, mock_async_openai):
        provider = OpenAIGPT5Provider()
        mock_openai.assert_called_once()
        mock_async_openai.assert_called_once()
        assert provider.model == "gpt-5"
        assert provider.client is not None
        assert provider.async_client is not None

    @patch('pixelbliss.prompt_engine.openai_gpt5.AsyncOpenAI')
    @patch('pixelbliss.prompt_engine.openai_gpt5.OpenAI')
    def test_make_base_with_knobs(self, mock_openai, mock_async_openai):
        """Test make_base_with_knobs method."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Generated base prompt with knobs"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIGPT5Provider()
        base_knobs = {
            "vibe": "serene",
            "palette": "monochrome powder blue",
            "style": "watercolor wash"
        }
        avoid_list = ["harsh clipping", "noise"]
        
        result = provider.make_base_with_knobs(base_knobs, avoid_list)
        assert result == "Generated base prompt with knobs"
        mock_client.chat.completions.create.assert_called_once()

    @patch('pixelbliss.prompt_engine.openai_gpt5.AsyncOpenAI')
    @patch('pixelbliss.prompt_engine.openai_gpt5.OpenAI')
    def test_make_variants_with_knobs(self, mock_openai, mock_async_openai):
        """Test make_variants_with_knobs method."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Variant with knobs"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIGPT5Provider()
        base_prompt = "A beautiful landscape"
        variant_knobs_list = [
            {"tone_curve": "high-key airy matte", "color_grade": "warm gold + muted", "surface_fx": "soft matte"},
            {"tone_curve": "low-key velvet deep", "color_grade": "cool silver + desaturated", "surface_fx": "pearl glow"}
        ]
        avoid_list = ["watermarks"]
        
        variants = provider.make_variants_with_knobs(base_prompt, 2, variant_knobs_list, avoid_list)
        assert len(variants) == 2
        assert all(v == "Variant with knobs" for v in variants)
        assert mock_client.chat.completions.create.call_count == 2

    @patch('pixelbliss.prompt_engine.openai_gpt5.AsyncOpenAI')
    @patch('pixelbliss.prompt_engine.openai_gpt5.OpenAI')
    def test_make_alt_text(self, mock_openai, mock_async_openai):
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Alt text description"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIGPT5Provider()
        result = provider.make_alt_text("base", "variant")
        assert result == "Alt text description"
        mock_client.chat.completions.create.assert_called_once()

    @patch('pixelbliss.prompt_engine.openai_gpt5.AsyncOpenAI')
    @patch('pixelbliss.prompt_engine.openai_gpt5.OpenAI')
    @pytest.mark.asyncio
    async def test_make_variants_with_knobs_async_success(self, mock_openai, mock_async_openai):
        """Test make_variants_with_knobs_async method."""
        mock_async_client = AsyncMock()
        mock_async_openai.return_value = mock_async_client
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Async variant with knobs"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_async_client.chat.completions.create.return_value = mock_response

        provider = OpenAIGPT5Provider()
        base_prompt = "A beautiful landscape"
        variant_knobs_list = [
            {"tone_curve": "high-key airy matte", "color_grade": "warm gold + muted", "surface_fx": "soft matte"},
            {"tone_curve": "low-key velvet deep", "color_grade": "cool silver + desaturated", "surface_fx": "pearl glow"}
        ]
        avoid_list = ["watermarks"]
        
        variants = await provider.make_variants_with_knobs_async(base_prompt, 2, variant_knobs_list, avoid_list, max_concurrency=1)
        assert len(variants) == 2
        assert all(v == "Async variant with knobs" for v in variants)

    @patch('pixelbliss.prompt_engine.openai_gpt5.AsyncOpenAI')
    @patch('pixelbliss.prompt_engine.openai_gpt5.OpenAI')
    @pytest.mark.asyncio
    async def test_make_variants_with_knobs_async_with_exceptions(self, mock_openai, mock_async_openai):
        """Test make_variants_with_knobs_async with exceptions."""
        mock_async_client = AsyncMock()
        mock_async_openai.return_value = mock_async_client
        
        # First call succeeds, second call fails
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Success variant"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        mock_async_client.chat.completions.create.side_effect = [mock_response, Exception("API Error")]

        provider = OpenAIGPT5Provider()
        base_prompt = "A beautiful landscape"
        variant_knobs_list = [
            {"tone_curve": "high-key airy matte", "color_grade": "warm gold + muted", "surface_fx": "soft matte"},
            {"tone_curve": "low-key velvet deep", "color_grade": "cool silver + desaturated", "surface_fx": "pearl glow"}
        ]
        
        variants = await provider.make_variants_with_knobs_async(base_prompt, 2, variant_knobs_list)
        assert len(variants) == 2
        assert variants[0] == "Success variant"
        assert "A beautiful landscape" in variants[1]  # Fallback variant contains base prompt
        assert "tone_curve: low-key velvet deep" in variants[1]  # Fallback contains knobs

    @patch('pixelbliss.prompt_engine.openai_gpt5.AsyncOpenAI')
    @patch('pixelbliss.prompt_engine.openai_gpt5.OpenAI')
    @pytest.mark.asyncio
    async def test_make_variants_with_knobs_async_with_progress_logger(self, mock_openai, mock_async_openai):
        """Test make_variants_with_knobs_async with progress logger."""
        mock_async_client = AsyncMock()
        mock_async_openai.return_value = mock_async_client
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Async variant with knobs"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_async_client.chat.completions.create.return_value = mock_response

        mock_progress_logger = Mock()
        provider = OpenAIGPT5Provider()
        base_prompt = "A beautiful landscape"
        variant_knobs_list = [
            {"tone_curve": "high-key airy matte", "color_grade": "warm gold + muted", "surface_fx": "soft matte"},
            {"tone_curve": "low-key velvet deep", "color_grade": "cool silver + desaturated", "surface_fx": "pearl glow"}
        ]
        
        variants = await provider.make_variants_with_knobs_async(base_prompt, 2, variant_knobs_list, progress_logger=mock_progress_logger)
        
        assert len(variants) == 2
        mock_progress_logger.start_operation.assert_called_once_with("prompt_generation", 2, "parallel prompt generation")
        assert mock_progress_logger.update_operation_progress.call_count == 2
        mock_progress_logger.finish_operation.assert_called_once_with("prompt_generation", True)

    @patch('pixelbliss.prompt_engine.openai_gpt5.AsyncOpenAI')
    @patch('pixelbliss.prompt_engine.openai_gpt5.OpenAI')
    @pytest.mark.asyncio
    async def test_make_variants_with_knobs_async_with_failures_and_logger(self, mock_openai, mock_async_openai):
        """Test make_variants_with_knobs_async with failures and logger."""
        mock_async_client = AsyncMock()
        mock_async_openai.return_value = mock_async_client
        
        # All calls fail to trigger the warning path
        mock_async_client.chat.completions.create.side_effect = Exception("API Error")

        mock_progress_logger = Mock()
        provider = OpenAIGPT5Provider()
        base_prompt = "A beautiful landscape"
        variant_knobs_list = [
            {"tone_curve": "high-key airy matte", "color_grade": "warm gold + muted", "surface_fx": "soft matte"},
            {"tone_curve": "low-key velvet deep", "color_grade": "cool silver + desaturated", "surface_fx": "pearl glow"}
        ]
        
        variants = await provider.make_variants_with_knobs_async(base_prompt, 2, variant_knobs_list, progress_logger=mock_progress_logger)
        
        assert len(variants) == 2
        # Both should be fallback variants
        assert all("A beautiful landscape" in v for v in variants)
        
        mock_progress_logger.start_operation.assert_called_once_with("prompt_generation", 2, "parallel prompt generation")
        assert mock_progress_logger.update_operation_progress.call_count == 2
        mock_progress_logger.finish_operation.assert_called_once_with("prompt_generation", False)
        mock_progress_logger.warning.assert_called_once_with("2 prompt variants used fallback generation")

    @patch('pixelbliss.prompt_engine.openai_gpt5.AsyncOpenAI')
    @patch('pixelbliss.prompt_engine.openai_gpt5.OpenAI')
    @pytest.mark.asyncio
    async def test_make_variants_with_knobs_async_zero_concurrency(self, mock_openai, mock_async_openai):
        """Test make_variants_with_knobs_async with zero concurrency."""
        mock_async_client = AsyncMock()
        mock_async_openai.return_value = mock_async_client
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Async variant with knobs"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_async_client.chat.completions.create.return_value = mock_response

        provider = OpenAIGPT5Provider()
        base_prompt = "A beautiful landscape"
        variant_knobs_list = [
            {"tone_curve": "high-key airy matte", "color_grade": "warm gold + muted", "surface_fx": "soft matte"},
            {"tone_curve": "low-key velvet deep", "color_grade": "cool silver + desaturated", "surface_fx": "pearl glow"}
        ]
        
        # Test with max_concurrency=0 to trigger the None semaphore path
        variants = await provider.make_variants_with_knobs_async(base_prompt, 2, variant_knobs_list, max_concurrency=0)
        assert len(variants) == 2
        assert all(v == "Async variant with knobs" for v in variants)

    @patch('pixelbliss.prompt_engine.openai_gpt5.AsyncOpenAI')
    @patch('pixelbliss.prompt_engine.openai_gpt5.OpenAI')
    def test_make_text_only_blurb_fallback(self, mock_openai, mock_async_openai):
        """Test _make_text_only_blurb fallback when API fails."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # First call fails, should return simple fallback
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        provider = OpenAIGPT5Provider()
        result = provider._make_text_only_blurb("nature")
        
        assert result == "In nature, we find beauty."
        mock_client.chat.completions.create.assert_called_once()
