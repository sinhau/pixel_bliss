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
    def test_make_base(self, mock_openai, mock_async_openai):
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Generated base prompt"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIGPT5Provider()
        result = provider.make_base("sunset")
        assert result == "Generated base prompt"
        mock_client.chat.completions.create.assert_called_once()

    @patch('pixelbliss.prompt_engine.openai_gpt5.AsyncOpenAI')
    @patch('pixelbliss.prompt_engine.openai_gpt5.OpenAI')
    def test_make_variants_from_base(self, mock_openai, mock_async_openai):
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Variant prompt"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIGPT5Provider()
        variants = provider.make_variants_from_base("base prompt", 2, ["style1", "style2"])
        assert len(variants) == 2
        assert all(v == "Variant prompt" for v in variants)
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
    async def test_generate_single_variant_async_with_semaphore(self, mock_openai, mock_async_openai):
        mock_async_client = AsyncMock()
        mock_async_openai.return_value = mock_async_client
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Async variant prompt"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_async_client.chat.completions.create.return_value = mock_response

        provider = OpenAIGPT5Provider()
        semaphore = asyncio.Semaphore(1)
        result = await provider._generate_single_variant_async("base prompt", "style1", semaphore)
        assert result == "Async variant prompt"
        mock_async_client.chat.completions.create.assert_called_once()

    @patch('pixelbliss.prompt_engine.openai_gpt5.AsyncOpenAI')
    @patch('pixelbliss.prompt_engine.openai_gpt5.OpenAI')
    @pytest.mark.asyncio
    async def test_generate_single_variant_async_without_semaphore(self, mock_openai, mock_async_openai):
        mock_async_client = AsyncMock()
        mock_async_openai.return_value = mock_async_client
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Async variant prompt"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_async_client.chat.completions.create.return_value = mock_response

        provider = OpenAIGPT5Provider()
        result = await provider._generate_single_variant_async("base prompt", "style1", None)
        assert result == "Async variant prompt"
        mock_async_client.chat.completions.create.assert_called_once()

    @patch('pixelbliss.prompt_engine.openai_gpt5.AsyncOpenAI')
    @patch('pixelbliss.prompt_engine.openai_gpt5.OpenAI')
    @pytest.mark.asyncio
    async def test_make_variants_from_base_async_success(self, mock_openai, mock_async_openai):
        mock_async_client = AsyncMock()
        mock_async_openai.return_value = mock_async_client
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Async variant prompt"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_async_client.chat.completions.create.return_value = mock_response

        provider = OpenAIGPT5Provider()
        variants = await provider.make_variants_from_base_async("base prompt", 2, ["style1", "style2"], max_concurrency=1)
        assert len(variants) == 2
        assert all(v == "Async variant prompt" for v in variants)

    @patch('pixelbliss.prompt_engine.openai_gpt5.AsyncOpenAI')
    @patch('pixelbliss.prompt_engine.openai_gpt5.OpenAI')
    @pytest.mark.asyncio
    async def test_make_variants_from_base_async_with_exceptions(self, mock_openai, mock_async_openai):
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
        variants = await provider.make_variants_from_base_async("base prompt", 2, ["style1", "style2"])
        assert len(variants) == 2
        assert variants[0] == "Success variant"
        assert "base prompt" in variants[1]  # Fallback variant
        assert "style" in variants[1]  # Fallback variant contains a style

    @patch('pixelbliss.prompt_engine.openai_gpt5.AsyncOpenAI')
    @patch('pixelbliss.prompt_engine.openai_gpt5.OpenAI')
    @pytest.mark.asyncio
    async def test_make_variants_from_base_async_with_progress_logger(self, mock_openai, mock_async_openai):
        mock_async_client = AsyncMock()
        mock_async_openai.return_value = mock_async_client
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Async variant prompt"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_async_client.chat.completions.create.return_value = mock_response

        mock_progress_logger = Mock()
        provider = OpenAIGPT5Provider()
        variants = await provider.make_variants_from_base_async("base prompt", 2, ["style1", "style2"], progress_logger=mock_progress_logger)
        
        assert len(variants) == 2
        mock_progress_logger.start_operation.assert_called_once_with("prompt_generation", 2, "parallel prompt generation")
        assert mock_progress_logger.update_operation_progress.call_count == 2
        mock_progress_logger.finish_operation.assert_called_once_with("prompt_generation", True)

    @patch('pixelbliss.prompt_engine.openai_gpt5.AsyncOpenAI')
    @patch('pixelbliss.prompt_engine.openai_gpt5.OpenAI')
    @pytest.mark.asyncio
    async def test_make_variants_from_base_async_with_failures_and_logger(self, mock_openai, mock_async_openai):
        mock_async_client = AsyncMock()
        mock_async_openai.return_value = mock_async_client
        
        # All calls fail to trigger the warning path
        mock_async_client.chat.completions.create.side_effect = Exception("API Error")

        mock_progress_logger = Mock()
        provider = OpenAIGPT5Provider()
        variants = await provider.make_variants_from_base_async("base prompt", 2, ["style1", "style2"], progress_logger=mock_progress_logger)
        
        assert len(variants) == 2
        # Both should be fallback variants
        assert all("base prompt" in v and "style" in v for v in variants)
        
        mock_progress_logger.start_operation.assert_called_once_with("prompt_generation", 2, "parallel prompt generation")
        assert mock_progress_logger.update_operation_progress.call_count == 2
        mock_progress_logger.finish_operation.assert_called_once_with("prompt_generation", False)
        mock_progress_logger.warning.assert_called_once_with("2 prompt variants used fallback generation")

    @patch('pixelbliss.prompt_engine.openai_gpt5.AsyncOpenAI')
    @patch('pixelbliss.prompt_engine.openai_gpt5.OpenAI')
    @pytest.mark.asyncio
    async def test_make_variants_from_base_async_zero_concurrency(self, mock_openai, mock_async_openai):
        mock_async_client = AsyncMock()
        mock_async_openai.return_value = mock_async_client
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Async variant prompt"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_async_client.chat.completions.create.return_value = mock_response

        provider = OpenAIGPT5Provider()
        # Test with max_concurrency=0 to trigger the None semaphore path
        variants = await provider.make_variants_from_base_async("base prompt", 2, ["style1", "style2"], max_concurrency=0)
        assert len(variants) == 2
        assert all(v == "Async variant prompt" for v in variants)
