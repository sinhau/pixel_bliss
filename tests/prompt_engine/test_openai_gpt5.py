import pytest
from unittest.mock import Mock, patch
from pixelbliss.prompt_engine.openai_gpt5 import OpenAIGPT5Provider


class TestOpenAIGPT5Provider:
    @patch('pixelbliss.prompt_engine.openai_gpt5.OpenAI')
    def test_init(self, mock_openai):
        provider = OpenAIGPT5Provider()
        mock_openai.assert_called_once()
        assert provider.model == "gpt-5"
        assert provider.client is not None

    @patch('pixelbliss.prompt_engine.openai_gpt5.OpenAI')
    def test_make_base(self, mock_openai):
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

    @patch('pixelbliss.prompt_engine.openai_gpt5.OpenAI')
    def test_make_variants_from_base(self, mock_openai):
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

    @patch('pixelbliss.prompt_engine.openai_gpt5.OpenAI')
    def test_make_alt_text(self, mock_openai):
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
