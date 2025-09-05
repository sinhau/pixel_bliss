import pytest
from unittest.mock import Mock, patch
from pixelbliss.prompts import get_provider, make_base, make_variants_from_base, make_alt_text
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
    def test_make_base(self, mock_get_provider):
        mock_provider = Mock()
        mock_provider.make_base.return_value = "base prompt"
        mock_get_provider.return_value = mock_provider
        cfg = Mock()
        result = make_base("category", cfg)
        assert result == "base prompt"
        mock_provider.make_base.assert_called_once_with("category")

    @patch('pixelbliss.prompts.get_provider')
    def test_make_variants_from_base(self, mock_get_provider):
        mock_provider = Mock()
        mock_provider.make_variants_from_base.return_value = ["variant1", "variant2"]
        mock_get_provider.return_value = mock_provider
        cfg = Mock()
        cfg.art_styles = ["style1"]
        result = make_variants_from_base("base", 2, cfg)
        assert result == ["variant1", "variant2"]
        mock_provider.make_variants_from_base.assert_called_once_with("base", 2, ["style1"])

    @patch('pixelbliss.prompts.get_provider')
    def test_make_alt_text(self, mock_get_provider):
        mock_provider = Mock()
        mock_provider.make_alt_text.return_value = "alt text"
        mock_get_provider.return_value = mock_provider
        cfg = Mock()
        result = make_alt_text("base", "variant", cfg)
        assert result == "alt text"
        mock_provider.make_alt_text.assert_called_once_with("base", "variant")
