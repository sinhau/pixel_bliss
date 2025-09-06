import pytest
from unittest.mock import Mock, patch
from pixelbliss import prompts
from pixelbliss.config import Config
from pixelbliss.prompt_engine.dummy_local import DummyLocalProvider
from pixelbliss.prompt_engine.openai_gpt5 import OpenAIGPT5Provider


class TestTwitterBlurb:
    """Test suite for Twitter blurb generation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.theme = "nature"
        self.base_prompt = "A beautiful landscape with mountains and trees"
        self.variant_prompt = "A serene mountain landscape with autumn trees and golden light"
        
        # Create mock config
        self.mock_config = Mock(spec=Config)
        self.mock_config.prompt_generation = Mock()
        self.mock_config.prompt_generation.provider = "dummy"
        self.mock_config.prompt_generation.model = "test-model"

    def test_make_twitter_blurb_with_dummy_provider(self):
        """Test Twitter blurb generation with dummy provider."""
        blurb = prompts.make_twitter_blurb(
            self.theme, 
            "/fake/image/path.jpg", 
            self.mock_config
        )
        
        assert blurb is not None
        assert isinstance(blurb, str)
        assert len(blurb) > 0
        assert len(blurb) <= 280  # Twitter character limit
        assert self.theme in blurb  # Should contain the theme

    def test_make_twitter_blurb_character_limit(self):
        """Test that Twitter blurb respects character limits."""
        blurb = prompts.make_twitter_blurb(
            self.theme, 
            "/fake/image/path.jpg", 
            self.mock_config
        )
        
        assert len(blurb) <= 280, f"Blurb too long: {len(blurb)} characters"

    def test_make_twitter_blurb_error_handling(self):
        """Test error handling in Twitter blurb generation."""
        # Mock provider that raises an exception
        with patch('pixelbliss.prompts.get_provider') as mock_get_provider:
            mock_provider = Mock()
            mock_provider.make_twitter_blurb.side_effect = Exception("API Error")
            mock_get_provider.return_value = mock_provider
            
            # Should return empty string on error
            blurb = prompts.make_twitter_blurb(
                self.theme, 
                "/fake/image/path.jpg", 
                self.mock_config
            )
            
            assert blurb == ""

    def test_dummy_provider_make_twitter_blurb(self):
        """Test DummyLocalProvider's make_twitter_blurb method directly."""
        provider = DummyLocalProvider()
        
        blurb = provider.make_twitter_blurb(
            self.theme, 
            "/fake/image/path.jpg"
        )
        
        assert blurb is not None
        assert isinstance(blurb, str)
        assert len(blurb) > 0
        assert len(blurb) <= 280
        assert self.theme in blurb

    def test_dummy_provider_consistent_output(self):
        """Test that dummy provider gives consistent output for same theme."""
        provider = DummyLocalProvider()
        
        blurb1 = provider.make_twitter_blurb(
            self.theme, 
            "/fake/image/path1.jpg"
        )
        
        blurb2 = provider.make_twitter_blurb(
            self.theme, 
            "/fake/image/path2.jpg"
        )
        
        # Should be the same since dummy provider uses hash of theme
        assert blurb1 == blurb2

    def test_dummy_provider_different_themes(self):
        """Test that dummy provider gives different output for different themes."""
        provider = DummyLocalProvider()
        
        blurb1 = provider.make_twitter_blurb(
            "nature", 
            "/fake/image/path.jpg"
        )
        
        blurb2 = provider.make_twitter_blurb(
            "cosmic", 
            "/fake/image/path.jpg"
        )
        
        # Should be different for different themes
        assert blurb1 != blurb2

    @patch('pixelbliss.prompt_engine.openai_gpt5.AsyncOpenAI')
    @patch('pixelbliss.prompt_engine.openai_gpt5.OpenAI')
    @patch('builtins.open', create=True)
    @patch('pathlib.Path.read_bytes')
    def test_openai_provider_make_twitter_blurb(self, mock_read_bytes, mock_open, mock_openai_class, mock_async_openai_class):
        """Test OpenAIGPT5Provider's make_twitter_blurb method with vision."""
        # Mock image file reading
        mock_read_bytes.return_value = b"fake_image_data"
        
        # Mock the OpenAI client response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Nature whispers\nits ancient secrets—\npeace flows within."
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        mock_async_openai_class.return_value = Mock()
        
        provider = OpenAIGPT5Provider()
        
        blurb = provider.make_twitter_blurb(
            self.theme, 
            "/fake/image/path.jpg"
        )
        
        assert blurb == "Nature whispers\nits ancient secrets—\npeace flows within."
        
        # Verify the API was called with correct parameters
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        
        assert call_args[1]['model'] == 'gpt-5'
        assert call_args[1]['max_completion_tokens'] == 100
        assert len(call_args[1]['messages']) == 2
        assert call_args[1]['messages'][0]['role'] == 'system'
        assert call_args[1]['messages'][1]['role'] == 'user'
        
        # Check that the user message contains our theme and has multimodal content
        user_message = call_args[1]['messages'][1]['content']
        assert isinstance(user_message, list)
        assert len(user_message) == 2
        assert user_message[0]['type'] == 'text'
        assert self.theme in user_message[0]['text']
        assert user_message[1]['type'] == 'image_url'

    @patch('pixelbliss.prompt_engine.openai_gpt5.AsyncOpenAI')
    @patch('pixelbliss.prompt_engine.openai_gpt5.OpenAI')
    @patch('pathlib.Path.read_bytes')
    def test_openai_provider_character_limit_truncation(self, mock_read_bytes, mock_openai_class, mock_async_openai_class):
        """Test that OpenAI provider truncates long responses."""
        # Mock image file reading
        mock_read_bytes.return_value = b"fake_image_data"
        
        # Mock a very long response
        long_response = "This is a very long haiku that exceeds the character limit. " * 10
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = long_response
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        mock_async_openai_class.return_value = Mock()
        
        provider = OpenAIGPT5Provider()
        
        blurb = provider.make_twitter_blurb(
            self.theme, 
            "/fake/image/path.jpg"
        )
        
        assert len(blurb) <= 250  # Should be truncated
        assert blurb.endswith("...")  # Should have ellipsis

    @patch('pixelbliss.prompt_engine.openai_gpt5.AsyncOpenAI')
    @patch('pixelbliss.prompt_engine.openai_gpt5.OpenAI')
    @patch('pathlib.Path.read_bytes')
    def test_openai_provider_multiline_truncation(self, mock_read_bytes, mock_openai_class, mock_async_openai_class):
        """Test that OpenAI provider truncates multiline responses properly."""
        # Mock image file reading
        mock_read_bytes.return_value = b"fake_image_data"
        
        # Mock a multiline response where some lines are too long
        multiline_response = "Short line\nThis is a very long line that would exceed the character limit when combined with other lines and should be truncated properly\nAnother short line\nYet another line that makes it too long"
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = multiline_response
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        mock_async_openai_class.return_value = Mock()
        
        provider = OpenAIGPT5Provider()
        
        blurb = provider.make_twitter_blurb(
            self.theme, 
            "/fake/image/path.jpg"
        )
        
        assert len(blurb) <= 250
        # Should preserve line structure when possible
        if '\n' in blurb:
            lines = blurb.split('\n')
            assert len(lines) >= 1

    def test_get_provider_with_openai_config(self):
        """Test that get_provider returns OpenAI provider for openai config."""
        config = Mock(spec=Config)
        config.prompt_generation = Mock()
        config.prompt_generation.provider = "openai"
        config.prompt_generation.model = "gpt-5"
        
        with patch('pixelbliss.prompt_engine.openai_gpt5.AsyncOpenAI'), \
             patch('pixelbliss.prompt_engine.openai_gpt5.OpenAI'):
            provider = prompts.get_provider(config)
            assert isinstance(provider, OpenAIGPT5Provider)

    def test_get_provider_with_dummy_config(self):
        """Test that get_provider returns dummy provider for dummy config."""
        config = Mock(spec=Config)
        config.prompt_generation = Mock()
        config.prompt_generation.provider = "dummy"
        
        provider = prompts.get_provider(config)
        assert isinstance(provider, DummyLocalProvider)

    def test_get_provider_with_invalid_config(self):
        """Test that get_provider raises error for invalid config."""
        config = Mock(spec=Config)
        config.prompt_generation = Mock()
        config.prompt_generation.provider = "invalid"
        
        with pytest.raises(ValueError, match="Unknown prompt provider: invalid"):
            prompts.get_provider(config)

    def test_blurb_content_quality(self):
        """Test that generated blurbs have appropriate content."""
        provider = DummyLocalProvider()
        
        themes = ["nature", "cosmic", "harmony", "flow"]
        
        for theme in themes:
            blurb = provider.make_twitter_blurb(
                theme, 
                "/fake/image/path.jpg"
            )
            
            # Should contain the theme
            assert theme in blurb
            
            # Should be poetic (contain line breaks for haiku or be contemplative)
            assert '\n' in blurb or any(word in blurb.lower() for word in ['find', 'moment', 'beauty', 'peace', 'wonder'])
            
            # Should not be empty or just whitespace
            assert blurb.strip()

    def test_integration_with_different_themes(self):
        """Test integration with various themes."""
        themes = [
            "abstract", "nature", "cosmic", "geometric", "organic", 
            "crystalline", "flow", "balance", "harmony", "unity"
        ]
        
        for theme in themes:
            blurb = prompts.make_twitter_blurb(
                theme, 
                "/fake/image/path.jpg", 
                self.mock_config
            )
            
            assert isinstance(blurb, str)
            assert len(blurb) <= 280
            # For dummy provider, should contain the theme
            if self.mock_config.prompt_generation.provider == "dummy":
                assert theme in blurb

    @patch('pixelbliss.prompt_engine.openai_gpt5.AsyncOpenAI')
    @patch('pixelbliss.prompt_engine.openai_gpt5.OpenAI')
    @patch('pathlib.Path.read_bytes')
    def test_openai_provider_image_read_failure(self, mock_read_bytes, mock_openai_class, mock_async_openai_class):
        """Test fallback when image reading fails."""
        # Mock image file reading failure
        mock_read_bytes.side_effect = Exception("File not found")
        
        # Mock the fallback text-only response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Nature finds its way."
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        mock_async_openai_class.return_value = Mock()
        
        provider = OpenAIGPT5Provider()
        
        blurb = provider.make_twitter_blurb(
            self.theme, 
            "/nonexistent/image/path.jpg"
        )
        
        assert blurb == "Nature finds its way."
        
        # Should have called the fallback text-only method
        mock_client.chat.completions.create.assert_called_once()
