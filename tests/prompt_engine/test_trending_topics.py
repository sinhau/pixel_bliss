"""
Tests for the trending topics provider.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from pixelbliss.prompt_engine.trending_topics import TrendingTopicsProvider


class TestTrendingTopicsProvider:
    """Test cases for TrendingTopicsProvider."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('pixelbliss.prompt_engine.trending_topics.OpenAI'), \
             patch('pixelbliss.prompt_engine.trending_topics.AsyncOpenAI'):
            self.provider = TrendingTopicsProvider(model="gpt-5")
    
    def test_init(self):
        """Test provider initialization."""
        assert self.provider.model == "gpt-5"
        assert hasattr(self.provider, 'client')
        assert hasattr(self.provider, 'async_client')
        assert hasattr(self.provider, 'logger')
    
    def test_get_trending_theme_success(self):
        """Test successful trending theme generation."""
        # Mock the client directly
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "aurora borealis"
        self.provider.client.chat.completions.create.return_value = mock_response
        
        # Test
        theme = self.provider.get_trending_theme()
        
        assert theme == "aurora borealis"
        self.provider.client.chat.completions.create.assert_called_once()
    
    def test_get_trending_theme_with_cleanup(self):
        """Test theme generation with text cleanup."""
        # Mock the client directly
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '"Cherry Blossoms"!'
        self.provider.client.chat.completions.create.return_value = mock_response
        
        # Test
        theme = self.provider.get_trending_theme()
        
        assert theme == "cherry blossoms"  # Should be cleaned up
    
    def test_get_trending_theme_failure_fallback(self):
        """Test fallback when trending theme generation fails."""
        # Mock the client to raise exception
        self.provider.client.chat.completions.create.side_effect = Exception("API Error")
        
        # Test
        theme = self.provider.get_trending_theme()
        
        # Should return one of the fallback themes
        fallback_themes = [
            "aurora borealis", "cherry blossoms", "cosmic wonder", "minimalist zen",
            "golden hour", "ocean waves", "mountain peaks", "forest mist",
            "desert dunes", "city lights", "abstract flow", "geometric harmony"
        ]
        assert theme in fallback_themes
    
    @pytest.mark.asyncio
    async def test_get_trending_theme_async_success(self):
        """Test successful async trending theme generation."""
        # Mock the async client directly with AsyncMock
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "geometric harmony"
        self.provider.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Test
        theme = await self.provider.get_trending_theme_async()
        
        assert theme == "geometric harmony"
        self.provider.async_client.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_trending_theme_async_failure_fallback(self):
        """Test async fallback when trending theme generation fails."""
        # Mock the async client to raise exception
        self.provider.async_client.chat.completions.create.side_effect = Exception("API Error")
        
        # Test
        theme = await self.provider.get_trending_theme_async()
        
        # Should return one of the fallback themes
        fallback_themes = [
            "aurora borealis", "cherry blossoms", "cosmic wonder", "minimalist zen",
            "golden hour", "ocean waves", "mountain peaks", "forest mist",
            "desert dunes", "city lights", "abstract flow", "geometric harmony"
        ]
        assert theme in fallback_themes
    
    def test_progress_logger_integration(self):
        """Test integration with progress logger."""
        # Mock progress logger
        mock_progress_logger = Mock()
        
        # Mock successful API call
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "forest mist"
        self.provider.client.chat.completions.create.return_value = mock_response
        
        # Test
        theme = self.provider.get_trending_theme(mock_progress_logger)
        
        assert theme == "forest mist"
        
        # Verify progress logger calls
        expected_calls = [
            ("substep", ("Fetching trending topics from web",)),
            ("substep", ("Trending theme generated", "forest mist"))
        ]
        
        actual_calls = [(call[0], call[1]) for call in mock_progress_logger.method_calls]
        for expected_call in expected_calls:
            assert expected_call in actual_calls
    
    def test_progress_logger_with_failure(self):
        """Test progress logger with API failure."""
        # Mock the client to raise exception
        self.provider.client.chat.completions.create.side_effect = Exception("API Error")
        
        # Mock progress logger
        mock_progress_logger = Mock()
        
        # Test
        theme = self.provider.get_trending_theme(mock_progress_logger)
        
        # Should get fallback theme
        fallback_themes = [
            "aurora borealis", "cherry blossoms", "cosmic wonder", "minimalist zen",
            "golden hour", "ocean waves", "mountain peaks", "forest mist",
            "desert dunes", "city lights", "abstract flow", "geometric harmony"
        ]
        assert theme in fallback_themes
        
        # Verify warning was logged
        mock_progress_logger.warning.assert_called_with(
            "Trending theme failed, using fallback"
        )
    
    def test_system_prompt_content(self):
        """Test that system prompt contains expected content."""
        # Mock the client directly
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "test theme"
        self.provider.client.chat.completions.create.return_value = mock_response
        
        # Test
        self.provider.get_trending_theme()
        
        # Verify the call was made
        self.provider.client.chat.completions.create.assert_called_once()
        call_args = self.provider.client.chat.completions.create.call_args
        
        # Check system prompt content
        messages = call_args[1]['messages']
        system_message = messages[0]
        
        assert system_message['role'] == 'system'
        assert 'PixelBliss Trend Analyst' in system_message['content']
        assert 'current web trends' in system_message['content']
        assert 'wallpaper' in system_message['content']
    
    def test_user_prompt_content(self):
        """Test that user prompt contains expected content."""
        # Mock the client directly
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "test theme"
        self.provider.client.chat.completions.create.return_value = mock_response
        
        # Test
        self.provider.get_trending_theme()
        
        # Verify the call was made
        call_args = self.provider.client.chat.completions.create.call_args
        
        # Check user prompt content
        messages = call_args[1]['messages']
        user_message = messages[1]
        
        assert user_message['role'] == 'user'
        assert 'Search the web' in user_message['content']
        assert 'current trending topics' in user_message['content']
        assert 'cultural movements' in user_message['content']
        assert 'seasonal themes' in user_message['content']
    
    def test_api_parameters(self):
        """Test that API is called with correct parameters."""
        # Mock the client directly
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "test theme"
        self.provider.client.chat.completions.create.return_value = mock_response
        
        # Test
        self.provider.get_trending_theme()
        
        # Verify API parameters
        call_args = self.provider.client.chat.completions.create.call_args
        
        assert call_args[1]['model'] == 'gpt-5'
        assert call_args[1]['max_completion_tokens'] == 100
        assert call_args[1]['temperature'] == 0.7
        assert len(call_args[1]['messages']) == 2
