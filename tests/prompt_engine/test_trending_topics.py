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
        with patch('pixelbliss.prompt_engine.trending_topics.AsyncOpenAI'):
            self.provider = TrendingTopicsProvider(model="gpt-5")
    
    def test_init(self):
        """Test provider initialization."""
        assert self.provider.model == "gpt-5"
        assert hasattr(self.provider, 'async_client')
        assert hasattr(self.provider, 'logger')
    
    
    
    
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
    
    @pytest.mark.asyncio
    async def test_get_trending_theme_async_with_cleanup(self):
        """Test async theme generation with text cleanup."""
        # Mock the async client directly
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '"Cherry Blossoms"!'
        self.provider.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Test
        theme = await self.provider.get_trending_theme_async()
        
        assert theme == "cherry blossoms"  # Should be cleaned up

    @pytest.mark.asyncio
    async def test_progress_logger_integration_async(self):
        """Test async integration with progress logger."""
        # Mock progress logger
        mock_progress_logger = Mock()
        
        # Mock successful API call
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "forest mist"
        self.provider.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Test
        theme = await self.provider.get_trending_theme_async(mock_progress_logger)
        
        assert theme == "forest mist"
        
        # Verify progress logger calls
        expected_calls = [
            ("substep", ("Fetching trending topics from web",)),
            ("substep", ("Trending theme generated", "forest mist"))
        ]
        
        actual_calls = [(call[0], call[1]) for call in mock_progress_logger.method_calls]
        for expected_call in expected_calls:
            assert expected_call in actual_calls

    @pytest.mark.asyncio
    async def test_progress_logger_with_failure_async(self):
        """Test async progress logger with API failure."""
        # Mock the async client to raise exception
        self.provider.async_client.chat.completions.create.side_effect = Exception("API Error")
        
        # Mock progress logger
        mock_progress_logger = Mock()
        
        # Test
        theme = await self.provider.get_trending_theme_async(mock_progress_logger)
        
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

    @pytest.mark.asyncio
    async def test_api_parameters_async(self):
        """Test that async API is called with correct parameters."""
        # Mock the async client directly
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "test theme"
        self.provider.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Test
        await self.provider.get_trending_theme_async()
        
        # Verify API parameters
        call_args = self.provider.async_client.chat.completions.create.call_args
        
        assert call_args[1]['model'] == 'gpt-5'
        assert call_args[1]['max_completion_tokens'] == 100
        assert call_args[1]['temperature'] == 0.7
        assert len(call_args[1]['messages']) == 2
