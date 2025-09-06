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
        mock_theme_recommendation = Mock()
        mock_theme_recommendation.theme = "Geometric harmony with clean lines and modern minimalist aesthetics"
        mock_theme_recommendation.reasoning = "Trending due to modern design movements"
        mock_response.choices[0].message.parsed = mock_theme_recommendation
        self.provider.async_client.chat.completions.parse = AsyncMock(return_value=mock_response)
        
        # Test
        theme = await self.provider.get_trending_theme_async()
        
        assert theme == "Geometric harmony with clean lines and modern minimalist aesthetics"
        self.provider.async_client.chat.completions.parse.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_trending_theme_async_failure_propagates(self):
        """Test that async API failures propagate as expected."""
        # Mock the async client to raise exception
        self.provider.async_client.chat.completions.parse = AsyncMock(side_effect=Exception("API Error"))
        
        # Test - should raise the exception since there's no fallback handling
        with pytest.raises(Exception, match="API Error"):
            await self.provider.get_trending_theme_async()
    
    @pytest.mark.asyncio
    async def test_get_trending_theme_async_with_cleanup(self):
        """Test async theme generation with text cleanup."""
        # Mock the async client directly
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_theme_recommendation = Mock()
        mock_theme_recommendation.theme = "  Delicate cherry blossoms in full bloom creating a dreamy pink canopy  "
        mock_theme_recommendation.reasoning = "Spring seasonal trend"
        mock_response.choices[0].message.parsed = mock_theme_recommendation
        self.provider.async_client.chat.completions.parse = AsyncMock(return_value=mock_response)
        
        # Test
        theme = await self.provider.get_trending_theme_async()
        
        assert theme == "Delicate cherry blossoms in full bloom creating a dreamy pink canopy"  # Should be cleaned up

    @pytest.mark.asyncio
    async def test_progress_logger_integration_async(self):
        """Test async integration with progress logger."""
        # Mock progress logger
        mock_progress_logger = Mock()
        
        # Mock successful API call
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_theme_recommendation = Mock()
        mock_theme_recommendation.theme = "Misty forest with ethereal morning light filtering through ancient trees"
        mock_theme_recommendation.reasoning = "Nature trends are popular"
        mock_response.choices[0].message.parsed = mock_theme_recommendation
        self.provider.async_client.chat.completions.parse = AsyncMock(return_value=mock_response)
        
        # Test
        theme = await self.provider.get_trending_theme_async(mock_progress_logger)
        
        assert theme == "Misty forest with ethereal morning light filtering through ancient trees"
        
        # Verify progress logger calls
        expected_calls = [
            ("substep", ("Fetching trending topics from web",)),
            ("substep", ("Trending theme generated", "Misty forest with ethereal morning light filtering through ancient trees"))
        ]
        
        actual_calls = [(call[0], call[1]) for call in mock_progress_logger.method_calls]
        for expected_call in expected_calls:
            assert expected_call in actual_calls

    @pytest.mark.asyncio
    async def test_progress_logger_with_failure_async(self):
        """Test async progress logger with API failure."""
        # Mock the async client to raise exception
        self.provider.async_client.chat.completions.parse = AsyncMock(side_effect=Exception("API Error"))
        
        # Mock progress logger
        mock_progress_logger = Mock()
        
        # Test - should raise the exception since there's no fallback handling
        with pytest.raises(Exception, match="API Error"):
            await self.provider.get_trending_theme_async(mock_progress_logger)
        
        # Verify progress logger was called before the exception
        mock_progress_logger.substep.assert_called_with("Fetching trending topics from web")

    @pytest.mark.asyncio
    async def test_api_parameters_async(self):
        """Test that async API is called with correct parameters."""
        # Mock the async client directly
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_theme_recommendation = Mock()
        mock_theme_recommendation.theme = "Modern abstract geometric patterns with vibrant color gradients"
        mock_theme_recommendation.reasoning = "Test reasoning"
        mock_response.choices[0].message.parsed = mock_theme_recommendation
        self.provider.async_client.chat.completions.parse = AsyncMock(return_value=mock_response)
        
        # Test
        await self.provider.get_trending_theme_async()
        
        # Verify API parameters (based on your simplified implementation)
        call_args = self.provider.async_client.chat.completions.parse.call_args
        
        assert call_args[1]['model'] == 'gpt-5'
        assert len(call_args[1]['messages']) == 2
        assert call_args[1]['tools'] == [{"type": "web_search"}]
        assert 'response_format' in call_args[1]  # Verify structured outputs are used
        
        # Verify message structure
        messages = call_args[1]['messages']
        assert messages[0]['role'] == 'system'
        assert messages[1]['role'] == 'user'
