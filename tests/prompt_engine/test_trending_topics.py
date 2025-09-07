"""
Tests for the trending topics provider.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from pixelbliss.prompt_engine.trending_topics import TrendingTopicsProvider, ThemeRecommendation, MultipleThemeRecommendations


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
        # Mock phase 1: web search (responses.create)
        mock_research_response = Mock()
        mock_research_response.choices = [Mock()]
        mock_research_response.choices[0].message.content = "- Trend 1\n- Trend 2"
        self.provider.async_client.responses.create = AsyncMock(return_value=mock_research_response)

        # Mock phase 2: structured parse (responses.parse)
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_theme_recommendation = Mock()
        mock_theme_recommendation.theme = "Geometric harmony with clean lines and modern minimalist aesthetics"
        mock_theme_recommendation.reasoning = "Trending due to modern design movements"
        mock_response.choices[0].message.parsed = mock_theme_recommendation
        self.provider.async_client.responses.parse = AsyncMock(return_value=mock_response)
        
        # Test
        theme = await self.provider.get_trending_theme_async()
        
        assert theme == "Geometric harmony with clean lines and modern minimalist aesthetics"
        self.provider.async_client.responses.parse.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_trending_theme_async_failure_propagates(self):
        """Test that async API failures propagate as expected."""
        # Mock phase 1 succeed
        mock_research_response = Mock()
        mock_research_response.choices = [Mock()]
        mock_research_response.choices[0].message.content = "- Trend A\n- Trend B"
        self.provider.async_client.responses.create = AsyncMock(return_value=mock_research_response)
        # Mock phase 2 raise
        self.provider.async_client.responses.parse = AsyncMock(side_effect=Exception("API Error"))
        
        # Test - should raise the exception since there's no fallback handling
        with pytest.raises(Exception, match="API Error"):
            await self.provider.get_trending_theme_async()
    
    @pytest.mark.asyncio
    async def test_get_trending_theme_async_with_cleanup(self):
        """Test async theme generation with text cleanup."""
        # Mock phase 1: web search (responses.create)
        mock_research_response = Mock()
        mock_research_response.choices = [Mock()]
        mock_research_response.choices[0].message.content = "- Spring blossoms\n- Seasonal festivals"
        self.provider.async_client.responses.create = AsyncMock(return_value=mock_research_response)

        # Mock phase 2: structured parse (responses.parse)
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_theme_recommendation = Mock()
        mock_theme_recommendation.theme = "  Delicate cherry blossoms in full bloom creating a dreamy pink canopy  "
        mock_theme_recommendation.reasoning = "Spring seasonal trend"
        mock_response.choices[0].message.parsed = mock_theme_recommendation
        self.provider.async_client.responses.parse = AsyncMock(return_value=mock_response)
        
        # Test
        theme = await self.provider.get_trending_theme_async()
        
        assert theme == "Delicate cherry blossoms in full bloom creating a dreamy pink canopy"  # Should be cleaned up

    @pytest.mark.asyncio
    async def test_progress_logger_integration_async(self):
        """Test async integration with progress logger."""
        # Mock progress logger
        mock_progress_logger = Mock()
        
        # Phase 1
        mock_research_response = Mock()
        mock_research_response.choices = [Mock()]
        mock_research_response.choices[0].message.content = "- Nature\n- Ethereal light"
        self.provider.async_client.responses.create = AsyncMock(return_value=mock_research_response)

        # Phase 2
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_theme_recommendation = Mock()
        mock_theme_recommendation.theme = "Misty forest with ethereal morning light filtering through ancient trees"
        mock_theme_recommendation.reasoning = "Nature trends are popular"
        mock_response.choices[0].message.parsed = mock_theme_recommendation
        self.provider.async_client.responses.parse = AsyncMock(return_value=mock_response)
        
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
        # Phase 1 succeed
        mock_research_response = Mock()
        mock_research_response.choices = [Mock()]
        mock_research_response.choices[0].message.content = "- Item 1\n- Item 2"
        self.provider.async_client.responses.create = AsyncMock(return_value=mock_research_response)

        # Phase 2 fails
        self.provider.async_client.responses.parse = AsyncMock(side_effect=Exception("API Error"))
        
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
        # Phase 1 (responses.create) mock
        mock_research_response = Mock()
        mock_research_response.choices = [Mock()]
        mock_research_response.choices[0].message.content = "- Key trend 1\n- Key trend 2"
        self.provider.async_client.responses.create = AsyncMock(return_value=mock_research_response)

        # Phase 2 (responses.parse) mock
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_theme_recommendation = Mock()
        mock_theme_recommendation.theme = "Modern abstract geometric patterns with vibrant color gradients"
        mock_theme_recommendation.reasoning = "Test reasoning"
        mock_response.choices[0].message.parsed = mock_theme_recommendation
        self.provider.async_client.responses.parse = AsyncMock(return_value=mock_response)
        
        # Test
        await self.provider.get_trending_theme_async()
        
        # Verify API parameters for phase 1 (web_search)
        call_args_create = self.provider.async_client.responses.create.call_args
        assert call_args_create[1]['model'] == 'gpt-5'
        assert 'instructions' in call_args_create[1]
        assert isinstance(call_args_create[1]['instructions'], str) and len(call_args_create[1]['instructions']) > 0
        assert 'input' in call_args_create[1]
        assert isinstance(call_args_create[1]['input'], str) and len(call_args_create[1]['input']) > 0
        assert call_args_create[1]['tools'] == [{"type": "web_search"}]
        # Ensure we don't pass messages when using Responses API
        assert 'messages' not in call_args_create[1]

        # Verify API parameters for phase 2 (structured outputs)
        call_args_parse = self.provider.async_client.responses.parse.call_args
        assert call_args_parse[1]['model'] == 'gpt-5'
        assert 'instructions' in call_args_parse[1]
        assert 'input' in call_args_parse[1]
        assert 'text_format' in call_args_parse[1]  # Verify structured outputs are used
        # Ensure we don't pass tools to parse phase
        assert 'tools' not in call_args_parse[1]
        # Ensure we don't pass messages
        assert 'messages' not in call_args_parse[1]

    @pytest.mark.asyncio
    async def test_get_trending_themes_async_success(self):
        """Test successful async multiple themes generation."""
        # Mock phase 1: web search (responses.create)
        mock_research_response = Mock()
        mock_research_response.choices = [Mock()]
        mock_research_response.choices[0].message.content = "- Nature trends\n- Abstract art\n- Seasonal themes"
        self.provider.async_client.responses.create = AsyncMock(return_value=mock_research_response)

        # Mock phase 2: structured parse (responses.parse)
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_themes = [
            ThemeRecommendation(theme="Misty forest landscapes", reasoning="Nature is trending"),
            ThemeRecommendation(theme="Geometric abstract patterns", reasoning="Modern design movement"),
            ThemeRecommendation(theme="Autumn color palettes", reasoning="Seasonal relevance")
        ]
        mock_multiple_themes = MultipleThemeRecommendations(themes=mock_themes)
        mock_response.choices[0].message.parsed = mock_multiple_themes
        self.provider.async_client.responses.parse = AsyncMock(return_value=mock_response)
        
        # Test
        themes = await self.provider.get_trending_themes_async()
        
        assert len(themes) == 3
        assert themes[0].theme == "Misty forest landscapes"
        assert themes[1].theme == "Geometric abstract patterns"
        assert themes[2].theme == "Autumn color palettes"
        self.provider.async_client.responses.parse.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_trending_themes_async_with_progress_logger(self):
        """Test multiple themes generation with progress logger."""
        # Mock progress logger
        mock_progress_logger = Mock()
        
        # Phase 1
        mock_research_response = Mock()
        mock_research_response.choices = [Mock()]
        mock_research_response.choices[0].message.content = "- Cultural trends\n- Tech aesthetics"
        self.provider.async_client.responses.create = AsyncMock(return_value=mock_research_response)

        # Phase 2
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_themes = [
            ThemeRecommendation(theme="Cyberpunk cityscapes", reasoning="Tech culture trending"),
            ThemeRecommendation(theme="Minimalist zen gardens", reasoning="Wellness movement"),
            ThemeRecommendation(theme="Retro synthwave aesthetics", reasoning="Nostalgia trend")
        ]
        mock_multiple_themes = MultipleThemeRecommendations(themes=mock_themes)
        mock_response.choices[0].message.parsed = mock_multiple_themes
        self.provider.async_client.responses.parse = AsyncMock(return_value=mock_response)
        
        # Test
        themes = await self.provider.get_trending_themes_async(mock_progress_logger)
        
        assert len(themes) == 3
        assert themes[0].theme == "Cyberpunk cityscapes"
        
        # Verify progress logger calls
        expected_calls = [
            ("substep", ("Fetching trending topics for multiple themes",)),
            ("substep", ("Multiple trending themes generated", "3 themes"))
        ]
        
        actual_calls = [(call[0], call[1]) for call in mock_progress_logger.method_calls]
        for expected_call in expected_calls:
            assert expected_call in actual_calls

    @pytest.mark.asyncio
    async def test_get_trending_themes_async_failure_propagates(self):
        """Test that multiple themes API failures propagate as expected."""
        # Mock phase 1 succeed
        mock_research_response = Mock()
        mock_research_response.choices = [Mock()]
        mock_research_response.choices[0].message.content = "- Trend A\n- Trend B"
        self.provider.async_client.responses.create = AsyncMock(return_value=mock_research_response)
        # Mock phase 2 raise
        self.provider.async_client.responses.parse = AsyncMock(side_effect=Exception("API Error"))
        
        # Test - should raise the exception since there's no fallback handling
        with pytest.raises(Exception, match="API Error"):
            await self.provider.get_trending_themes_async()

    @pytest.mark.asyncio
    async def test_get_trending_themes_async_api_parameters(self):
        """Test that multiple themes API is called with correct parameters."""
        # Phase 1 (responses.create) mock
        mock_research_response = Mock()
        mock_research_response.choices = [Mock()]
        mock_research_response.choices[0].message.content = "- Key trend 1\n- Key trend 2"
        self.provider.async_client.responses.create = AsyncMock(return_value=mock_research_response)

        # Phase 2 (responses.parse) mock
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_themes = [
            ThemeRecommendation(theme="Theme 1", reasoning="Reason 1"),
            ThemeRecommendation(theme="Theme 2", reasoning="Reason 2"),
            ThemeRecommendation(theme="Theme 3", reasoning="Reason 3")
        ]
        mock_multiple_themes = MultipleThemeRecommendations(themes=mock_themes)
        mock_response.choices[0].message.parsed = mock_multiple_themes
        self.provider.async_client.responses.parse = AsyncMock(return_value=mock_response)
        
        # Test
        await self.provider.get_trending_themes_async()
        
        # Verify API parameters for phase 2 (structured outputs)
        call_args_parse = self.provider.async_client.responses.parse.call_args
        assert call_args_parse[1]['model'] == 'gpt-5'
        assert 'instructions' in call_args_parse[1]
        assert 'input' in call_args_parse[1]
        assert 'text_format' in call_args_parse[1]  # Verify structured outputs are used
        # Should use MultipleThemeRecommendations format
        assert call_args_parse[1]['text_format'] == MultipleThemeRecommendations
