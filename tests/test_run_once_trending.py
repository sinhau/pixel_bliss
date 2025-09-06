"""
Tests for trending topics integration in run_once.py.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from pixelbliss.run_once import generate_theme_hint_async
from pixelbliss.config import Config, TrendingThemes


class TestTrendingThemesIntegration:
    """Test cases for trending themes integration in run_once.py."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config.trending_themes = TrendingThemes(
            enabled=True,
            provider="openai",
            model="gpt-5",
            fallback_enabled=True,
            async_enabled=True
        )
    
    
    @patch('pixelbliss.prompt_engine.trending_topics.TrendingTopicsProvider')
    @pytest.mark.asyncio
    async def test_generate_theme_hint_async_trending_enabled(self, mock_provider_class):
        """Test async theme generation with trending topics enabled."""
        # Mock provider
        mock_provider = Mock()
        mock_provider_class.return_value = mock_provider
        mock_provider.get_trending_theme_async = AsyncMock(return_value="geometric harmony")
        
        # Test
        theme = await generate_theme_hint_async(self.config)
        
        assert theme == "geometric harmony"
        mock_provider_class.assert_called_once_with(model="gpt-5")
        mock_provider.get_trending_theme_async.assert_called_once_with(None)
    
    @patch('pixelbliss.prompt_engine.trending_topics.TrendingTopicsProvider')
    @pytest.mark.asyncio
    async def test_generate_theme_hint_async_with_progress_logger(self, mock_provider_class):
        """Test async theme generation with progress logger."""
        # Mock provider
        mock_provider = Mock()
        mock_provider_class.return_value = mock_provider
        mock_provider.get_trending_theme_async = AsyncMock(return_value="minimalist zen")
        
        # Mock progress logger
        mock_progress_logger = Mock()
        
        # Test
        theme = await generate_theme_hint_async(self.config, mock_progress_logger)
        
        assert theme == "minimalist zen"
        mock_provider.get_trending_theme_async.assert_called_once_with(mock_progress_logger)
    
    @pytest.mark.asyncio
    async def test_generate_theme_hint_async_trending_disabled(self):
        """Test async theme generation with trending topics disabled."""
        # Disable trending themes
        self.config.trending_themes.enabled = False
        
        # Test
        theme = await generate_theme_hint_async(self.config)
        
        # Should return one of the fallback themes
        fallback_themes = [
            "abstract", "nature", "cosmic", "geometric", "organic", "crystalline", "flow",
            "balance", "harmony", "unity", "duality", "symmetry", "asymmetry",
            "cycles", "growth", "renewal", "emergence", "evolution",
            "interconnection", "networks", "continuum", "wholeness", "infinity",
            "order and randomness", "pattern", "repetition", "rhythm",
            "fractal", "spirals", "tessellation", "lattice", "grid",
            "waveforms", "fields", "orbits", "constellations", "topography", "cartography",
            "elemental", "terrestrial", "celestial", "aquatic", "mineral",
            "botanical", "aerial", "seasonal", "weather",
            "journey", "thresholds", "liminality", "sanctuary", "play",
            "curiosity", "wonder", "stillness", "openness", "simplicity",
            "order and flow", "cause and effect", "microcosm and macrocosm"
        ]
        assert theme in fallback_themes
    
    @patch('pixelbliss.prompt_engine.trending_topics.TrendingTopicsProvider')
    @pytest.mark.asyncio
    async def test_generate_theme_hint_async_trending_failure_with_fallback(self, mock_provider_class):
        """Test async theme generation when trending fails but fallback is enabled."""
        # Mock provider to raise exception
        mock_provider = Mock()
        mock_provider_class.return_value = mock_provider
        mock_provider.get_trending_theme_async = AsyncMock(side_effect=Exception("API Error"))
        
        # Mock progress logger
        mock_progress_logger = Mock()
        
        # Test
        theme = await generate_theme_hint_async(self.config, mock_progress_logger)
        
        # Should return one of the fallback themes
        fallback_themes = [
            "abstract", "nature", "cosmic", "geometric", "organic", "crystalline", "flow",
            "balance", "harmony", "unity", "duality", "symmetry", "asymmetry",
            "cycles", "growth", "renewal", "emergence", "evolution",
            "interconnection", "networks", "continuum", "wholeness", "infinity",
            "order and randomness", "pattern", "repetition", "rhythm",
            "fractal", "spirals", "tessellation", "lattice", "grid",
            "waveforms", "fields", "orbits", "constellations", "topography", "cartography",
            "elemental", "terrestrial", "celestial", "aquatic", "mineral",
            "botanical", "aerial", "seasonal", "weather",
            "journey", "thresholds", "liminality", "sanctuary", "play",
            "curiosity", "wonder", "stillness", "openness", "simplicity",
            "order and flow", "cause and effect", "microcosm and macrocosm"
        ]
        assert theme in fallback_themes
        
        # Verify progress logger was called
        mock_progress_logger.warning.assert_called_with(
            "Trending topics failed, using fallback themes"
        )
    
    @patch('pixelbliss.prompt_engine.trending_topics.TrendingTopicsProvider')
    @pytest.mark.asyncio
    async def test_generate_theme_hint_async_trending_failure_no_fallback(self, mock_provider_class):
        """Test async theme generation when trending fails and fallback is disabled."""
        # Disable fallback
        self.config.trending_themes.fallback_enabled = False
        
        # Mock provider to raise exception
        mock_provider = Mock()
        mock_provider_class.return_value = mock_provider
        mock_provider.get_trending_theme_async = AsyncMock(side_effect=Exception("API Error"))
        
        # Test - should raise the exception
        with pytest.raises(Exception, match="API Error"):
            await generate_theme_hint_async(self.config)
    
    def test_fallback_themes_consistency(self):
        """Test that fallback themes are consistent in async version."""
        # Disable trending to force fallback
        self.config.trending_themes.enabled = False
        
        # Get themes from async function multiple times
        import asyncio
        async def collect_async_themes():
            themes = set()
            for _ in range(50):
                theme = await generate_theme_hint_async(self.config)
                themes.add(theme)
            return themes
        
        async_themes = asyncio.run(collect_async_themes())
        
        # Should draw from the expected pool of themes
        expected_themes = {
            "abstract", "nature", "cosmic", "geometric", "organic", "crystalline", "flow",
            "balance", "harmony", "unity", "duality", "symmetry", "asymmetry",
            "cycles", "growth", "renewal", "emergence", "evolution",
            "interconnection", "networks", "continuum", "wholeness", "infinity",
            "order and randomness", "pattern", "repetition", "rhythm",
            "fractal", "spirals", "tessellation", "lattice", "grid",
            "waveforms", "fields", "orbits", "constellations", "topography", "cartography",
            "elemental", "terrestrial", "celestial", "aquatic", "mineral",
            "botanical", "aerial", "seasonal", "weather",
            "journey", "thresholds", "liminality", "sanctuary", "play",
            "curiosity", "wonder", "stillness", "openness", "simplicity",
            "order and flow", "cause and effect", "microcosm and macrocosm"
        }
        
        # All themes should be from the expected set
        assert async_themes.issubset(expected_themes)
