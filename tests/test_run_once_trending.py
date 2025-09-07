"""
Tests for trending topics integration in run_once.py.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from pixelbliss.run_once import generate_theme_hint_async
from pixelbliss.config import Config, TrendingThemes, Discord
from pixelbliss.prompt_engine.trending_topics import ThemeRecommendation


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
        self.config.discord = Discord(
            enabled=False,  # Default to disabled for most tests
            bot_token_env="DISCORD_BOT_TOKEN",
            user_id_env="DISCORD_USER_ID",
            timeout_sec=60,
            batch_size=5
        )
    
    
    @patch('pixelbliss.prompt_engine.trending_topics.TrendingTopicsProvider')
    @pytest.mark.asyncio
    async def test_generate_theme_hint_async_trending_enabled_discord_disabled(self, mock_provider_class):
        """Test async theme generation with trending topics enabled but Discord disabled."""
        # Mock provider
        mock_provider = Mock()
        mock_provider_class.return_value = mock_provider
        mock_themes = [
            ThemeRecommendation(theme="geometric harmony", reasoning="Modern design trend"),
            ThemeRecommendation(theme="nature landscapes", reasoning="Seasonal trend")
        ]
        mock_provider.get_trending_themes_async = AsyncMock(return_value=mock_themes)
        
        # Test
        theme = await generate_theme_hint_async(self.config)
        
        # Should use random theme when Discord disabled
        assert theme in ["geometric harmony", "nature landscapes"]
        mock_provider_class.assert_called_once_with(model="gpt-5")
        mock_provider.get_trending_themes_async.assert_called_once_with(None)
    
    @patch('pixelbliss.prompt_engine.trending_topics.TrendingTopicsProvider')
    @pytest.mark.asyncio
    async def test_generate_theme_hint_async_with_progress_logger(self, mock_provider_class):
        """Test async theme generation with progress logger."""
        # Mock provider
        mock_provider = Mock()
        mock_provider_class.return_value = mock_provider
        mock_themes = [
            ThemeRecommendation(theme="minimalist zen", reasoning="Wellness trend")
        ]
        mock_provider.get_trending_themes_async = AsyncMock(return_value=mock_themes)
        
        # Mock progress logger
        mock_progress_logger = Mock()
        
        # Test
        theme = await generate_theme_hint_async(self.config, mock_progress_logger)
        
        assert theme == "minimalist zen"
        mock_provider.get_trending_themes_async.assert_called_once_with(mock_progress_logger)
    
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
        mock_provider.get_trending_themes_async = AsyncMock(side_effect=Exception("API Error"))
        
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
        mock_provider.get_trending_themes_async = AsyncMock(side_effect=Exception("API Error"))
        
        # Test - should raise the exception
        with pytest.raises(Exception, match="API Error"):
            await generate_theme_hint_async(self.config)

    @patch('pixelbliss.alerts.discord_select.ask_user_to_select_theme')
    @patch('pixelbliss.prompt_engine.trending_topics.TrendingTopicsProvider')
    @pytest.mark.asyncio
    async def test_generate_theme_hint_async_with_discord_selection(self, mock_provider_class, mock_discord_select):
        """Test async theme generation with Discord theme selection."""
        # Enable Discord
        self.config.discord.enabled = True
        
        # Mock provider
        mock_provider = Mock()
        mock_provider_class.return_value = mock_provider
        mock_themes = [
            ThemeRecommendation(theme="cyberpunk cityscapes", reasoning="Tech trend"),
            ThemeRecommendation(theme="zen gardens", reasoning="Wellness trend"),
            ThemeRecommendation(theme="abstract geometry", reasoning="Art trend")
        ]
        mock_provider.get_trending_themes_async = AsyncMock(return_value=mock_themes)
        
        # Mock Discord selection - user selects second theme
        async def mock_select_theme(*args, **kwargs):
            return "zen gardens"
        mock_discord_select.side_effect = mock_select_theme
        
        # Test
        theme = await generate_theme_hint_async(self.config)
        
        assert theme == "zen gardens"
        mock_provider.get_trending_themes_async.assert_called_once_with(None)
        mock_discord_select.assert_called_once()

    @patch('pixelbliss.alerts.discord_select.ask_user_to_select_theme')
    @patch('pixelbliss.prompt_engine.trending_topics.TrendingTopicsProvider')
    @pytest.mark.asyncio
    async def test_generate_theme_hint_async_discord_fallback_selection(self, mock_provider_class, mock_discord_select):
        """Test async theme generation when user selects fallback via Discord."""
        # Enable Discord
        self.config.discord.enabled = True
        
        # Mock provider
        mock_provider = Mock()
        mock_provider_class.return_value = mock_provider
        mock_themes = [
            ThemeRecommendation(theme="theme1", reasoning="reason1"),
            ThemeRecommendation(theme="theme2", reasoning="reason2")
        ]
        mock_provider.get_trending_themes_async = AsyncMock(return_value=mock_themes)
        
        # Mock Discord selection - user selects fallback
        async def mock_select_fallback(*args, **kwargs):
            return "fallback"
        mock_discord_select.side_effect = mock_select_fallback
        
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

    @patch('pixelbliss.alerts.discord_select.ask_user_to_select_theme')
    @patch('pixelbliss.prompt_engine.trending_topics.TrendingTopicsProvider')
    @pytest.mark.asyncio
    async def test_generate_theme_hint_async_discord_timeout(self, mock_provider_class, mock_discord_select):
        """Test async theme generation when Discord selection times out."""
        # Enable Discord
        self.config.discord.enabled = True
        
        # Mock provider
        mock_provider = Mock()
        mock_provider_class.return_value = mock_provider
        mock_themes = [
            ThemeRecommendation(theme="timeout theme 1", reasoning="timeout reason 1"),
            ThemeRecommendation(theme="timeout theme 2", reasoning="timeout reason 2")
        ]
        mock_provider.get_trending_themes_async = AsyncMock(return_value=mock_themes)
        
        # Mock Discord selection - timeout (returns None)
        async def mock_select_timeout(*args, **kwargs):
            return None
        mock_discord_select.side_effect = mock_select_timeout
        
        # Test
        theme = await generate_theme_hint_async(self.config)
        
        # Should use random theme as fallback
        assert theme in ["timeout theme 1", "timeout theme 2"]
    
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
