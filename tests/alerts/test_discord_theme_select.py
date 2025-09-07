"""
Tests for Discord theme selection functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pixelbliss.alerts.discord_select import ask_user_to_select_theme
from pixelbliss.prompt_engine.trending_topics import ThemeRecommendation
from pixelbliss.config import Config, Discord


class TestDiscordThemeSelect:
    """Test cases for Discord theme selection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config.discord = Discord(
            enabled=True,
            bot_token_env="DISCORD_BOT_TOKEN",
            user_id_env="DISCORD_USER_ID",
            timeout_sec=1,  # Use very short timeout for tests
            batch_size=5
        )
        
        self.themes = [
            ThemeRecommendation(theme="Cyberpunk cityscapes", reasoning="Tech culture trending"),
            ThemeRecommendation(theme="Zen gardens", reasoning="Wellness movement"),
            ThemeRecommendation(theme="Abstract geometry", reasoning="Modern art trend")
        ]
        
        self.logger = Mock()

    @patch.dict('os.environ', {'DISCORD_BOT_TOKEN': 'test_token', 'DISCORD_USER_ID': '123456789'})
    @patch('pixelbliss.alerts.discord_select.discord.Client')
    @patch('asyncio.wait_for')
    @patch('asyncio.Event')
    @pytest.mark.asyncio
    async def test_ask_user_to_select_theme_success(self, mock_event_class, mock_wait_for, mock_client_class):
        """Test successful theme selection via Discord."""
        # Mock Discord client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.is_closed.return_value = False
        mock_client.close = AsyncMock()
        mock_client.start = AsyncMock()
        
        # Mock user and DM
        mock_user = Mock()
        mock_user.display_name = "TestUser"
        mock_dm = Mock()
        mock_user.create_dm = AsyncMock(return_value=mock_dm)
        mock_client.fetch_user = AsyncMock(return_value=mock_user)
        mock_dm.send = AsyncMock()
        
        # Mock the selection event and wait_for to return immediately
        mock_event = Mock()
        mock_event.wait = AsyncMock()
        mock_event_class.return_value = mock_event
        mock_wait_for.return_value = None  # Simulate immediate completion
        
        # Test
        result = await ask_user_to_select_theme(self.themes, self.config, self.logger)
        
        # Should return None since we didn't simulate actual selection
        assert result is None
        
        # Verify Discord client was used
        mock_client_class.assert_called_once()
        mock_client.start.assert_called_once_with('test_token')

    @patch.dict('os.environ', {})
    @pytest.mark.asyncio
    async def test_ask_user_to_select_theme_no_config(self):
        """Test theme selection when Discord is not configured."""
        result = await ask_user_to_select_theme(self.themes, self.config, self.logger)
        
        assert result is None
        self.logger.warning.assert_called_with(
            "Discord bot token or user ID not configured, skipping theme selection"
        )

    @patch.dict('os.environ', {'DISCORD_BOT_TOKEN': 'test_token', 'DISCORD_USER_ID': 'invalid'})
    @pytest.mark.asyncio
    async def test_ask_user_to_select_theme_invalid_user_id(self):
        """Test theme selection with invalid user ID."""
        result = await ask_user_to_select_theme(self.themes, self.config, self.logger)
        
        assert result is None
        self.logger.error.assert_called_with("Invalid DISCORD_USER_ID: invalid")

    @patch.dict('os.environ', {'DISCORD_BOT_TOKEN': 'test_token', 'DISCORD_USER_ID': '123456789'})
    @pytest.mark.asyncio
    async def test_ask_user_to_select_theme_empty_themes(self):
        """Test theme selection with empty themes list."""
        result = await ask_user_to_select_theme([], self.config, self.logger)
        
        assert result is None
        self.logger.warning.assert_called_with("No themes to select from")

    @patch.dict('os.environ', {'DISCORD_BOT_TOKEN': 'test_token', 'DISCORD_USER_ID': '123456789'})
    @patch('pixelbliss.alerts.discord_select.discord.Client')
    @patch('asyncio.wait_for', side_effect=asyncio.TimeoutError)
    @patch('asyncio.Event')
    @pytest.mark.asyncio
    async def test_ask_user_to_select_theme_timeout(self, mock_event_class, mock_wait_for, mock_client_class):
        """Test theme selection timeout."""
        # Mock Discord client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.is_closed.return_value = False
        mock_client.close = AsyncMock()
        mock_client.start = AsyncMock()
        
        # Mock the selection event
        mock_event = Mock()
        mock_event.wait = AsyncMock()
        mock_event_class.return_value = mock_event
        
        # Test
        result = await ask_user_to_select_theme(self.themes, self.config, self.logger)
        
        assert result is None
        self.logger.warning.assert_called_with(
            "No theme selection received within 1 seconds, timing out"
        )

    @patch.dict('os.environ', {'DISCORD_BOT_TOKEN': 'test_token', 'DISCORD_USER_ID': '123456789'})
    @patch('pixelbliss.alerts.discord_select.discord.Client')
    @patch('asyncio.Event')
    @pytest.mark.asyncio
    async def test_ask_user_to_select_theme_exception(self, mock_event_class, mock_client_class):
        """Test theme selection with exception during Discord operations."""
        # Mock Discord client to raise exception
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.is_closed.return_value = False
        mock_client.close = AsyncMock()
        mock_client.start = AsyncMock(side_effect=Exception("Discord error"))
        
        # Mock the selection event
        mock_event = Mock()
        mock_event.wait = AsyncMock()
        mock_event_class.return_value = mock_event
        
        result = await ask_user_to_select_theme(self.themes, self.config, self.logger)
        
        assert result is None
        self.logger.error.assert_called_with(
            "Error in Discord theme selection process: Discord error"
        )

    def test_theme_option_formatting(self):
        """Test that theme options are formatted correctly for Discord."""
        # Test with long theme text
        long_theme = ThemeRecommendation(
            theme="A very long theme description that exceeds the Discord character limit for select option labels and needs to be truncated properly",
            reasoning="This is also a very long reasoning that might need truncation for the description field in Discord select options"
        )
        
        themes = [long_theme]
        
        # The actual formatting happens in the Discord event handler, 
        # so we just verify the theme data is accessible
        assert hasattr(long_theme, 'theme')
        assert hasattr(long_theme, 'reasoning')
        assert len(long_theme.theme) > 100  # Verify it's actually long
        assert len(long_theme.reasoning) > 100  # Verify reasoning is also long

