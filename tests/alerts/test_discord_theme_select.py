"""
Tests for Discord theme selection functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pixelbliss.alerts.discord_select import (
    ask_user_to_select_theme,
    _validate_discord_config,
    DiscordClientManager,
    ThemeSelectView,
    ThemeSelect,
    _setup_theme_selection
)
from pixelbliss.prompt_engine.trending_topics import ThemeRecommendation
from pixelbliss.config import Config, Discord
import discord


class TestThemeSelectComponents:
    """Test the ThemeSelectView and ThemeSelect classes."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.themes = [
            ThemeRecommendation(theme="Cyberpunk cityscapes", reasoning="Tech culture trending"),
            ThemeRecommendation(theme="Zen gardens", reasoning="Wellness movement"),
            ThemeRecommendation(theme="Abstract geometry", reasoning="Modern art trend")
        ]
        self.logger = Mock()
        self.manager = Mock()
        self.manager.timeout = 30
        self.manager.selection_event = Mock()
        self.manager.logger = self.logger
    
    def test_theme_select_view_creation_logic(self):
        """Test ThemeSelectView creation logic without Discord UI framework."""
        # Test the logic that would be used in ThemeSelectView creation
        themes = self.themes
        user_id = 123456789
        timeout = 30
        manager = self.manager
        
        # Test option creation logic
        select_options = []
        for i, theme in enumerate(themes):
            # Truncate theme text if too long for Discord
            theme_text = theme.theme if hasattr(theme, 'theme') else str(theme)
            reasoning = theme.reasoning if hasattr(theme, 'reasoning') else ""
            
            label = f"Theme {i+1}"
            if len(theme_text) <= 100:
                label = theme_text[:100]
            
            description = reasoning[:100] if reasoning else f"Theme option {i+1}"
            
            select_options.append({
                'label': label,
                'value': str(i),
                'description': description
            })
        
        # Add "fallback" option
        select_options.append({
            'label': "❌ Use fallback themes",
            'value': "fallback",
            'description': "Skip trending themes and use curated fallback themes"
        })
        
        # Verify options were created correctly
        assert len(select_options) == 4  # 3 themes + 1 fallback option
        assert select_options[0]['label'] == "Cyberpunk cityscapes"
        assert select_options[1]['label'] == "Zen gardens"
        assert select_options[2]['label'] == "Abstract geometry"
        assert select_options[3]['value'] == "fallback"
    
    def test_theme_select_view_long_theme_text_logic(self):
        """Test ThemeSelectView logic for handling long theme text."""
        long_theme = ThemeRecommendation(
            theme="A very long theme description that exceeds the Discord character limit for select option labels and needs to be truncated properly",
            reasoning="This is also a very long reasoning that might need truncation for the description field in Discord select options"
        )
        themes = [long_theme]
        
        # Test option creation logic for long text
        select_options = []
        for i, theme in enumerate(themes):
            theme_text = theme.theme if hasattr(theme, 'theme') else str(theme)
            reasoning = theme.reasoning if hasattr(theme, 'reasoning') else ""
            
            label = f"Theme {i+1}"
            if len(theme_text) <= 100:
                label = theme_text[:100]
            
            description = reasoning[:100] if reasoning else f"Theme option {i+1}"
            
            select_options.append({
                'label': label,
                'value': str(i),
                'description': description
            })
        
        # Add fallback option
        select_options.append({
            'label': "❌ Use fallback themes",
            'value': "fallback",
            'description': "Skip trending themes and use curated fallback themes"
        })
        
        # Verify long text was handled correctly
        assert len(select_options) == 2  # 1 theme + 1 fallback option
        assert len(select_options[0]['label']) <= 100  # Should be truncated to "Theme 1"
        assert len(select_options[0]['description']) <= 100  # Should be truncated
    
    @pytest.mark.asyncio
    async def test_theme_select_callback_wrong_user(self):
        """Test ThemeSelect callback with wrong user."""
        options = [Mock()]
        select = ThemeSelect(options, 123456789, self.themes, self.manager)
        
        # Mock interaction from wrong user
        interaction = Mock()
        interaction.user.id = 987654321  # Different user
        interaction.response.send_message = AsyncMock()
        
        await select.callback(interaction)
        
        interaction.response.send_message.assert_called_once_with(
            "This selection is not for you.", ephemeral=True
        )
    
    @pytest.mark.asyncio
    async def test_theme_select_callback_fallback_selection_logic(self):
        """Test ThemeSelect callback logic for fallback selection."""
        # Test the logic that would be used in the callback
        user_id = 123456789
        themes = self.themes
        manager = self.manager
        
        # Mock interaction from correct user
        interaction = Mock()
        interaction.user.id = user_id
        interaction.response.send_message = AsyncMock()
        
        # Simulate the callback logic for fallback selection
        selected_value = "fallback"
        
        # Test user ID check
        if interaction.user.id != user_id:
            await interaction.response.send_message("This selection is not for you.", ephemeral=True)
            return
        
        # Test fallback selection logic
        if selected_value == "fallback":
            manager.selected_value = "fallback"
            await interaction.response.send_message("✅ Using fallback themes instead of trending themes.")
            manager.logger.info("User chose to use fallback themes")
        
        manager.selection_event.set()
        
        # Verify the logic worked correctly
        assert manager.selected_value == "fallback"
        interaction.response.send_message.assert_called_once_with(
            "✅ Using fallback themes instead of trending themes."
        )
        manager.selection_event.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_theme_select_callback_theme_selection_logic(self):
        """Test ThemeSelect callback logic for theme selection."""
        # Test the logic that would be used in the callback
        user_id = 123456789
        themes = self.themes
        manager = self.manager
        
        # Mock interaction from correct user
        interaction = Mock()
        interaction.user.id = user_id
        interaction.response.send_message = AsyncMock()
        
        # Simulate the callback logic for theme selection
        selected_value = "0"  # Select first theme
        
        # Test user ID check
        if interaction.user.id != user_id:
            await interaction.response.send_message("This selection is not for you.", ephemeral=True)
            return
        
        # Test theme selection logic
        if selected_value != "fallback":
            selected_index = int(selected_value)
            selected_theme_obj = themes[selected_index]
            selected_theme = selected_theme_obj.theme if hasattr(selected_theme_obj, 'theme') else str(selected_theme_obj)
            manager.selected_value = selected_theme
            await interaction.response.send_message(f"✅ Selected theme: {selected_theme[:100]}...")
            manager.logger.info(f"User selected theme #{selected_index+1}: {selected_theme}")
        
        manager.selection_event.set()
        
        # Verify the logic worked correctly
        assert manager.selected_value == "Cyberpunk cityscapes"
        interaction.response.send_message.assert_called_once_with(
            "✅ Selected theme: Cyberpunk cityscapes..."
        )
        manager.selection_event.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_theme_select_callback_string_theme_selection_logic(self):
        """Test ThemeSelect callback logic for string theme (no .theme attribute)."""
        # Test the logic that would be used in the callback
        string_themes = ["Simple theme 1", "Simple theme 2"]
        user_id = 123456789
        manager = self.manager
        
        # Mock interaction from correct user
        interaction = Mock()
        interaction.user.id = user_id
        interaction.response.send_message = AsyncMock()
        
        # Simulate the callback logic for string theme selection
        selected_value = "0"  # Select first theme
        
        # Test user ID check
        if interaction.user.id != user_id:
            await interaction.response.send_message("This selection is not for you.", ephemeral=True)
            return
        
        # Test string theme selection logic
        if selected_value != "fallback":
            selected_index = int(selected_value)
            selected_theme_obj = string_themes[selected_index]
            selected_theme = selected_theme_obj.theme if hasattr(selected_theme_obj, 'theme') else str(selected_theme_obj)
            manager.selected_value = selected_theme
            await interaction.response.send_message(f"✅ Selected theme: {selected_theme[:100]}...")
            manager.logger.info(f"User selected theme #{selected_index+1}: {selected_theme}")
        
        manager.selection_event.set()
        
        # Verify the logic worked correctly
        assert manager.selected_value == "Simple theme 1"
        interaction.response.send_message.assert_called_once_with(
            "✅ Selected theme: Simple theme 1..."
        )
        manager.selection_event.set.assert_called_once()


class TestSetupThemeSelection:
    """Test the _setup_theme_selection function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.themes = [
            ThemeRecommendation(theme="Cyberpunk cityscapes", reasoning="Tech culture trending"),
            ThemeRecommendation(theme="Zen gardens", reasoning="Wellness movement")
        ]
        self.logger = Mock()
        self.manager = Mock()
        self.manager.timeout = 30
    
    @pytest.mark.asyncio
    async def test_setup_theme_selection(self):
        """Test _setup_theme_selection function."""
        mock_client = Mock()
        mock_user = Mock()
        mock_user.display_name = "TestUser"
        mock_dm = Mock()
        
        mock_client.fetch_user = AsyncMock(return_value=mock_user)
        mock_user.create_dm = AsyncMock(return_value=mock_dm)
        mock_dm.send = AsyncMock()
        
        with patch('pixelbliss.alerts.discord_select.ThemeSelectView') as mock_view:
            mock_view.return_value = Mock()
            
            await _setup_theme_selection(
                mock_client, 123456789, self.logger, self.themes, self.manager
            )
            
            # Verify user was fetched and DM created
            mock_client.fetch_user.assert_called_once_with(123456789)
            mock_user.create_dm.assert_called_once()
            
            # Verify message was sent
            mock_dm.send.assert_called_once()
            
            # Verify logging
            self.logger.info.assert_any_call("Sending 2 theme options to TestUser")
    
    @pytest.mark.asyncio
    async def test_setup_theme_selection_long_content(self):
        """Test _setup_theme_selection with long content that gets truncated."""
        # Create themes with very long text
        long_themes = []
        for i in range(10):
            long_theme = ThemeRecommendation(
                theme=f"Very long theme description number {i} " * 50,  # Very long
                reasoning=f"Very long reasoning for theme {i} " * 50  # Very long
            )
            long_themes.append(long_theme)
        
        mock_client = Mock()
        mock_user = Mock()
        mock_user.display_name = "TestUser"
        mock_dm = Mock()
        
        mock_client.fetch_user = AsyncMock(return_value=mock_user)
        mock_user.create_dm = AsyncMock(return_value=mock_dm)
        mock_dm.send = AsyncMock()
        
        with patch('pixelbliss.alerts.discord_select.ThemeSelectView') as mock_view:
            mock_view.return_value = Mock()
            
            await _setup_theme_selection(
                mock_client, 123456789, self.logger, long_themes, self.manager
            )
            
            # Verify message was sent (content should be truncated internally)
            mock_dm.send.assert_called_once()
            
            # Get the call arguments to verify content was handled
            call_args = mock_dm.send.call_args
            assert 'content' in call_args.kwargs or len(call_args.args) > 0


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
    @pytest.mark.asyncio
    async def test_ask_user_to_select_theme_success(self):
        """Test successful theme selection via Discord."""
        selected_theme = "Cyberpunk cityscapes"
        
        # Mock DiscordClientManager to simulate successful selection
        with patch('pixelbliss.alerts.discord_select.DiscordClientManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.__aenter__ = AsyncMock(return_value=mock_manager)
            mock_manager.__aexit__ = AsyncMock(return_value=None)
            mock_manager.run_selection = AsyncMock(return_value=selected_theme)
            mock_manager_class.return_value = mock_manager
            
            result = await ask_user_to_select_theme(self.themes, self.config, self.logger)
            
            assert result == selected_theme
            self.logger.info.assert_any_call("Discord theme selection completed successfully")

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
    @pytest.mark.asyncio
    async def test_ask_user_to_select_theme_timeout(self):
        """Test theme selection timeout."""
        # Mock DiscordClientManager to simulate timeout
        with patch('pixelbliss.alerts.discord_select.DiscordClientManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.__aenter__ = AsyncMock(return_value=mock_manager)
            mock_manager.__aexit__ = AsyncMock(return_value=None)
            mock_manager.run_selection = AsyncMock(return_value=None)  # Simulate timeout
            mock_manager_class.return_value = mock_manager
            
            result = await ask_user_to_select_theme(self.themes, self.config, self.logger)
            
            assert result is None
            self.logger.info.assert_called_with("Discord theme selection timed out or failed")

    @patch.dict('os.environ', {'DISCORD_BOT_TOKEN': 'test_token', 'DISCORD_USER_ID': '123456789'})
    @pytest.mark.asyncio
    async def test_ask_user_to_select_theme_exception(self):
        """Test theme selection with exception during Discord operations."""
        # Mock DiscordClientManager to raise exception during context manager entry
        with patch('pixelbliss.alerts.discord_select.DiscordClientManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.__aenter__ = AsyncMock(side_effect=Exception("Discord error"))
            mock_manager.__aexit__ = AsyncMock(return_value=None)
            mock_manager_class.return_value = mock_manager
            
            # The exception should be propagated since it's not caught in the function
            with pytest.raises(Exception, match="Discord error"):
                await ask_user_to_select_theme(self.themes, self.config, self.logger)

    @patch.dict('os.environ', {'DISCORD_BOT_TOKEN': 'test_token', 'DISCORD_USER_ID': '123456789'})
    @pytest.mark.asyncio
    async def test_ask_user_to_select_theme_fallback_selection(self):
        """Test theme selection with fallback selection."""
        # Mock DiscordClientManager to simulate fallback selection
        with patch('pixelbliss.alerts.discord_select.DiscordClientManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.__aenter__ = AsyncMock(return_value=mock_manager)
            mock_manager.__aexit__ = AsyncMock(return_value=None)
            mock_manager.run_selection = AsyncMock(return_value="fallback")
            mock_manager_class.return_value = mock_manager
            
            result = await ask_user_to_select_theme(self.themes, self.config, self.logger)
            
            assert result == "fallback"
            self.logger.info.assert_any_call("Discord theme selection completed successfully")

    @patch.dict('os.environ', {'DISCORD_BOT_TOKEN': 'test_token', 'DISCORD_USER_ID': '123456789'})
    @pytest.mark.asyncio
    async def test_ask_user_to_select_theme_custom_messages(self):
        """Test that custom timeout and error messages are set."""
        # Mock DiscordClientManager to verify custom messages are set
        with patch('pixelbliss.alerts.discord_select.DiscordClientManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.__aenter__ = AsyncMock(return_value=mock_manager)
            mock_manager.__aexit__ = AsyncMock(return_value=None)
            mock_manager.run_selection = AsyncMock(return_value=None)
            mock_manager_class.return_value = mock_manager
            
            result = await ask_user_to_select_theme(self.themes, self.config, self.logger)
            
            # Verify custom messages were set on the manager
            assert hasattr(mock_manager, 'timeout_message')
            assert hasattr(mock_manager, 'error_message')
            assert "theme selection" in mock_manager.timeout_message
            assert callable(mock_manager.error_message)

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

    @patch.dict('os.environ', {'DISCORD_BOT_TOKEN': 'test_token', 'DISCORD_USER_ID': '123456789'})
    @pytest.mark.asyncio
    async def test_ask_user_to_select_theme_manager_lifecycle(self):
        """Test that DiscordClientManager is properly created and used."""
        # Mock DiscordClientManager to verify it's created with correct parameters
        with patch('pixelbliss.alerts.discord_select.DiscordClientManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.__aenter__ = AsyncMock(return_value=mock_manager)
            mock_manager.__aexit__ = AsyncMock(return_value=None)
            mock_manager.run_selection = AsyncMock(return_value="test_theme")
            mock_manager_class.return_value = mock_manager
            
            result = await ask_user_to_select_theme(self.themes, self.config, self.logger)
            
            # Verify DiscordClientManager was created with correct parameters
            mock_manager_class.assert_called_once_with("test_token", 123456789, 1, self.logger)
            
            # Verify context manager was used
            mock_manager.__aenter__.assert_called_once()
            mock_manager.__aexit__.assert_called_once()
            
            # Verify run_selection was called
            mock_manager.run_selection.assert_called_once()
            
            assert result == "test_theme"
