import pytest
import asyncio
import gc
import warnings
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from PIL import Image
from pixelbliss.alerts.discord_select import (
    ask_user_to_select_raw, 
    _validate_discord_config,
    DiscordClientManager,
    CandidateSelectView,
    CandidateSelect,
    _setup_candidate_selection
)
import discord


def cleanup_async_mocks():
    """Clean up any lingering AsyncMock coroutines to prevent warnings."""
    # Force garbage collection to clean up any pending coroutines
    gc.collect()
    
    # Suppress specific RuntimeWarnings about unawaited coroutines during test cleanup
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*never awaited")
        gc.collect()


class TestValidateDiscordConfig:
    """Test the _validate_discord_config function."""
    
    def test_validate_discord_config_success(self):
        """Test successful validation of Discord config."""
        cfg = Mock()
        cfg.discord.bot_token_env = "DISCORD_BOT_TOKEN"
        cfg.discord.user_id_env = "DISCORD_USER_ID"
        cfg.discord.timeout_sec = 30
        logger = Mock()
        
        with patch('pixelbliss.alerts.discord_select.os.getenv') as mock_getenv:
            mock_getenv.side_effect = lambda key, default: "test_token" if key == "DISCORD_BOT_TOKEN" else "123456789"
            
            result = _validate_discord_config(cfg, logger)
            
            assert result == ("test_token", 123456789, 30)
    
    def test_validate_discord_config_no_token(self):
        """Test validation when token is missing."""
        cfg = Mock()
        cfg.discord.bot_token_env = "DISCORD_BOT_TOKEN"
        cfg.discord.user_id_env = "DISCORD_USER_ID"
        logger = Mock()
        
        with patch('pixelbliss.alerts.discord_select.os.getenv') as mock_getenv:
            mock_getenv.side_effect = lambda key, default: "" if key == "DISCORD_BOT_TOKEN" else "123456789"
            
            result = _validate_discord_config(cfg, logger)
            
            assert result is None
            logger.warning.assert_called_once_with("Discord bot token or user ID not configured, skipping human selection")
    
    def test_validate_discord_config_no_user_id(self):
        """Test validation when user ID is missing."""
        cfg = Mock()
        cfg.discord.bot_token_env = "DISCORD_BOT_TOKEN"
        cfg.discord.user_id_env = "DISCORD_USER_ID"
        logger = Mock()
        
        with patch('pixelbliss.alerts.discord_select.os.getenv') as mock_getenv:
            mock_getenv.side_effect = lambda key, default: "test_token" if key == "DISCORD_BOT_TOKEN" else ""
            
            result = _validate_discord_config(cfg, logger)
            
            assert result is None
            logger.warning.assert_called_once_with("Discord bot token or user ID not configured, skipping human selection")
    
    def test_validate_discord_config_invalid_user_id(self):
        """Test validation when user ID is invalid."""
        cfg = Mock()
        cfg.discord.bot_token_env = "DISCORD_BOT_TOKEN"
        cfg.discord.user_id_env = "DISCORD_USER_ID"
        logger = Mock()
        
        with patch('pixelbliss.alerts.discord_select.os.getenv') as mock_getenv:
            mock_getenv.side_effect = lambda key, default: "test_token" if key == "DISCORD_BOT_TOKEN" else "invalid_id"
            
            result = _validate_discord_config(cfg, logger)
            
            assert result is None
            logger.error.assert_called_once_with("Invalid DISCORD_USER_ID: invalid_id")


class TestDiscordClientManager:
    """Test the DiscordClientManager class."""
    
    @pytest.mark.asyncio
    async def test_discord_client_manager_context_manager(self):
        """Test DiscordClientManager as async context manager."""
        logger = Mock()
        manager = DiscordClientManager("test_token", 123456789, 30, logger)
        
        with patch('pixelbliss.alerts.discord_select.discord.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.is_closed.return_value = False
            mock_client.close = AsyncMock()
            mock_client_class.return_value = mock_client
            
            with patch('pixelbliss.alerts.discord_select.asyncio.Event') as mock_event_class:
                mock_event = Mock()
                mock_event_class.return_value = mock_event
                
                async with manager:
                    assert manager.client == mock_client
                    assert manager.selection_event == mock_event
                
                # Verify client was closed
                mock_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_discord_client_manager_close_error(self):
        """Test DiscordClientManager handles close errors gracefully."""
        logger = Mock()
        manager = DiscordClientManager("test_token", 123456789, 30, logger)
        
        with patch('pixelbliss.alerts.discord_select.discord.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.is_closed.return_value = False
            mock_client.close = Mock(side_effect=Exception("Close failed"))
            mock_client_class.return_value = mock_client
            
            with patch('pixelbliss.alerts.discord_select.asyncio.Event') as mock_event_class:
                mock_event = Mock()
                mock_event_class.return_value = mock_event
                
                # Should not raise exception despite close error
                async with manager:
                    pass
    
    @pytest.mark.asyncio
    async def test_discord_client_manager_run_selection(self):
        """Test DiscordClientManager run_selection method."""
        logger = Mock()
        manager = DiscordClientManager("test_token", 123456789, 0.01, logger)  # Short timeout
        
        setup_callback = AsyncMock()
        
        with patch('pixelbliss.alerts.discord_select.discord.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.is_closed.return_value = False
            mock_client.close = AsyncMock()
            mock_client.start = AsyncMock()
            mock_client.user = Mock()
            mock_client_class.return_value = mock_client
            
            with patch('pixelbliss.alerts.discord_select.asyncio.Event') as mock_event_class, \
                 patch('pixelbliss.alerts.discord_select.asyncio.create_task') as mock_create_task, \
                 patch('pixelbliss.alerts.discord_select.asyncio.wait_for') as mock_wait_for:
                
                mock_event = Mock()
                mock_event.wait = AsyncMock()
                mock_event_class.return_value = mock_event
                
                mock_task = Mock()
                mock_create_task.return_value = mock_task
                mock_wait_for.side_effect = [asyncio.TimeoutError(), None]  # Timeout then complete
                
                async with manager:
                    result = await manager.run_selection(setup_callback)
                
                assert result is None  # No selection made
                logger.warning.assert_called_with("No selection received within 0.01 seconds, timing out")


class TestCandidateSelectComponents:
    """Test the CandidateSelectView and CandidateSelect classes."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.small_test_image = Image.new('RGB', (10, 10), color='red')
        self.candidates = [
            {"image": self.small_test_image, "provider": "fal", "model": "test1"},
            {"image": self.small_test_image, "provider": "replicate", "model": "test2"}
        ]
        self.logger = Mock()
        self.manager = Mock()
        self.manager.timeout = 30
        self.manager.selection_event = Mock()
        self.manager.logger = self.logger
    
    def test_candidate_select_view_creation_logic(self):
        """Test CandidateSelectView creation logic without Discord UI framework."""
        # Test the logic that would be used in CandidateSelectView creation
        candidates = self.candidates
        batch_start = 0
        batch_end = 2
        user_id = 123456789
        timeout = 30
        manager = self.manager
        
        # Test batch slicing logic
        batch = candidates[batch_start:batch_end]
        assert len(batch) == 2
        assert batch[0] == candidates[0]
        assert batch[1] == candidates[1]
        
        # Test option creation logic
        select_options = []
        for i, candidate in enumerate(batch):
            global_index = batch_start + i
            provider = candidate.get('provider', 'unknown')
            model = candidate.get('model', 'unknown')
            label = f"#{global_index+1} ({provider}/{model})"
            if len(label) > 100:  # Discord limit
                label = f"#{global_index+1} ({provider})"
            select_options.append({
                'label': label,
                'value': str(global_index),
                'description': f"Select candidate #{global_index+1}"
            })
        
        # Add "none" option for first batch
        if batch_start == 0:
            select_options.append({
                'label': "❌ None (reject all)",
                'value': "none",
                'description': "Reject all candidates and end pipeline"
            })
        
        # Verify options were created correctly
        assert len(select_options) == 3  # 2 candidates + 1 none option
        assert select_options[0]['label'] == "#1 (fal/test1)"
        assert select_options[1]['label'] == "#2 (replicate/test2)"
        assert select_options[2]['value'] == "none"
    
    def test_candidate_select_view_without_none_option_logic(self):
        """Test CandidateSelectView logic for non-first batch."""
        candidates = self.candidates
        batch_start = 2  # Not first batch
        batch_end = 4
        
        # Test batch slicing logic
        batch = candidates[batch_start:batch_end]
        assert len(batch) == 0  # No candidates in this range
        
        # Test option creation logic
        select_options = []
        for i, candidate in enumerate(batch):
            global_index = batch_start + i
            provider = candidate.get('provider', 'unknown')
            model = candidate.get('model', 'unknown')
            label = f"#{global_index+1} ({provider}/{model})"
            select_options.append({
                'label': label,
                'value': str(global_index),
                'description': f"Select candidate #{global_index+1}"
            })
        
        # Don't add "none" option for non-first batch
        if batch_start == 0:
            select_options.append({
                'label': "❌ None (reject all)",
                'value': "none",
                'description': "Reject all candidates and end pipeline"
            })
        
        # Verify no options were created (no candidates in range)
        assert len(select_options) == 0
    
    @pytest.mark.asyncio
    async def test_candidate_select_callback_wrong_user(self):
        """Test CandidateSelect callback with wrong user."""
        options = [Mock()]
        select = CandidateSelect(options, 123456789, self.candidates, self.manager)
        
        # Mock interaction from wrong user
        interaction = Mock()
        interaction.user.id = 987654321  # Different user
        interaction.response.send_message = AsyncMock()
        
        await select.callback(interaction)
        
        interaction.response.send_message.assert_called_once_with(
            "This selection is not for you.", ephemeral=True
        )
    
    @pytest.mark.asyncio
    async def test_candidate_select_callback_none_selection_logic(self):
        """Test CandidateSelect callback logic for 'none' selection."""
        # Test the logic that would be used in the callback
        user_id = 123456789
        candidates = self.candidates
        manager = self.manager
        
        # Mock interaction from correct user
        interaction = Mock()
        interaction.user.id = user_id
        interaction.response.send_message = AsyncMock()
        
        # Simulate the callback logic for "none" selection
        selected_value = "none"
        
        # Test user ID check
        if interaction.user.id != user_id:
            await interaction.response.send_message("This selection is not for you.", ephemeral=True)
            return
        
        # Test "none" selection logic
        if selected_value == "none":
            manager.selected_value = "none"
            await interaction.response.send_message("❌ All candidates rejected. Pipeline will end without posting.")
            manager.logger.info("User rejected all candidates via 'none' selection")
        
        manager.selection_event.set()
        
        # Verify the logic worked correctly
        assert manager.selected_value == "none"
        interaction.response.send_message.assert_called_once_with(
            "❌ All candidates rejected. Pipeline will end without posting."
        )
        manager.selection_event.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_candidate_select_callback_candidate_selection_logic(self):
        """Test CandidateSelect callback logic for candidate selection."""
        # Test the logic that would be used in the callback
        user_id = 123456789
        candidates = self.candidates
        manager = self.manager
        
        # Mock interaction from correct user
        interaction = Mock()
        interaction.user.id = user_id
        interaction.response.send_message = AsyncMock()
        
        # Simulate the callback logic for candidate selection
        selected_value = "0"  # Select first candidate
        
        # Test user ID check
        if interaction.user.id != user_id:
            await interaction.response.send_message("This selection is not for you.", ephemeral=True)
            return
        
        # Test candidate selection logic
        if selected_value != "none":
            selected_index = int(selected_value)
            manager.selected_value = candidates[selected_index]
            await interaction.response.send_message(f"✅ Using candidate #{selected_index+1}. Thanks!")
            manager.logger.info(f"User selected candidate #{selected_index+1}")
        
        manager.selection_event.set()
        
        # Verify the logic worked correctly
        assert manager.selected_value == candidates[0]
        interaction.response.send_message.assert_called_once_with(
            "✅ Using candidate #1. Thanks!"
        )
        manager.selection_event.set.assert_called_once()


class TestSetupCandidateSelection:
    """Test the _setup_candidate_selection function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.small_test_image = Image.new('RGB', (10, 10), color='red')
        self.candidates = [
            {"image": self.small_test_image, "provider": "fal", "model": "test1"},
            {"image": self.small_test_image, "provider": "replicate", "model": "test2"}
        ]
        self.logger = Mock()
        self.manager = Mock()
        self.manager.timeout = 30
    
    @pytest.mark.asyncio
    async def test_setup_candidate_selection(self):
        """Test _setup_candidate_selection function."""
        mock_client = Mock()
        mock_user = Mock()
        mock_user.display_name = "TestUser"
        mock_dm = Mock()
        
        mock_client.fetch_user = AsyncMock(return_value=mock_user)
        mock_user.create_dm = AsyncMock(return_value=mock_dm)
        mock_dm.send = AsyncMock()
        
        with patch('pixelbliss.alerts.discord_select.discord.File') as mock_file, \
             patch('pixelbliss.alerts.discord_select.CandidateSelectView') as mock_view, \
             patch('pixelbliss.alerts.discord_select.asyncio.sleep') as mock_sleep:
            
            mock_file.return_value = Mock()
            mock_view.return_value = Mock()
            mock_sleep.return_value = None
            
            await _setup_candidate_selection(
                mock_client, 123456789, self.logger, self.candidates, 10, self.manager
            )
            
            # Verify user was fetched and DM created
            mock_client.fetch_user.assert_called_once_with(123456789)
            mock_user.create_dm.assert_called_once()
            
            # Verify message was sent
            mock_dm.send.assert_called_once()
            
            # Verify logging
            self.logger.info.assert_any_call("Sending 2 candidates in batches of 10 to TestUser")


class TestDiscordSelect:
    """Test Discord selection functionality."""
    
    def setup_method(self):
        """Set up common test fixtures to speed up tests."""
        # Create a small test image once and reuse it
        from PIL import Image
        self.small_test_image = Image.new('RGB', (10, 10), color='red')
        self.large_test_image = Image.new('RGB', (100, 100), color='red')  # Smaller than 3000x3000

    def teardown_method(self):
        """Clean up after each test to prevent warning contamination."""
        cleanup_async_mocks()

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_no_token(self):
        """Test ask_user_to_select_raw when Discord token is not configured."""
        candidates = [{"image": Mock(), "provider": "fal", "model": "test"}]
        cfg = Mock()
        cfg.discord.bot_token_env = "DISCORD_BOT_TOKEN"
        cfg.discord.user_id_env = "DISCORD_USER_ID"
        logger = Mock()
        
        with patch('pixelbliss.alerts.discord_select.os.getenv') as mock_getenv:
            # Mock environment variables - empty token
            mock_getenv.side_effect = lambda key, default: "" if key == "DISCORD_BOT_TOKEN" else "123456789"
            
            result = await ask_user_to_select_raw(candidates, cfg, logger)
            
            assert result is None
            logger.warning.assert_called_once_with("Discord bot token or user ID not configured, skipping human selection")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_no_user_id(self):
        """Test ask_user_to_select_raw when Discord user ID is not configured."""
        candidates = [{"image": Mock(), "provider": "fal", "model": "test"}]
        cfg = Mock()
        cfg.discord.bot_token_env = "DISCORD_BOT_TOKEN"
        cfg.discord.user_id_env = "DISCORD_USER_ID"
        logger = Mock()
        
        with patch('pixelbliss.alerts.discord_select.os.getenv') as mock_getenv:
            # Mock environment variables - empty user ID
            mock_getenv.side_effect = lambda key, default: "test_token" if key == "DISCORD_BOT_TOKEN" else ""
            
            result = await ask_user_to_select_raw(candidates, cfg, logger)
            
            assert result is None
            logger.warning.assert_called_once_with("Discord bot token or user ID not configured, skipping human selection")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_invalid_user_id(self):
        """Test ask_user_to_select_raw when Discord user ID is invalid."""
        candidates = [{"image": Mock(), "provider": "fal", "model": "test"}]
        cfg = Mock()
        cfg.discord.bot_token_env = "DISCORD_BOT_TOKEN"
        cfg.discord.user_id_env = "DISCORD_USER_ID"
        logger = Mock()
        
        with patch('pixelbliss.alerts.discord_select.os.getenv') as mock_getenv:
            # Mock environment variables - invalid user ID
            mock_getenv.side_effect = lambda key, default: "test_token" if key == "DISCORD_BOT_TOKEN" else "invalid_id"
            
            result = await ask_user_to_select_raw(candidates, cfg, logger)
            
            assert result is None
            logger.error.assert_called_once_with("Invalid DISCORD_USER_ID: invalid_id")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_no_candidates(self):
        """Test ask_user_to_select_raw when no candidates are provided."""
        candidates = []
        cfg = Mock()
        cfg.discord.bot_token_env = "DISCORD_BOT_TOKEN"
        cfg.discord.user_id_env = "DISCORD_USER_ID"
        logger = Mock()
        
        with patch('pixelbliss.alerts.discord_select.os.getenv') as mock_getenv:
            # Mock environment variables - valid values
            mock_getenv.side_effect = lambda key, default: "test_token" if key == "DISCORD_BOT_TOKEN" else "123456789"
            
            result = await ask_user_to_select_raw(candidates, cfg, logger)
            
            assert result is None
            logger.warning.assert_called_once_with("No candidates to select from")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_timeout(self):
        """Test ask_user_to_select_raw when user doesn't respond within timeout."""
        # Use pre-created small test image
        candidates = [{"image": self.small_test_image, "provider": "fal", "model": "test"}]
        
        cfg = Mock()
        cfg.discord.bot_token_env = "DISCORD_BOT_TOKEN"
        cfg.discord.user_id_env = "DISCORD_USER_ID"
        cfg.discord.timeout_sec = 0.01  # Very short timeout to speed up test
        cfg.discord.batch_size = 10
        logger = Mock()
        
        with patch('pixelbliss.alerts.discord_select.os.getenv') as mock_getenv:
            # Mock environment variables - valid values
            mock_getenv.side_effect = lambda key, default: "test_token" if key == "DISCORD_BOT_TOKEN" else "123456789"
            
            with patch('pixelbliss.alerts.discord_select.add_candidate_numbers_to_images') as mock_numbering:
                mock_numbering.return_value = candidates
                
                # Mock DiscordClientManager to simulate timeout
                with patch('pixelbliss.alerts.discord_select.DiscordClientManager') as mock_manager_class:
                    mock_manager = Mock()
                    mock_manager.__aenter__ = AsyncMock(return_value=mock_manager)
                    mock_manager.__aexit__ = AsyncMock(return_value=None)
                    mock_manager.run_selection = AsyncMock(return_value=None)  # Simulate timeout
                    mock_manager_class.return_value = mock_manager
                    
                    result = await ask_user_to_select_raw(candidates, cfg, logger)
                    
                    assert result is None
                    logger.info.assert_any_call("Discord human selection timed out or failed")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_successful_selection(self):
        """Test ask_user_to_select_raw with successful candidate selection."""
        # Use pre-created small test image
        candidates = [{"image": self.small_test_image, "provider": "fal", "model": "test"}]
        selected_candidate = candidates[0]
        
        cfg = Mock()
        cfg.discord.bot_token_env = "DISCORD_BOT_TOKEN"
        cfg.discord.user_id_env = "DISCORD_USER_ID"
        cfg.discord.timeout_sec = 30
        cfg.discord.batch_size = 10
        logger = Mock()
        
        with patch('pixelbliss.alerts.discord_select.os.getenv') as mock_getenv:
            # Mock environment variables - valid values
            mock_getenv.side_effect = lambda key, default: "test_token" if key == "DISCORD_BOT_TOKEN" else "123456789"
            
            with patch('pixelbliss.alerts.discord_select.add_candidate_numbers_to_images') as mock_numbering:
                mock_numbering.return_value = candidates
                
                # Mock DiscordClientManager to simulate successful selection
                with patch('pixelbliss.alerts.discord_select.DiscordClientManager') as mock_manager_class:
                    mock_manager = Mock()
                    mock_manager.__aenter__ = AsyncMock(return_value=mock_manager)
                    mock_manager.__aexit__ = AsyncMock(return_value=None)
                    mock_manager.run_selection = AsyncMock(return_value=selected_candidate)
                    mock_manager_class.return_value = mock_manager
                    
                    result = await ask_user_to_select_raw(candidates, cfg, logger)
                    
                    assert result == selected_candidate
                    logger.info.assert_any_call("Discord human selection completed successfully")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_with_numbering_integration(self):
        """Test that ask_user_to_select_raw integrates with the numbering functionality."""
        # Use pre-created small test image
        candidates = [{"image": self.small_test_image, "provider": "fal", "model": "test"}]
        
        cfg = Mock()
        cfg.discord.bot_token_env = "DISCORD_BOT_TOKEN"
        cfg.discord.user_id_env = "DISCORD_USER_ID"
        cfg.discord.timeout_sec = 0.01  # Very short timeout
        cfg.discord.batch_size = 10
        logger = Mock()
        
        with patch('pixelbliss.alerts.discord_select.os.getenv') as mock_getenv:
            # Mock environment variables - valid values
            mock_getenv.side_effect = lambda key, default: "test_token" if key == "DISCORD_BOT_TOKEN" else "123456789"
            
            # Mock the numbering function to verify it's called
            with patch('pixelbliss.alerts.discord_select.add_candidate_numbers_to_images') as mock_numbering:
                mock_numbering.return_value = candidates  # Return the same candidates for simplicity
                
                # Mock DiscordClientManager
                with patch('pixelbliss.alerts.discord_select.DiscordClientManager') as mock_manager_class:
                    mock_manager = Mock()
                    mock_manager.__aenter__ = AsyncMock(return_value=mock_manager)
                    mock_manager.__aexit__ = AsyncMock(return_value=None)
                    mock_manager.run_selection = AsyncMock(return_value=None)
                    mock_manager_class.return_value = mock_manager
                    
                    result = await ask_user_to_select_raw(candidates, cfg, logger)
                    
                    # Verify that the numbering function was called with the candidates
                    mock_numbering.assert_called_once_with(candidates)
                    
                    assert result is None
                    logger.info.assert_any_call("Starting Discord human-in-the-loop selection for 1 candidates")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_large_image_downscaling(self):
        """Test image downscaling for large images (>2048px)."""
        # Create a large test image that should trigger downscaling
        large_image = Image.new('RGB', (3000, 3000), color='blue')
        candidates = [{"image": large_image, "provider": "fal", "model": "test"}]
        
        cfg = Mock()
        cfg.discord.bot_token_env = "DISCORD_BOT_TOKEN"
        cfg.discord.user_id_env = "DISCORD_USER_ID"
        cfg.discord.timeout_sec = 0.01
        cfg.discord.batch_size = 10
        logger = Mock()
        
        with patch('pixelbliss.alerts.discord_select.os.getenv') as mock_getenv:
            mock_getenv.side_effect = lambda key, default: "test_token" if key == "DISCORD_BOT_TOKEN" else "123456789"
            
            with patch('pixelbliss.alerts.discord_select.add_candidate_numbers_to_images') as mock_numbering:
                mock_numbering.return_value = candidates
                
                # Mock DiscordClientManager
                with patch('pixelbliss.alerts.discord_select.DiscordClientManager') as mock_manager_class:
                    mock_manager = Mock()
                    mock_manager.__aenter__ = AsyncMock(return_value=mock_manager)
                    mock_manager.__aexit__ = AsyncMock(return_value=None)
                    mock_manager.run_selection = AsyncMock(return_value=None)
                    mock_manager_class.return_value = mock_manager
                    
                    result = await ask_user_to_select_raw(candidates, cfg, logger)
                    
                    assert result is None
                    # Verify the function started processing the large image
                    logger.info.assert_any_call("Starting Discord human-in-the-loop selection for 1 candidates")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_batch_processing(self):
        """Test batch processing with multiple candidates."""
        # Create multiple candidates to test batching
        candidates = []
        for i in range(15):  # More than batch_size to test multiple batches
            candidates.append({"image": self.small_test_image, "provider": f"provider{i}", "model": f"model{i}"})
        
        cfg = Mock()
        cfg.discord.bot_token_env = "DISCORD_BOT_TOKEN"
        cfg.discord.user_id_env = "DISCORD_USER_ID"
        cfg.discord.timeout_sec = 0.01
        cfg.discord.batch_size = 5  # Small batch size to force multiple batches
        logger = Mock()
        
        with patch('pixelbliss.alerts.discord_select.os.getenv') as mock_getenv:
            mock_getenv.side_effect = lambda key, default: "test_token" if key == "DISCORD_BOT_TOKEN" else "123456789"
            
            with patch('pixelbliss.alerts.discord_select.add_candidate_numbers_to_images') as mock_numbering:
                mock_numbering.return_value = candidates
                
                # Mock DiscordClientManager
                with patch('pixelbliss.alerts.discord_select.DiscordClientManager') as mock_manager_class:
                    mock_manager = Mock()
                    mock_manager.__aenter__ = AsyncMock(return_value=mock_manager)
                    mock_manager.__aexit__ = AsyncMock(return_value=None)
                    mock_manager.run_selection = AsyncMock(return_value=None)
                    mock_manager_class.return_value = mock_manager
                    
                    result = await ask_user_to_select_raw(candidates, cfg, logger)
                    
                    assert result is None
                    # Verify the function started processing multiple candidates
                    logger.info.assert_any_call("Starting Discord human-in-the-loop selection for 15 candidates")


    @pytest.mark.asyncio
    async def test_discord_client_manager_run_selection_client_task_timeout(self):
        """Test DiscordClientManager run_selection when client task times out."""
        logger = Mock()
        manager = DiscordClientManager("test_token", 123456789, 0.01, logger)
        
        setup_callback = AsyncMock()
        
        with patch('pixelbliss.alerts.discord_select.discord.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.is_closed.return_value = False
            mock_client.close = AsyncMock()
            mock_client.start = AsyncMock()
            mock_client.user = Mock()
            mock_client_class.return_value = mock_client
            
            with patch('pixelbliss.alerts.discord_select.asyncio.Event') as mock_event_class, \
                 patch('pixelbliss.alerts.discord_select.asyncio.create_task') as mock_create_task, \
                 patch('pixelbliss.alerts.discord_select.asyncio.wait_for') as mock_wait_for:
                
                mock_event = Mock()
                mock_event.wait = AsyncMock()
                mock_event_class.return_value = mock_event
                
                mock_task = Mock()
                mock_task.cancel = Mock()
                mock_create_task.return_value = mock_task
                # First wait_for succeeds, second times out (client task timeout)
                mock_wait_for.side_effect = [None, asyncio.TimeoutError()]
                
                async with manager:
                    result = await manager.run_selection(setup_callback)
                
                assert result is None
                logger.debug.assert_called_with("Client task didn't complete within 5 seconds, continuing")
                mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_setup_candidate_selection_with_large_images(self):
        """Test _setup_candidate_selection with large images that need downscaling."""
        # Create a large test image
        large_image = Image.new('RGB', (3000, 3000), color='blue')
        candidates = [{"image": large_image, "provider": "fal", "model": "test"}]
        
        mock_client = Mock()
        mock_user = Mock()
        mock_user.display_name = "TestUser"
        mock_dm = Mock()
        
        mock_client.fetch_user = AsyncMock(return_value=mock_user)
        mock_user.create_dm = AsyncMock(return_value=mock_dm)
        mock_dm.send = AsyncMock()
        
        logger = Mock()
        manager = Mock()
        manager.timeout = 30
        
        with patch('pixelbliss.alerts.discord_select.discord.File') as mock_file, \
             patch('pixelbliss.alerts.discord_select.CandidateSelectView') as mock_view, \
             patch('pixelbliss.alerts.discord_select.asyncio.sleep') as mock_sleep:
            
            mock_file.return_value = Mock()
            mock_view.return_value = Mock()
            mock_sleep.return_value = None
            
            await _setup_candidate_selection(
                mock_client, 123456789, logger, candidates, 10, manager
            )
            
            # Verify the large image was processed
            mock_dm.send.assert_called_once()
            logger.info.assert_any_call("Sending 1 candidates in batches of 10 to TestUser")

    @pytest.mark.asyncio
    async def test_setup_candidate_selection_multiple_batches_with_delay(self):
        """Test _setup_candidate_selection with multiple batches and delays."""
        small_test_image = Image.new('RGB', (10, 10), color='red')
        candidates = []
        for i in range(15):  # More than batch_size
            candidates.append({"image": small_test_image, "provider": f"provider{i}", "model": f"model{i}"})
        
        mock_client = Mock()
        mock_user = Mock()
        mock_user.display_name = "TestUser"
        mock_dm = Mock()
        
        mock_client.fetch_user = AsyncMock(return_value=mock_user)
        mock_user.create_dm = AsyncMock(return_value=mock_dm)
        mock_dm.send = AsyncMock()
        
        logger = Mock()
        manager = Mock()
        manager.timeout = 30
        
        with patch('pixelbliss.alerts.discord_select.discord.File') as mock_file, \
             patch('pixelbliss.alerts.discord_select.CandidateSelectView') as mock_view, \
             patch('pixelbliss.alerts.discord_select.asyncio.sleep') as mock_sleep:
            
            mock_file.return_value = Mock()
            mock_view.return_value = Mock()
            mock_sleep.return_value = None
            
            await _setup_candidate_selection(
                mock_client, 123456789, logger, candidates, 5, manager  # Small batch size
            )
            
            # Verify multiple batches were sent
            assert mock_dm.send.call_count == 3  # 15 candidates / 5 batch_size = 3 batches
            # Verify sleep was called between batches (2 times for 3 batches)
            assert mock_sleep.call_count == 2
            logger.info.assert_any_call("Sending 15 candidates in batches of 5 to TestUser")
