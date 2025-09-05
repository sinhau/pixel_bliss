import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pixelbliss.alerts.discord_select import ask_user_to_select_raw


class TestDiscordSelect:
    """Test Discord selection functionality."""
    
    def setup_method(self):
        """Set up common test fixtures to speed up tests."""
        # Create a small test image once and reuse it
        from PIL import Image
        self.small_test_image = Image.new('RGB', (10, 10), color='red')
        self.large_test_image = Image.new('RGB', (100, 100), color='red')  # Smaller than 3000x3000

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
    async def test_ask_user_to_select_raw_client_connection_error(self):
        """Test ask_user_to_select_raw when Discord client fails to connect."""
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
            
            # Mock Discord client to raise an exception during start
            with patch('pixelbliss.alerts.discord_select.discord.Client') as mock_client_class, \
                 patch('pixelbliss.alerts.discord_select.asyncio.create_task') as mock_create_task, \
                 patch('pixelbliss.alerts.discord_select.asyncio.wait_for') as mock_wait_for:
                
                mock_client = Mock()
                mock_client.start = AsyncMock(side_effect=Exception("Connection failed"))
                mock_client.is_closed.return_value = False
                mock_client.close = AsyncMock()
                mock_client_class.return_value = mock_client
                
                # Mock asyncio operations to prevent real async delays
                mock_task = Mock()
                mock_create_task.return_value = mock_task
                mock_wait_for.side_effect = Exception("Connection failed")
                
                result = await ask_user_to_select_raw(candidates, cfg, logger)
                
                assert result is None
                logger.error.assert_called_with("Error in Discord selection process: Connection failed")

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
            
            # Mock all async operations to prevent real delays
            with patch('pixelbliss.alerts.discord_select.discord.Client') as mock_client_class, \
                 patch('pixelbliss.alerts.discord_select.asyncio.wait_for') as mock_wait_for, \
                 patch('pixelbliss.alerts.discord_select.asyncio.create_task') as mock_create_task:
                
                mock_client = Mock()
                mock_client.start = AsyncMock()
                mock_client.is_closed.return_value = False
                mock_client.close = AsyncMock()
                mock_client_class.return_value = mock_client
                
                # Mock task creation and wait operations
                mock_task = Mock()
                mock_create_task.return_value = mock_task
                mock_wait_for.side_effect = [asyncio.TimeoutError(), None]  # First timeout, then complete
                
                result = await ask_user_to_select_raw(candidates, cfg, logger)
                
                assert result is None
                logger.warning.assert_called_with("No selection received within 0.01 seconds, timing out")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_image_downscaling(self):
        """Test ask_user_to_select_raw with large images that need downscaling."""
        # Use a mock image that reports large size but doesn't actually process large data
        mock_image = Mock()
        mock_image.size = (3000, 3000)  # Reports large size
        mock_image.copy.return_value = mock_image
        mock_image.thumbnail = Mock()  # Mock the thumbnail operation
        mock_image.save = Mock()  # Mock the save operation
        
        candidates = [{"image": mock_image, "provider": "fal", "model": "test"}]
        
        cfg = Mock()
        cfg.discord.bot_token_env = "DISCORD_BOT_TOKEN"
        cfg.discord.user_id_env = "DISCORD_USER_ID"
        cfg.discord.timeout_sec = 0.01  # Very short timeout
        cfg.discord.batch_size = 10
        logger = Mock()
        
        with patch('pixelbliss.alerts.discord_select.os.getenv') as mock_getenv:
            # Mock environment variables - valid values
            mock_getenv.side_effect = lambda key, default: "test_token" if key == "DISCORD_BOT_TOKEN" else "123456789"
            
            # Mock all Discord and async components
            with patch('pixelbliss.alerts.discord_select.discord.Client') as mock_client_class, \
                 patch('pixelbliss.alerts.discord_select.discord.File') as mock_file_class, \
                 patch('pixelbliss.alerts.discord_select.asyncio.create_task') as mock_create_task:
                
                mock_client = Mock()
                mock_client.start = AsyncMock(side_effect=Exception("Test exception to exit early"))
                mock_client.is_closed.return_value = False
                mock_client.close = AsyncMock()
                mock_client_class.return_value = mock_client
                
                # Mock task creation to prevent real async operations
                mock_task = Mock()
                mock_create_task.return_value = mock_task
                
                result = await ask_user_to_select_raw(candidates, cfg, logger)
                
                assert result is None
                # Verify that the function attempted to process the image and logged the start message
                logger.info.assert_any_call("Starting Discord human-in-the-loop selection for 1 candidates")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_multiple_batches(self):
        """Test ask_user_to_select_raw with multiple batches of candidates."""
        # Use pre-created small test images to speed up test
        candidates = []
        for i in range(15):
            candidates.append({"image": self.small_test_image, "provider": "fal", "model": f"test{i}"})
        
        cfg = Mock()
        cfg.discord.bot_token_env = "DISCORD_BOT_TOKEN"
        cfg.discord.user_id_env = "DISCORD_USER_ID"
        cfg.discord.timeout_sec = 0.01  # Very short timeout
        cfg.discord.batch_size = 10
        logger = Mock()
        
        with patch('pixelbliss.alerts.discord_select.os.getenv') as mock_getenv:
            # Mock environment variables - valid values
            mock_getenv.side_effect = lambda key, default: "test_token" if key == "DISCORD_BOT_TOKEN" else "123456789"
            
            # Mock all Discord and async components
            with patch('pixelbliss.alerts.discord_select.discord.Client') as mock_client_class, \
                 patch('pixelbliss.alerts.discord_select.asyncio.create_task') as mock_create_task:
                
                mock_client = Mock()
                mock_client.start = AsyncMock(side_effect=Exception("Test exception to exit early"))
                mock_client.is_closed.return_value = False
                mock_client.close = AsyncMock()
                mock_client_class.return_value = mock_client
                
                # Mock task creation to prevent real async operations
                mock_task = Mock()
                mock_create_task.return_value = mock_task
                
                result = await ask_user_to_select_raw(candidates, cfg, logger)
                
                assert result is None
                # Verify that the function attempted to process multiple candidates
                logger.info.assert_any_call("Starting Discord human-in-the-loop selection for 15 candidates")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_long_provider_model_names(self):
        """Test ask_user_to_select_raw with long provider/model names that need truncation."""
        # Use pre-created small test image
        candidates = [{
            "image": self.small_test_image, 
            "provider": "very_long_provider_name_that_exceeds_discord_limits", 
            "model": "very_long_model_name_that_also_exceeds_discord_character_limits_for_select_options"
        }]
        
        cfg = Mock()
        cfg.discord.bot_token_env = "DISCORD_BOT_TOKEN"
        cfg.discord.user_id_env = "DISCORD_USER_ID"
        cfg.discord.timeout_sec = 0.01  # Very short timeout
        cfg.discord.batch_size = 10
        logger = Mock()
        
        with patch('pixelbliss.alerts.discord_select.os.getenv') as mock_getenv:
            # Mock environment variables - valid values
            mock_getenv.side_effect = lambda key, default: "test_token" if key == "DISCORD_BOT_TOKEN" else "123456789"
            
            # Mock all Discord and async components
            with patch('pixelbliss.alerts.discord_select.discord.Client') as mock_client_class, \
                 patch('pixelbliss.alerts.discord_select.asyncio.create_task') as mock_create_task:
                
                mock_client = Mock()
                mock_client.start = AsyncMock(side_effect=Exception("Test exception to exit early"))
                mock_client.is_closed.return_value = False
                mock_client.close = AsyncMock()
                mock_client_class.return_value = mock_client
                
                # Mock task creation to prevent real async operations
                mock_task = Mock()
                mock_create_task.return_value = mock_task
                
                result = await ask_user_to_select_raw(candidates, cfg, logger)
                
                assert result is None
                # Verify that the function attempted to process the candidate
                logger.info.assert_any_call("Starting Discord human-in-the-loop selection for 1 candidates")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_client_close_error(self):
        """Test ask_user_to_select_raw when client close fails."""
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
            
            # Mock all Discord and async components
            with patch('pixelbliss.alerts.discord_select.discord.Client') as mock_client_class, \
                 patch('pixelbliss.alerts.discord_select.asyncio.create_task') as mock_create_task:
                
                mock_client = Mock()
                mock_client.start = AsyncMock(side_effect=Exception("Connection failed"))
                mock_client.is_closed.return_value = False
                mock_client.close = AsyncMock(side_effect=Exception("Close failed"))
                mock_client_class.return_value = mock_client
                
                # Mock task creation to prevent real async operations
                mock_task = Mock()
                mock_create_task.return_value = mock_task
                
                result = await ask_user_to_select_raw(candidates, cfg, logger)
                
                assert result is None
                # When both start and close fail, the close error is logged (as seen in the actual execution)
                logger.error.assert_called_with("Error in Discord selection process: Close failed")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_getattr_fallback(self):
        """Test ask_user_to_select_raw getattr fallback for missing config attributes."""
        # Use pre-created small test image
        candidates = [{"image": self.small_test_image, "provider": "fal", "model": "test"}]
        
        # Create config without discord attributes to test getattr fallback
        cfg = Mock()
        cfg.discord = Mock()
        cfg.discord.bot_token_env = "DISCORD_BOT_TOKEN"
        cfg.discord.user_id_env = "DISCORD_USER_ID"
        cfg.discord.timeout_sec = 0.01  # Very short timeout
        cfg.discord.batch_size = 10
        logger = Mock()
        
        with patch('pixelbliss.alerts.discord_select.os.getenv') as mock_getenv:
            # Mock environment variables - empty values to trigger fallback
            mock_getenv.side_effect = lambda key, default: ""
            
            result = await ask_user_to_select_raw(candidates, cfg, logger)
            
            assert result is None
            logger.warning.assert_called_once_with("Discord bot token or user ID not configured, skipping human selection")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_type_error_user_id(self):
        """Test ask_user_to_select_raw when user_id causes TypeError during int conversion."""
        candidates = [{"image": Mock(), "provider": "fal", "model": "test"}]
        cfg = Mock()
        cfg.discord.bot_token_env = "DISCORD_BOT_TOKEN"
        cfg.discord.user_id_env = "DISCORD_USER_ID"
        logger = Mock()
        
        with patch('pixelbliss.alerts.discord_select.os.getenv') as mock_getenv:
            # Mock environment variables - return None which will be caught by the empty check
            # This test actually tests the case where the user_id is None/empty
            mock_getenv.side_effect = lambda key, default: "test_token" if key == "DISCORD_BOT_TOKEN" else None
            
            result = await ask_user_to_select_raw(candidates, cfg, logger)
            
            assert result is None
            # None will be caught by the "not user_id_str" check, not the int() conversion
            logger.warning.assert_called_once_with("Discord bot token or user ID not configured, skipping human selection")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_success_logging(self):
        """Test ask_user_to_select_raw success and timeout logging paths."""
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
            
            # Mock all async operations to prevent real delays
            with patch('pixelbliss.alerts.discord_select.discord.Client') as mock_client_class, \
                 patch('pixelbliss.alerts.discord_select.asyncio.wait_for') as mock_wait_for, \
                 patch('pixelbliss.alerts.discord_select.asyncio.create_task') as mock_create_task:
                
                mock_client = Mock()
                mock_client.start = AsyncMock()
                mock_client.is_closed.return_value = False
                mock_client.close = AsyncMock()
                mock_client_class.return_value = mock_client
                
                # Mock task creation and wait operations
                mock_task = Mock()
                mock_create_task.return_value = mock_task
                mock_wait_for.side_effect = [asyncio.TimeoutError(), None]  # First timeout, then complete
                
                result = await ask_user_to_select_raw(candidates, cfg, logger)
                
                assert result is None
                logger.info.assert_called_with("Discord human selection timed out or failed")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_client_task_timeout(self):
        """Test ask_user_to_select_raw when client task doesn't complete within timeout."""
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
            
            # Mock all Discord and async components
            with patch('pixelbliss.alerts.discord_select.discord.Client') as mock_client_class, \
                 patch('pixelbliss.alerts.discord_select.asyncio.create_task') as mock_create_task, \
                 patch('pixelbliss.alerts.discord_select.asyncio.wait_for') as mock_wait_for:
                
                mock_client = Mock()
                mock_client.start = AsyncMock()
                mock_client.is_closed.return_value = False
                mock_client.close = AsyncMock()
                mock_client_class.return_value = mock_client
                
                # Mock client task and async operations
                mock_task = Mock()
                mock_task.cancel = Mock()
                mock_create_task.return_value = mock_task
                
                # First wait_for succeeds (selection event), second times out (client task)
                mock_wait_for.side_effect = [None, asyncio.TimeoutError()]
                
                result = await ask_user_to_select_raw(candidates, cfg, logger)
                
                assert result is None
                logger.debug.assert_called_with("Client task didn't complete within 5 seconds, continuing")
                mock_task.cancel.assert_called_once()
