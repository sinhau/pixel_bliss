import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pixelbliss.alerts.discord_select import ask_user_to_select_raw


class TestDiscordSelect:
    """Test Discord selection functionality."""

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_no_token(self):
        """Test ask_user_to_select_raw when Discord token is not configured."""
        candidates = [{"image": Mock(), "provider": "fal", "model": "test"}]
        cfg = Mock()
        cfg.discord.bot_token = ""
        cfg.discord.user_id = "123456789"
        logger = Mock()
        
        result = await ask_user_to_select_raw(candidates, cfg, logger)
        
        assert result is None
        logger.warning.assert_called_once_with("Discord bot token or user ID not configured, skipping human selection")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_no_user_id(self):
        """Test ask_user_to_select_raw when Discord user ID is not configured."""
        candidates = [{"image": Mock(), "provider": "fal", "model": "test"}]
        cfg = Mock()
        cfg.discord.bot_token = "test_token"
        cfg.discord.user_id = ""
        logger = Mock()
        
        result = await ask_user_to_select_raw(candidates, cfg, logger)
        
        assert result is None
        logger.warning.assert_called_once_with("Discord bot token or user ID not configured, skipping human selection")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_invalid_user_id(self):
        """Test ask_user_to_select_raw when Discord user ID is invalid."""
        candidates = [{"image": Mock(), "provider": "fal", "model": "test"}]
        cfg = Mock()
        cfg.discord.bot_token = "test_token"
        cfg.discord.user_id = "invalid_id"
        logger = Mock()
        
        result = await ask_user_to_select_raw(candidates, cfg, logger)
        
        assert result is None
        logger.error.assert_called_once_with("Invalid DISCORD_USER_ID: invalid_id")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_no_candidates(self):
        """Test ask_user_to_select_raw when no candidates are provided."""
        candidates = []
        cfg = Mock()
        cfg.discord.bot_token = "test_token"
        cfg.discord.user_id = "123456789"
        logger = Mock()
        
        result = await ask_user_to_select_raw(candidates, cfg, logger)
        
        assert result is None
        logger.warning.assert_called_once_with("No candidates to select from")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_client_connection_error(self):
        """Test ask_user_to_select_raw when Discord client fails to connect."""
        from PIL import Image
        
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='red')
        candidates = [{"image": test_image, "provider": "fal", "model": "test"}]
        
        cfg = Mock()
        cfg.discord.bot_token = "test_token"
        cfg.discord.user_id = "123456789"
        cfg.discord.timeout_sec = 1
        cfg.discord.batch_size = 10
        logger = Mock()
        
        # Mock Discord client to raise an exception during start
        with patch('pixelbliss.alerts.discord_select.discord.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.start = AsyncMock(side_effect=Exception("Connection failed"))
            mock_client.is_closed.return_value = False
            mock_client.close = AsyncMock()
            mock_client_class.return_value = mock_client
            
            result = await ask_user_to_select_raw(candidates, cfg, logger)
            
            assert result is None
            logger.error.assert_called_with("Error in Discord selection process: Connection failed")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_timeout(self):
        """Test ask_user_to_select_raw when user doesn't respond within timeout."""
        from PIL import Image
        
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='red')
        candidates = [{"image": test_image, "provider": "fal", "model": "test"}]
        
        cfg = Mock()
        cfg.discord.bot_token = "test_token"
        cfg.discord.user_id = "123456789"
        cfg.discord.timeout_sec = 0.1  # Very short timeout
        cfg.discord.batch_size = 10
        logger = Mock()
        
        # Mock Discord client
        with patch('pixelbliss.alerts.discord_select.discord.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.start = AsyncMock()
            mock_client.is_closed.return_value = False
            mock_client.close = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock the on_ready event to not trigger selection
            with patch('pixelbliss.alerts.discord_select.asyncio.Event') as mock_event_class:
                mock_event = Mock()
                mock_event.wait = AsyncMock(side_effect=asyncio.TimeoutError())
                mock_event_class.return_value = mock_event
                
                result = await ask_user_to_select_raw(candidates, cfg, logger)
                
                assert result is None
                logger.warning.assert_called_with("No selection received within 0.1 seconds, timing out")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_image_downscaling(self):
        """Test ask_user_to_select_raw with large images that need downscaling."""
        from PIL import Image
        
        # Create a large test image that will trigger downscaling
        test_image = Image.new('RGB', (3000, 3000), color='red')
        candidates = [{"image": test_image, "provider": "fal", "model": "test"}]
        
        cfg = Mock()
        cfg.discord.bot_token = "test_token"
        cfg.discord.user_id = "123456789"
        cfg.discord.timeout_sec = 1
        cfg.discord.batch_size = 10
        logger = Mock()
        
        # Mock Discord components
        with patch('pixelbliss.alerts.discord_select.discord.Client') as mock_client_class, \
             patch('pixelbliss.alerts.discord_select.discord.File') as mock_file_class:
            
            mock_client = Mock()
            mock_client.start = AsyncMock(side_effect=Exception("Test exception to exit early"))
            mock_client.is_closed.return_value = False
            mock_client.close = AsyncMock()
            mock_client_class.return_value = mock_client
            
            result = await ask_user_to_select_raw(candidates, cfg, logger)
            
            assert result is None
            # Verify that the function attempted to process the image
            logger.info.assert_called_with("Starting Discord human-in-the-loop selection for 1 candidates")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_multiple_batches(self):
        """Test ask_user_to_select_raw with multiple batches of candidates."""
        from PIL import Image
        
        # Create 15 test images to trigger multiple batches (batch_size = 10)
        candidates = []
        for i in range(15):
            test_image = Image.new('RGB', (100, 100), color='red')
            candidates.append({"image": test_image, "provider": "fal", "model": f"test{i}"})
        
        cfg = Mock()
        cfg.discord.bot_token = "test_token"
        cfg.discord.user_id = "123456789"
        cfg.discord.timeout_sec = 1
        cfg.discord.batch_size = 10
        logger = Mock()
        
        # Mock Discord components
        with patch('pixelbliss.alerts.discord_select.discord.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.start = AsyncMock(side_effect=Exception("Test exception to exit early"))
            mock_client.is_closed.return_value = False
            mock_client.close = AsyncMock()
            mock_client_class.return_value = mock_client
            
            result = await ask_user_to_select_raw(candidates, cfg, logger)
            
            assert result is None
            # Verify that the function attempted to process multiple candidates
            logger.info.assert_called_with("Starting Discord human-in-the-loop selection for 15 candidates")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_long_provider_model_names(self):
        """Test ask_user_to_select_raw with long provider/model names that need truncation."""
        from PIL import Image
        
        # Create a test image with very long provider/model names
        test_image = Image.new('RGB', (100, 100), color='red')
        candidates = [{
            "image": test_image, 
            "provider": "very_long_provider_name_that_exceeds_discord_limits", 
            "model": "very_long_model_name_that_also_exceeds_discord_character_limits_for_select_options"
        }]
        
        cfg = Mock()
        cfg.discord.bot_token = "test_token"
        cfg.discord.user_id = "123456789"
        cfg.discord.timeout_sec = 1
        cfg.discord.batch_size = 10
        logger = Mock()
        
        # Mock Discord components
        with patch('pixelbliss.alerts.discord_select.discord.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.start = AsyncMock(side_effect=Exception("Test exception to exit early"))
            mock_client.is_closed.return_value = False
            mock_client.close = AsyncMock()
            mock_client_class.return_value = mock_client
            
            result = await ask_user_to_select_raw(candidates, cfg, logger)
            
            assert result is None
            # Verify that the function attempted to process the candidate
            logger.info.assert_called_with("Starting Discord human-in-the-loop selection for 1 candidates")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_client_close_error(self):
        """Test ask_user_to_select_raw when client close fails."""
        from PIL import Image
        
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='red')
        candidates = [{"image": test_image, "provider": "fal", "model": "test"}]
        
        cfg = Mock()
        cfg.discord.bot_token = "test_token"
        cfg.discord.user_id = "123456789"
        cfg.discord.timeout_sec = 0.1
        cfg.discord.batch_size = 10
        logger = Mock()
        
        # Mock Discord client
        with patch('pixelbliss.alerts.discord_select.discord.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.start = AsyncMock(side_effect=Exception("Connection failed"))
            mock_client.is_closed.return_value = False
            mock_client.close = AsyncMock(side_effect=Exception("Close failed"))
            mock_client_class.return_value = mock_client
            
            result = await ask_user_to_select_raw(candidates, cfg, logger)
            
            assert result is None
            logger.error.assert_called_with("Error in Discord selection process: Connection failed")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_getattr_fallback(self):
        """Test ask_user_to_select_raw getattr fallback for missing config attributes."""
        from PIL import Image
        
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='red')
        candidates = [{"image": test_image, "provider": "fal", "model": "test"}]
        
        # Create config without discord attributes to test getattr fallback
        cfg = Mock()
        cfg.discord = Mock()
        # Remove attributes to test getattr fallback
        delattr(cfg.discord, 'bot_token')
        delattr(cfg.discord, 'user_id')
        cfg.discord.timeout_sec = 1
        cfg.discord.batch_size = 10
        logger = Mock()
        
        result = await ask_user_to_select_raw(candidates, cfg, logger)
        
        assert result is None
        logger.warning.assert_called_once_with("Discord bot token or user ID not configured, skipping human selection")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_type_error_user_id(self):
        """Test ask_user_to_select_raw when user_id causes TypeError during int conversion."""
        candidates = [{"image": Mock(), "provider": "fal", "model": "test"}]
        cfg = Mock()
        cfg.discord.bot_token = "test_token"
        cfg.discord.user_id = None  # This will cause TypeError in int()
        logger = Mock()
        
        result = await ask_user_to_select_raw(candidates, cfg, logger)
        
        assert result is None
        logger.error.assert_called_once_with("Invalid DISCORD_USER_ID: None")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_success_logging(self):
        """Test ask_user_to_select_raw success and timeout logging paths."""
        from PIL import Image
        
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='red')
        candidates = [{"image": test_image, "provider": "fal", "model": "test"}]
        
        cfg = Mock()
        cfg.discord.bot_token = "test_token"
        cfg.discord.user_id = "123456789"
        cfg.discord.timeout_sec = 0.1
        cfg.discord.batch_size = 10
        logger = Mock()
        
        # Test timeout path
        with patch('pixelbliss.alerts.discord_select.discord.Client') as mock_client_class, \
             patch('pixelbliss.alerts.discord_select.asyncio.wait_for') as mock_wait_for:
            
            mock_client = Mock()
            mock_client.start = AsyncMock()
            mock_client.is_closed.return_value = False
            mock_client.close = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock wait_for to raise TimeoutError
            mock_wait_for.side_effect = asyncio.TimeoutError()
            
            result = await ask_user_to_select_raw(candidates, cfg, logger)
            
            assert result is None
            logger.info.assert_called_with("Discord human selection timed out or failed")

    @pytest.mark.asyncio
    async def test_ask_user_to_select_raw_client_task_timeout(self):
        """Test ask_user_to_select_raw when client task doesn't complete within timeout."""
        from PIL import Image
        
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='red')
        candidates = [{"image": test_image, "provider": "fal", "model": "test"}]
        
        cfg = Mock()
        cfg.discord.bot_token = "test_token"
        cfg.discord.user_id = "123456789"
        cfg.discord.timeout_sec = 0.1
        cfg.discord.batch_size = 10
        logger = Mock()
        
        # Mock Discord client and asyncio components
        with patch('pixelbliss.alerts.discord_select.discord.Client') as mock_client_class, \
             patch('pixelbliss.alerts.discord_select.asyncio.create_task') as mock_create_task, \
             patch('pixelbliss.alerts.discord_select.asyncio.wait_for') as mock_wait_for:
            
            mock_client = Mock()
            mock_client.start = AsyncMock()
            mock_client.is_closed.return_value = False
            mock_client.close = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock client task
            mock_task = Mock()
            mock_task.cancel = Mock()
            mock_create_task.return_value = mock_task
            
            # First wait_for succeeds (selection event), second times out (client task)
            mock_wait_for.side_effect = [None, asyncio.TimeoutError()]
            
            result = await ask_user_to_select_raw(candidates, cfg, logger)
            
            assert result is None
            logger.debug.assert_called_with("Client task didn't complete within 5 seconds, continuing")
            mock_task.cancel.assert_called_once()
