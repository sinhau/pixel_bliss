import os
import asyncio
import logging
from typing import List, Dict, Optional, Any, Union
from io import BytesIO
from PIL import Image
import discord
from discord.ext import commands
from pixelbliss.imaging.numbering import add_candidate_numbers_to_images


def _validate_discord_config(cfg, logger: logging.Logger) -> Optional[tuple[str, int, int]]:
    """
    Validate Discord configuration and return token, user_id, and timeout.
    
    Args:
        cfg: Configuration object containing Discord settings
        logger: Logger instance for debug/error messages
        
    Returns:
        Optional[tuple]: (token, user_id, timeout) if valid, None otherwise
    """
    # Get Discord configuration - access environment variables directly
    token = os.getenv(cfg.discord.bot_token_env, '')
    user_id_str = os.getenv(cfg.discord.user_id_env, '')
    timeout = cfg.discord.timeout_sec
    
    if not token or not user_id_str:
        logger.warning("Discord bot token or user ID not configured, skipping human selection")
        return None
    
    try:
        user_id = int(user_id_str)
    except (ValueError, TypeError):
        logger.error(f"Invalid DISCORD_USER_ID: {user_id_str}")
        return None
    
    return token, user_id, timeout


class DiscordClientManager:
    """Manages Discord client lifecycle and common operations."""
    
    def __init__(self, token: str, user_id: int, timeout: int, logger: logging.Logger):
        self.token = token
        self.user_id = user_id
        self.timeout = timeout
        self.logger = logger
        self.client = None
        self.selection_event = None
        self.selected_value = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        # Create Discord client
        intents = discord.Intents.none()
        self.client = discord.Client(intents=intents)
        self.selection_event = asyncio.Event()
        
        # Set up error handler
        @self.client.event
        async def on_error(event, *args, **kwargs):
            self.logger.error(f"Discord client error in {event}: {args}")
            self.selection_event.set()
        
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Close the client
        if self.client and not self.client.is_closed():
            try:
                await self.client.close()
            except (TypeError, Exception):
                # Handle case where client.close() is a Mock (not AsyncMock) during testing
                # or when close() raises an exception
                try:
                    if hasattr(self.client.close, '__call__'):
                        self.client.close()
                except:
                    # Ignore any errors during cleanup in tests
                    pass
    
    async def run_selection(self, setup_callback):
        """
        Run the Discord selection process.
        
        Args:
            setup_callback: Async function that sets up the Discord interaction
        """
        # Set up the on_ready handler
        @self.client.event
        async def on_ready():
            try:
                self.logger.debug(f"Discord bot connected as {self.client.user}")
                await setup_callback(self.client, self.user_id, self.logger)
            except Exception as e:
                self.logger.error(f"Error in Discord setup: {e}")
                self.selection_event.set()  # Signal completion even on error
        
        try:
            # Start the client
            client_task = asyncio.create_task(self.client.start(self.token))
            
            # Wait for selection or timeout
            try:
                await asyncio.wait_for(self.selection_event.wait(), timeout=self.timeout)
            except asyncio.TimeoutError:
                # Use a more specific message that can be customized by the caller
                timeout_msg = getattr(self, 'timeout_message', f"No selection received within {self.timeout} seconds, timing out")
                self.logger.warning(timeout_msg)
            
            # Close the client
            if not self.client.is_closed():
                try:
                    await self.client.close()
                except (TypeError, Exception):
                    # Handle case where client.close() is a Mock (not AsyncMock) during testing
                    # or when close() raises an exception
                    try:
                        if hasattr(self.client.close, '__call__'):
                            self.client.close()
                    except:
                        # Ignore any errors during cleanup in tests
                        pass
            
            # Wait for client task to complete
            try:
                await asyncio.wait_for(client_task, timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.debug("Client task didn't complete within 5 seconds, continuing")
                client_task.cancel()
        
        except Exception as e:
            # Use a more specific message that can be customized by the caller
            if hasattr(self, 'error_message') and callable(self.error_message):
                error_msg = self.error_message(e)
            else:
                error_msg = getattr(self, 'error_message', f"Error in Discord selection process: {e}")
            self.logger.error(error_msg)
            if self.client and not self.client.is_closed():
                try:
                    await self.client.close()
                except (TypeError, Exception):
                    # Handle case where client.close() is a Mock (not AsyncMock) during testing
                    # or when close() raises an exception
                    try:
                        if hasattr(self.client.close, '__call__'):
                            self.client.close()
                    except:
                        # Ignore any errors during cleanup in tests
                        pass
        
        return self.selected_value


class CandidateSelectView(discord.ui.View):
    """Discord UI View for candidate selection."""
    
    def __init__(self, candidates: List[Dict], batch_start: int, batch_end: int, 
                 user_id: int, timeout: int, manager: DiscordClientManager):
        super().__init__(timeout=timeout)
        self.candidates = candidates
        self.batch_start = batch_start
        self.batch_end = batch_end
        self.user_id = user_id
        self.manager = manager
        
        # Create select menu options
        select_options = []
        batch = candidates[batch_start:batch_end]
        
        for i, candidate in enumerate(batch):
            global_index = batch_start + i
            # Include some metadata in the label if available
            provider = candidate.get('provider', 'unknown')
            model = candidate.get('model', 'unknown')
            label = f"#{global_index+1} ({provider}/{model})"
            if len(label) > 100:  # Discord limit
                label = f"#{global_index+1} ({provider})"
            select_options.append(
                discord.SelectOption(
                    label=label,
                    value=str(global_index),
                    description=f"Select candidate #{global_index+1}"
                )
            )
        
        # Add "none" option to reject all candidates (only in the first batch)
        if batch_start == 0:
            select_options.append(
                discord.SelectOption(
                    label="❌ None (reject all)",
                    value="none",
                    description="Reject all candidates and end pipeline"
                )
            )
        
        # Add the select menu to the view
        self.add_item(CandidateSelect(select_options, self.user_id, self.candidates, self.manager))


class CandidateSelect(discord.ui.Select):
    """Discord UI Select component for candidate selection."""
    
    def __init__(self, options: List[discord.SelectOption], user_id: int, 
                 candidates: List[Dict], manager: DiscordClientManager):
        super().__init__(
            placeholder=f"Choose from candidates...",
            options=options
        )
        self.user_id = user_id
        self.candidates = candidates
        self.manager = manager
    
    async def callback(self, interaction: discord.Interaction):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("This selection is not for you.", ephemeral=True)
            return
        
        selected_value = self.values[0]
        
        if selected_value == "none":
            # User rejected all candidates - set to special sentinel value
            self.manager.selected_value = "none"
            await interaction.response.send_message("❌ All candidates rejected. Pipeline will end without posting.")
            self.manager.logger.info("User rejected all candidates via 'none' selection")
        else:
            selected_index = int(selected_value)
            # Return the index instead of the modified candidate object to avoid identity issues
            self.manager.selected_value = selected_index
            await interaction.response.send_message(f"✅ Using candidate #{selected_index+1}. Thanks!")
            self.manager.logger.info(f"User selected candidate #{selected_index+1}")
        
        self.manager.selection_event.set()


class ThemeSelectView(discord.ui.View):
    """Discord UI View for theme selection."""
    
    def __init__(self, themes: List, user_id: int, timeout: int, manager: DiscordClientManager):
        super().__init__(timeout=timeout)
        self.themes = themes
        self.user_id = user_id
        self.manager = manager
        
        # Create select menu options for themes
        select_options = []
        for i, theme in enumerate(themes):
            # Truncate theme text if too long for Discord
            theme_text = theme.theme if hasattr(theme, 'theme') else str(theme)
            reasoning = theme.reasoning if hasattr(theme, 'reasoning') else ""
            
            label = f"Theme {i+1}"
            if len(theme_text) <= 100:
                label = theme_text[:100]
            
            description = reasoning[:100] if reasoning else f"Theme option {i+1}"
            
            select_options.append(
                discord.SelectOption(
                    label=label,
                    value=str(i),
                    description=description
                )
            )
        
        # Add "none" option to use fallback themes
        select_options.append(
            discord.SelectOption(
                label="❌ Use fallback themes",
                value="fallback",
                description="Skip trending themes and use curated fallback themes"
            )
        )
        
        # Add the select menu to the view
        self.add_item(ThemeSelect(select_options, self.user_id, self.themes, self.manager))


class ThemeSelect(discord.ui.Select):
    """Discord UI Select component for theme selection."""
    
    def __init__(self, options: List[discord.SelectOption], user_id: int, 
                 themes: List, manager: DiscordClientManager):
        super().__init__(
            placeholder="Choose a theme for wallpaper generation...",
            options=options
        )
        self.user_id = user_id
        self.themes = themes
        self.manager = manager
    
    async def callback(self, interaction: discord.Interaction):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("This selection is not for you.", ephemeral=True)
            return
        
        selected_value = self.values[0]
        
        if selected_value == "fallback":
            # User chose to use fallback themes
            self.manager.selected_value = "fallback"
            await interaction.response.send_message("✅ Using fallback themes instead of trending themes.")
            self.manager.logger.info("User chose to use fallback themes")
        else:
            selected_index = int(selected_value)
            selected_theme_obj = self.themes[selected_index]
            selected_theme = selected_theme_obj.theme if hasattr(selected_theme_obj, 'theme') else str(selected_theme_obj)
            self.manager.selected_value = selected_theme
            await interaction.response.send_message(f"✅ Selected theme: {selected_theme[:100]}...")
            self.manager.logger.info(f"User selected theme #{selected_index+1}: {selected_theme}")
        
        self.manager.selection_event.set()


async def _setup_candidate_selection(client: discord.Client, user_id: int, logger: logging.Logger, 
                                   candidates: List[Dict], batch_size: int, manager: DiscordClientManager):
    """Setup callback for candidate selection."""
    # Get user and create DM channel
    user = await client.fetch_user(user_id)
    dm = await user.create_dm()
    
    logger.info(f"Sending {len(candidates)} candidates in batches of {batch_size} to {user.display_name}")
    
    # Send candidates in batches (candidates should already be numbered)
    for batch_start in range(0, len(candidates), batch_size):
        batch_end = min(batch_start + batch_size, len(candidates))
        batch = candidates[batch_start:batch_end]
        
        logger.debug(f"Sending batch {batch_start//batch_size + 1}: candidates {batch_start+1}-{batch_end}")
        
        # Prepare attachments
        files = []
        for i, candidate in enumerate(batch):
            global_index = batch_start + i
            
            # Convert PIL Image to JPEG bytes
            img = candidate['image']
            
            # Optionally downscale to keep file size reasonable
            if max(img.size) > 2048:
                img = img.copy()
                img.thumbnail((2048, 2048), Image.LANCZOS)
            
            # Convert to JPEG bytes
            img_bytes = BytesIO()
            img.save(img_bytes, format='JPEG', quality=85)
            img_bytes.seek(0)
            
            filename = f"candidate_{global_index+1:03d}.jpg"
            files.append(discord.File(img_bytes, filename=filename))
        
        # Create and send the batch message with select view
        view = CandidateSelectView(candidates, batch_start, batch_end, user_id, manager.timeout, manager)
        content = f"**Batch {batch_start//batch_size + 1}** - Candidates {batch_start+1}-{batch_end} of {len(candidates)}"
        await dm.send(content=content, files=files, view=view)
        
        # Small delay between batches to avoid rate limits
        if batch_end < len(candidates):
            await asyncio.sleep(0.3)
    
    logger.info("All candidate batches sent, waiting for selection...")


async def _setup_theme_selection(client: discord.Client, user_id: int, logger: logging.Logger, 
                                themes: List, manager: DiscordClientManager):
    """Setup callback for theme selection."""
    # Get user and create DM channel
    user = await client.fetch_user(user_id)
    dm = await user.create_dm()
    
    logger.info(f"Sending {len(themes)} theme options to {user.display_name}")
    
    # Create content with theme details
    content_lines = ["**🎨 Choose a trending theme for wallpaper generation:**\n"]
    for i, theme in enumerate(themes, 1):
        theme_text = theme.theme if hasattr(theme, 'theme') else str(theme)
        reasoning = theme.reasoning if hasattr(theme, 'reasoning') else ""
        content_lines.append(f"**{i}.** {theme_text}")
        if reasoning:
            content_lines.append(f"   *{reasoning}*")
        content_lines.append("")
    
    content = "\n".join(content_lines)
    
    # Discord has a 2000 character limit for message content
    if len(content) > 1900:
        content = content[:1900] + "...\n\n*Use the dropdown below to select a theme.*"
    
    # Send the theme selection message
    view = ThemeSelectView(themes, user_id, manager.timeout, manager)
    await dm.send(content=content, view=view)
    
    logger.info("Theme options sent, waiting for selection...")


async def ask_user_to_select_raw(candidates: List[Dict], cfg, logger: logging.Logger) -> Optional[Dict]:
    """
    Sends DMs in batches of images (inline attachments). Each batch includes a Select menu
    with options for the global indices. Returns the first selected candidate, or None on timeout.
    
    Args:
        candidates: List of candidate dictionaries with 'image' (PIL.Image) key
        cfg: Configuration object containing Discord settings
        logger: Logger instance for debug/error messages
        
    Returns:
        Optional[Dict]: The selected candidate dict, or None on timeout/error
    """
    # Validate Discord configuration
    config_result = _validate_discord_config(cfg, logger)
    if not config_result:
        return None
    
    token, user_id, timeout = config_result
    batch_size = cfg.discord.batch_size
    
    if not candidates:
        logger.warning("No candidates to select from")
        return None
    
    logger.info(f"Starting Discord human-in-the-loop selection for {len(candidates)} candidates")
    
    # Add candidate numbers to images for easier selection (before Discord setup)
    numbered_candidates = add_candidate_numbers_to_images(candidates)
    
    # Use the Discord client manager
    async with DiscordClientManager(token, user_id, timeout, logger) as manager:
        selected_candidate = await manager.run_selection(
            lambda client, user_id, logger: _setup_candidate_selection(
                client, user_id, logger, numbered_candidates, batch_size, manager
            )
        )
    
    if selected_candidate:
        logger.info("Discord human selection completed successfully")
    else:
        logger.info("Discord human selection timed out or failed")
    
    return selected_candidate


async def ask_user_to_select_theme(themes: List, cfg, logger: logging.Logger) -> Optional[str]:
    """
    Sends DMs with theme options for user selection. Returns the selected theme string, or None on timeout.
    
    Args:
        themes: List of ThemeRecommendation objects with 'theme' and 'reasoning' attributes
        cfg: Configuration object containing Discord settings
        logger: Logger instance for debug/error messages
        
    Returns:
        Optional[str]: The selected theme string, or None on timeout/error
    """
    # Validate Discord configuration
    config_result = _validate_discord_config(cfg, logger)
    if not config_result:
        logger.warning("Discord bot token or user ID not configured, skipping theme selection")
        return None
    
    token, user_id, timeout = config_result
    
    if not themes:
        logger.warning("No themes to select from")
        return None
    
    logger.info(f"Starting Discord theme selection for {len(themes)} themes")
    
    # Use the Discord client manager
    async with DiscordClientManager(token, user_id, timeout, logger) as manager:
        # Set custom error messages for theme selection
        manager.timeout_message = f"No theme selection received within {timeout} seconds, timing out"
        manager.error_message = lambda e: f"Error in Discord theme selection process: {e}"
        
        selected_theme = await manager.run_selection(
            lambda client, user_id, logger: _setup_theme_selection(
                client, user_id, logger, themes, manager
            )
        )
    
    if selected_theme:
        logger.info("Discord theme selection completed successfully")
    else:
        logger.info("Discord theme selection timed out or failed")
    
    return selected_theme
