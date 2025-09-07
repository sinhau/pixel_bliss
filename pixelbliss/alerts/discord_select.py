import os
import asyncio
import logging
from typing import List, Dict, Optional
from io import BytesIO
from PIL import Image
import discord
from discord.ext import commands
from pixelbliss.imaging.numbering import add_candidate_numbers_to_images


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
    # Get Discord configuration - access environment variables directly
    token = os.getenv(cfg.discord.bot_token_env, '')
    user_id_str = os.getenv(cfg.discord.user_id_env, '')
    timeout = cfg.discord.timeout_sec
    batch_size = cfg.discord.batch_size
    
    if not token or not user_id_str:
        logger.warning("Discord bot token or user ID not configured, skipping human selection")
        return None
    
    try:
        user_id = int(user_id_str)
    except (ValueError, TypeError):
        logger.error(f"Invalid DISCORD_USER_ID: {user_id_str}")
        return None
    
    if not candidates:
        logger.warning("No candidates to select from")
        return None
    
    logger.info(f"Starting Discord human-in-the-loop selection for {len(candidates)} candidates")
    
    # Add candidate numbers to images for easier selection
    numbered_candidates = add_candidate_numbers_to_images(candidates)
    
    # Create Discord client
    intents = discord.Intents.none()
    client = discord.Client(intents=intents)
    
    selected_candidate = None
    selection_event = asyncio.Event()
    
    @client.event
    async def on_ready():
        nonlocal selected_candidate
        try:
            logger.debug(f"Discord bot connected as {client.user}")
            
            # Get user and create DM channel
            user = await client.fetch_user(user_id)
            dm = await user.create_dm()
            
            logger.info(f"Sending {len(candidates)} candidates in batches of {batch_size} to {user.display_name}")
            
            # Send candidates in batches
            for batch_start in range(0, len(candidates), batch_size):
                batch_end = min(batch_start + batch_size, len(candidates))
                batch = numbered_candidates[batch_start:batch_end]
                
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
                
                # Create select menu for this batch
                select_options = []
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
                
                class CandidateSelect(discord.ui.View):
                    def __init__(self):
                        super().__init__(timeout=timeout)
                    
                    @discord.ui.select(
                        placeholder=f"Choose from candidates {batch_start+1}-{batch_end}...",
                        options=select_options
                    )
                    async def select_candidate(self, interaction: discord.Interaction, select: discord.ui.Select):
                        nonlocal selected_candidate
                        
                        if interaction.user.id != user_id:
                            await interaction.response.send_message("This selection is not for you.", ephemeral=True)
                            return
                        
                        selected_value = select.values[0]
                        
                        if selected_value == "none":
                            # User rejected all candidates - set to special sentinel value
                            selected_candidate = "none"
                            await interaction.response.send_message("❌ All candidates rejected. Pipeline will end without posting.")
                            logger.info("User rejected all candidates via 'none' selection")
                        else:
                            selected_index = int(selected_value)
                            selected_candidate = candidates[selected_index]
                            await interaction.response.send_message(f"✅ Using candidate #{selected_index+1}. Thanks!")
                            logger.info(f"User selected candidate #{selected_index+1}")
                        
                        selection_event.set()
                
                # Send the batch message
                view = CandidateSelect()
                content = f"**Batch {batch_start//batch_size + 1}** - Candidates {batch_start+1}-{batch_end} of {len(candidates)}"
                await dm.send(content=content, files=files, view=view)
                
                # Small delay between batches to avoid rate limits
                if batch_end < len(candidates):
                    await asyncio.sleep(0.3)
            
            logger.info("All candidate batches sent, waiting for selection...")
            
        except Exception as e:
            logger.error(f"Error sending Discord messages: {e}")
            selection_event.set()  # Signal completion even on error
    
    @client.event
    async def on_error(event, *args, **kwargs):
        logger.error(f"Discord client error in {event}: {args}")
        selection_event.set()
    
    try:
        # Start the client
        client_task = asyncio.create_task(client.start(token))
        
        # Wait for selection or timeout
        try:
            await asyncio.wait_for(selection_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"No selection received within {timeout} seconds, timing out")
        
        # Close the client
        if not client.is_closed():
            await client.close()
        
        # Wait for client task to complete
        try:
            await asyncio.wait_for(client_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.debug("Client task didn't complete within 5 seconds, continuing")
            client_task.cancel()
    
    except Exception as e:
        logger.error(f"Error in Discord selection process: {e}")
        if not client.is_closed():
            try:
                await client.close()
            except:
                pass
    
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
    # Get Discord configuration - access environment variables directly
    token = os.getenv(cfg.discord.bot_token_env, '')
    user_id_str = os.getenv(cfg.discord.user_id_env, '')
    timeout = cfg.discord.timeout_sec
    
    if not token or not user_id_str:
        logger.warning("Discord bot token or user ID not configured, skipping theme selection")
        return None
    
    try:
        user_id = int(user_id_str)
    except (ValueError, TypeError):
        logger.error(f"Invalid DISCORD_USER_ID: {user_id_str}")
        return None
    
    if not themes:
        logger.warning("No themes to select from")
        return None
    
    logger.info(f"Starting Discord theme selection for {len(themes)} themes")
    
    # Create Discord client
    intents = discord.Intents.none()
    client = discord.Client(intents=intents)
    
    selected_theme = None
    selection_event = asyncio.Event()
    
    @client.event
    async def on_ready():
        nonlocal selected_theme
        try:
            logger.debug(f"Discord bot connected as {client.user}")
            
            # Get user and create DM channel
            user = await client.fetch_user(user_id)
            dm = await user.create_dm()
            
            logger.info(f"Sending {len(themes)} theme options to {user.display_name}")
            
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
            
            class ThemeSelect(discord.ui.View):
                def __init__(self):
                    super().__init__(timeout=timeout)
                
                @discord.ui.select(
                    placeholder="Choose a theme for wallpaper generation...",
                    options=select_options
                )
                async def select_theme(self, interaction: discord.Interaction, select: discord.ui.Select):
                    nonlocal selected_theme
                    
                    if interaction.user.id != user_id:
                        await interaction.response.send_message("This selection is not for you.", ephemeral=True)
                        return
                    
                    selected_value = select.values[0]
                    
                    if selected_value == "fallback":
                        # User chose to use fallback themes
                        selected_theme = "fallback"
                        await interaction.response.send_message("✅ Using fallback themes instead of trending themes.")
                        logger.info("User chose to use fallback themes")
                    else:
                        selected_index = int(selected_value)
                        selected_theme_obj = themes[selected_index]
                        selected_theme = selected_theme_obj.theme if hasattr(selected_theme_obj, 'theme') else str(selected_theme_obj)
                        await interaction.response.send_message(f"✅ Selected theme: {selected_theme[:100]}...")
                        logger.info(f"User selected theme #{selected_index+1}: {selected_theme}")
                    
                    selection_event.set()
            
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
            view = ThemeSelect()
            await dm.send(content=content, view=view)
            
            logger.info("Theme options sent, waiting for selection...")
            
        except Exception as e:
            logger.error(f"Error sending Discord theme selection: {e}")
            selection_event.set()  # Signal completion even on error
    
    @client.event
    async def on_error(event, *args, **kwargs):
        logger.error(f"Discord client error in {event}: {args}")
        selection_event.set()
    
    try:
        # Start the client
        client_task = asyncio.create_task(client.start(token))
        
        # Wait for selection or timeout
        try:
            await asyncio.wait_for(selection_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"No theme selection received within {timeout} seconds, timing out")
        
        # Close the client
        if not client.is_closed():
            await client.close()
        
        # Wait for client task to complete
        try:
            await asyncio.wait_for(client_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.debug("Client task didn't complete within 5 seconds, continuing")
            client_task.cancel()
    
    except Exception as e:
        logger.error(f"Error in Discord theme selection process: {e}")
        if not client.is_closed():
            try:
                await client.close()
            except:
                pass
    
    if selected_theme:
        logger.info("Discord theme selection completed successfully")
    else:
        logger.info("Discord theme selection timed out or failed")
    
    return selected_theme
