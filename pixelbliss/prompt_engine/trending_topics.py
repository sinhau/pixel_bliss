"""
Trending topics service for dynamic theme recommendation using OpenAI GPT-5 with web search.

This module provides functionality to fetch current trending topics and recommend
themes for wallpaper generation based on real-time web data.
"""

import os
import time
from typing import Optional
from openai import AsyncOpenAI
from ..logging_config import get_logger

class TrendingTopicsProvider:
    """Provider for fetching trending topics and recommending themes using OpenAI GPT-5 with web search."""
    
    def __init__(self, model: str = "gpt-5"):
        """
        Initialize the trending topics provider.
        
        Args:
            model: The OpenAI model to use. Defaults to "gpt-5".
        """
        self.model = model
        self.async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.logger = get_logger('trending_topics')
    
    
    async def get_trending_theme_async(self, progress_logger=None) -> str:
        """
        Get a trending theme recommendation asynchronously.
        
        Args:
            progress_logger: Optional progress logger for tracking.
            
        Returns:
            str: A recommended theme based on trending topics.
        """
        try:
            if progress_logger:
                progress_logger.substep("Fetching trending topics from web")
            
            self.logger.info("Fetching trending topics and generating theme recommendation (async)")
            
            system_prompt = (
                "You are PixelBliss Trend Analyst, an expert at identifying current cultural trends, seasonal themes, "
                "and aesthetic movements that would make compelling wallpaper subjects. Your mission is to analyze "
                "current web trends and recommend a single theme that would create beautiful, timely wallpaper art.\n\n"
                
                "TREND ANALYSIS EXPERTISE:\n"
                "• Monitor current events, cultural movements, and seasonal trends\n"
                "• Identify aesthetic themes that resonate with contemporary audiences\n"
                "• Focus on visually compelling subjects that translate well to wallpaper art\n"
                "• Balance trending relevance with timeless visual appeal\n\n"
                
                "THEME RECOMMENDATION CRITERIA:\n"
                "• Must be visually rich and suitable for wallpaper generation\n"
                "• Should reflect current cultural zeitgeist or seasonal relevance\n"
                "• Avoid overly specific or commercial references\n"
                "• Focus on concepts that inspire beautiful, contemplative imagery\n"
                "• Consider global trends, not just regional ones\n\n"
                
                "OUTPUT FORMAT:\n"
                "Provide a single, concise theme recommendation (1-3 words) that captures the essence "
                "of current trends while being suitable for aesthetic wallpaper generation.\n\n"
                
                "EXAMPLES OF GOOD THEMES:\n"
                "• 'aurora borealis' (during solar activity news)\n"
                "• 'cherry blossoms' (during spring season)\n"
                "• 'cosmic wonder' (during space exploration news)\n"
                "• 'minimalist zen' (during wellness trends)\n"
                "• 'golden hour' (during photography trends)"
            )
            
            user_prompt = (
                "Search the web for current trending topics, cultural movements, seasonal themes, and aesthetic trends. "
                "Based on your findings, recommend ONE theme that would make a beautiful, timely wallpaper. "
                "Consider:\n\n"
                "1. Current events and cultural moments\n"
                "2. Seasonal relevance and natural phenomena\n"
                "3. Aesthetic movements and design trends\n"
                "4. Popular visual themes in art and photography\n"
                "5. Emerging color palettes and visual styles\n\n"
                "Provide your recommendation as a single, concise theme (1-3 words) that captures the current zeitgeist "
                "while being perfect for generating beautiful wallpaper art."
            )
            
            start_time = time.time()
            
            # Use web search enabled model
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=100,
                temperature=0.7,
                # Enable web search if available
                tools=[{"type": "web_search"}] if hasattr(self.async_client, 'tools') else None
            )
            
            generation_time = time.time() - start_time
            theme = response.choices[0].message.content.strip()
            
            # Clean up the theme (remove quotes, extra punctuation)
            theme = theme.strip('"\'.,!?').lower()
            
            self.logger.info(f"Generated trending theme in {generation_time:.2f}s: {theme}")
            if progress_logger:
                progress_logger.substep("Trending theme generated", theme)
            
            return theme
            
        except Exception as e:
            self.logger.error(f"Failed to get trending theme: {e}")
            if progress_logger:
                progress_logger.warning("Trending theme failed, using fallback")
            
            # Fallback to a curated list of generally trending themes
            fallback_themes = [
                "aurora borealis", "cherry blossoms", "cosmic wonder", "minimalist zen",
                "golden hour", "ocean waves", "mountain peaks", "forest mist",
                "desert dunes", "city lights", "abstract flow", "geometric harmony"
            ]
            
            import random
            fallback_theme = random.choice(fallback_themes)
            self.logger.info(f"Using fallback theme: {fallback_theme}")
            return fallback_theme
