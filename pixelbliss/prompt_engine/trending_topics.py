"""
Trending topics service for dynamic theme recommendation using OpenAI GPT-5 with web search.

This module provides functionality to fetch current trending topics and recommend
themes for wallpaper generation based on real-time web data.
"""

import os
import time
from typing import Optional
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from ..logging_config import get_logger


class ThemeRecommendation(BaseModel):
    """Structured response model for theme recommendations."""
    theme: str = Field(
        description="A theme recommendation (1-2 sentences) that fully describes the wallpaper theme"
    )
    reasoning: str = Field(
        description="Brief explanation of why this theme is trending and suitable for wallpapers"
    )


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
            
            "EXAMPLES OF GOOD THEMES:\n"
            "• 'Ethereal aurora borealis dancing across a starlit winter sky with vibrant green and purple hues' (during solar activity news)\n"
            "• 'Delicate cherry blossoms in full bloom creating a dreamy pink canopy over a serene Japanese garden' (during spring season)\n"
            "• 'Cosmic nebulae with swirling galaxies and distant stars in deep space blues and purples' (during space exploration news)\n"
            "• 'Minimalist zen garden with smooth river stones and gentle bamboo shadows in soft earth tones' (during wellness trends)\n"
            "• 'Golden hour sunlight filtering through misty forest trees creating warm amber and honey tones' (during photography trends)"
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
            "Provide your recommendation with a brief explanation of why this theme is trending and suitable for wallpapers."
        )
        
        start_time = time.time()
        
        # Use web search enabled model with structured outputs
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # Enable web search if available
            tools=[{"type": "web_search"}],
            # Use structured outputs
            response_format=ThemeRecommendation
        )
        
        generation_time = time.time() - start_time
        
        # Parse the structured response
        theme_recommendation = response.choices[0].message.parsed
        
        # Extract the theme (don't clean up since it can be 1-2 sentences)
        theme = theme_recommendation.theme.strip()
        
        self.logger.info(f"Generated trending theme in {generation_time:.2f}s: {theme}")
        if progress_logger:
            progress_logger.substep("Trending theme generated", theme)
        
        return theme
