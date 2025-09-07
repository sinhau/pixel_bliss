"""
Trending topics service for dynamic theme recommendation using OpenAI GPT-5 with web search.

This module provides functionality to fetch current trending topics and recommend
themes for wallpaper generation based on real-time web data.
"""

import os
import time
from typing import Optional, List
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from ..logging_config import get_logger


class ThemeRecommendation(BaseModel):
    """Structured response model for a single theme recommendation."""
    theme: str = Field(
        description="A theme recommendation (1-2 sentences) that fully describes the wallpaper theme"
    )
    reasoning: str = Field(
        description="Brief explanation of why this theme is trending and suitable for wallpapers"
    )


class MultipleThemeRecommendations(BaseModel):
    """Structured response model for multiple theme recommendations."""
    themes: List[ThemeRecommendation] = Field(
        description="List of 3-5 diverse theme recommendations based on current trends",
        min_length=3,
        max_length=5
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
        )
        
        user_prompt = (
            "Search the web for current trending topics, cultural movements, seasonal themes, and aesthetic trends. "
            "Based on your findings, recommend ONE theme that would make a beautiful, timely wallpaper. "
            "Consider:\n\n"
            "1. Current events and cultural moments\n"
            "2. Seasonal relevance and natural phenomena\n"
            "3. Aesthetic movements and design trends\n"
            "4. Popular visual themes in art and photography\n"
            "5. Emerging color palettes and visual styles\n"
            "6. Any relevent trending new about people, places, animals, etc.\n\n"
            "Provide your recommendation with a brief explanation of why this theme is trending and suitable for wallpapers."
        )
        
        start_time = time.time()
        
        # Phase 1: Use built-in web_search tool to gather current trends (Responses API supports web_search)
        research_response = await self.async_client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=(
                user_prompt
                + "\n\nWhen you've completed your web research, provide a concise summary of key trends and influences "
                "you found (bullet points are fine). Do not provide the final theme yet."
            ),
            tools=[{"type": "web_search"}],
        )

        # Extract text robustly across SDK variants
        research_summary = getattr(research_response, "output_text", None)
        if not isinstance(research_summary, str) or not research_summary.strip():
            try:
                research_summary = research_response.choices[0].message.content or ""
            except Exception:
                research_summary = ""
        
        # Phase 2: Convert research summary into a structured recommendation (Responses API with structured parsing)
        structured_response = await self.async_client.responses.parse(
            model=self.model,
            instructions=(
                "You convert web research into a single high-quality wallpaper theme recommendation. "
                "Don't describe any aesthetic details, simply provide a concise description of the theme to guide the subject matter of the wallpaper. "
                "Return ONLY a structured object that matches the provided schema."
            ),
            input=(
                "Here is the research summary of current trends:\n\n"
                f"{research_summary}\n\n"
                "Based on these findings, produce exactly one theme recommendation with a brief reasoning."
            ),
            text_format=ThemeRecommendation,
        )
        
        generation_time = time.time() - start_time
        
        # Parse the structured response robustly across SDK variants
        output_parsed = getattr(structured_response, "output_parsed", None)
        if isinstance(output_parsed, ThemeRecommendation):
            theme_recommendation = output_parsed
        else:
            theme_recommendation = structured_response.choices[0].message.parsed
        
        # Extract the theme (don't clean up since it can be 1-2 sentences)
        theme = theme_recommendation.theme.strip()
        
        self.logger.info(f"Generated trending theme in {generation_time:.2f}s: {theme}")
        if progress_logger:
            progress_logger.substep("Trending theme generated", theme)
        
        return theme
    
    async def get_trending_themes_async(self, progress_logger=None) -> List[ThemeRecommendation]:
        """
        Get multiple trending theme recommendations asynchronously.
        
        Args:
            progress_logger: Optional progress logger for tracking.
            
        Returns:
            List[ThemeRecommendation]: A list of 3-5 recommended themes based on trending topics.
        """
        if progress_logger:
            progress_logger.substep("Fetching trending topics for multiple themes")
        
        self.logger.info("Fetching trending topics and generating multiple theme recommendations (async)")
        
        system_prompt = (
            "You are PixelBliss Trend Analyst, an expert at identifying current cultural trends, seasonal themes, "
            "and aesthetic movements that would make compelling wallpaper subjects. Your mission is to analyze "
            "current web trends and recommend 3-5 diverse themes that would create beautiful, timely wallpaper art.\n\n"
            
            "TREND ANALYSIS EXPERTISE:\n"
            "• Monitor current events, cultural movements, and seasonal trends\n"
            "• Identify aesthetic themes that resonate with contemporary audiences\n"
            "• Focus on visually compelling subjects that translate well to wallpaper art\n"
            "• Balance trending relevance with timeless visual appeal\n"
            "• Ensure diversity across different categories (nature, abstract, cultural, etc.)\n\n"
        )
        
        user_prompt = (
            "Search the web for current trending topics, cultural movements, seasonal themes, and aesthetic trends. "
            "Based on your findings, recommend 3-5 DIVERSE themes that would make beautiful, timely wallpapers. "
            "Consider:\n\n"
            "1. Current events and cultural moments\n"
            "2. Seasonal relevance and natural phenomena\n"
            "3. Aesthetic movements and design trends\n"
            "4. Popular visual themes in art and photography\n"
            "5. Emerging color palettes and visual styles\n"
            "6. Any relevant trending news about people, places, animals, etc.\n\n"
            "Ensure the themes are diverse across different categories and styles. "
            "Each theme should be distinct and offer different visual possibilities."
        )
        
        start_time = time.time()
        
        # Phase 1: Use built-in web_search tool to gather current trends
        research_response = await self.async_client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=(
                user_prompt
                + "\n\nWhen you've completed your web research, provide a comprehensive summary of key trends and influences "
                "you found across different categories (bullet points are fine). Do not provide the final themes yet."
            ),
            tools=[{"type": "web_search"}],
        )

        # Extract text robustly across SDK variants
        research_summary = getattr(research_response, "output_text", None)
        if not isinstance(research_summary, str) or not research_summary.strip():
            try:
                research_summary = research_response.choices[0].message.content or ""
            except Exception:
                research_summary = ""
        
        # Phase 2: Convert research summary into structured recommendations
        structured_response = await self.async_client.responses.parse(
            model=self.model,
            instructions=(
                "You convert web research into 3-5 high-quality, diverse wallpaper theme recommendations. "
                "Don't describe any aesthetic details, simply provide concise descriptions of the themes to guide the subject matter of the wallpapers. "
                "Ensure themes are diverse across different categories (nature, abstract, cultural, seasonal, etc.). "
                "Return ONLY a structured object that matches the provided schema."
            ),
            input=(
                "Here is the research summary of current trends:\n\n"
                f"{research_summary}\n\n"
                "Based on these findings, produce 3-5 diverse theme recommendations with brief reasoning for each."
            ),
            text_format=MultipleThemeRecommendations,
        )
        
        generation_time = time.time() - start_time
        
        # Parse the structured response robustly across SDK variants
        output_parsed = getattr(structured_response, "output_parsed", None)
        if isinstance(output_parsed, MultipleThemeRecommendations):
            theme_recommendations = output_parsed
        else:
            theme_recommendations = structured_response.choices[0].message.parsed
        
        themes = theme_recommendations.themes
        
        self.logger.info(f"Generated {len(themes)} trending themes in {generation_time:.2f}s")
        for i, theme in enumerate(themes, 1):
            self.logger.info(f"Theme {i}: {theme.theme}")
        
        if progress_logger:
            progress_logger.substep("Multiple trending themes generated", f"{len(themes)} themes")
        
        return themes
