import asyncio
from typing import List, Optional, Dict
from .base import PromptProvider
from ..logging_config import get_logger

class DummyLocalProvider(PromptProvider):
    """Dummy local implementation of the PromptProvider protocol for testing."""

    def make_base_with_knobs(self, base_knobs: Dict[str, str], avoid_list: List[str] = None, theme: str = None) -> str:
        """
        Generate a base prompt using the knobs system with theme integration for testing.
        
        Args:
            base_knobs: Dictionary containing selected values for each base knob category
            avoid_list: List of elements to avoid in the generated prompt
            theme: Theme/category hint that describes what the image will be about
            
        Returns:
            str: Simple template-based prompt incorporating theme and knob values
        """
        # Create a simple template that incorporates the theme and knobs
        theme_text = f"themed around {theme}, " if theme else ""
        knobs_text = ", ".join([f"{knob}: {value}" for knob, value in base_knobs.items()])
        avoid_text = f", avoiding {', '.join(avoid_list)}" if avoid_list else ""
        
        return f"A beautiful aesthetic image {theme_text}with {knobs_text}, high detail, sharp focus{avoid_text}, no blur, no text."

    def make_variants_with_knobs(self, base_prompt: str, k: int, variant_knobs_list: List[Dict[str, str]], avoid_list: List[str] = None) -> List[str]:
        """
        Generate k variations of a base prompt using the knobs system for testing.
        
        Args:
            base_prompt: The original prompt to create variations from.
            k: Number of variations to generate.
            variant_knobs_list: List of k dictionaries, each containing variant knob values
            avoid_list: List of elements to avoid in the generated prompts
            
        Returns:
            List[str]: List of k simple prompt variations incorporating knob values
        """
        variants = []
        avoid_text = f", avoiding {', '.join(avoid_list)}" if avoid_list else ""
        
        for i, variant_knobs in enumerate(variant_knobs_list):
            knobs_text = ", ".join([f"{knob}: {value}" for knob, value in variant_knobs.items()])
            variants.append(f"{base_prompt} Variation {i+1}: with {knobs_text}{avoid_text}.")
        
        return variants

    def make_alt_text(self, base_prompt: str, variant_prompt: str) -> str:
        """
        Generate simple alt text for testing purposes.
        
        Args:
            base_prompt: The original base prompt (unused in dummy implementation).
            variant_prompt: The specific variant prompt (unused in dummy implementation).
            
        Returns:
            str: Generic alt text for testing.
        """
        return "A stunning aesthetic image featuring vibrant colors and intricate details."

    def make_twitter_blurb(self, theme: str, image_path: str) -> str:
        """
        Generate a simple spiritual/philosophical quote for testing purposes.
        
        Args:
            theme: The theme/category hint used for generation.
            image_path: Path to the generated image file (unused in dummy implementation).
            
        Returns:
            str: Simple spiritual/philosophical quote that complements the theme.
        """
        logger = get_logger('prompt_engine.dummy_local')
        logger.info(f"[dummy] Generating spiritual/philosophical quote (theme='{theme}')")
        logger.debug(f"[dummy] Image path provided: '{image_path}' (unused in dummy mode)")
        
        # Grounded spiritual/philosophical quotes for testing
        blurbs = [
            f"In {theme}, we discover that true beauty lies not in perfection, but in the harmony of imperfect moments.",
            f"Every element of {theme} reminds us that peace is not the absence of complexity, but finding stillness within it.",
            f"The essence of {theme} teaches us that wonder is always present—we need only pause to notice.",
            f"Through {theme}, we learn that the most profound truths are often found in the simplest observations.",
            f"In contemplating {theme}, we realize that beauty is not something we see, but something we recognize within ourselves.",
            f"The wisdom of {theme} shows us that every moment contains both the question and the answer we seek.",
            f"Within {theme} lies the reminder that we are both observers and participants in the dance of existence.",
            f"The spirit of {theme} whispers that gratitude transforms ordinary moments into sacred experiences."
        ]
        
        # Use hash of theme to get consistent but varied results
        index = hash(theme) % len(blurbs)
        selected_blurb = blurbs[index]
        
        logger.debug(f"[dummy] Selected quote template #{index + 1} of {len(blurbs)}")
        logger.info(f"[dummy] Quote generated successfully (len={len(selected_blurb)})")
        
        return selected_blurb

    async def make_variants_with_knobs_async(self, base_prompt: str, k: int, variant_knobs_list: List[Dict[str, str]], avoid_list: List[str] = None, max_concurrency: Optional[int] = None) -> List[str]:
        """
        Generate k variations of a base prompt using knobs system asynchronously for testing.
        
        Args:
            base_prompt: The original prompt to create variations from.
            k: Number of variations to generate.
            variant_knobs_list: List of k dictionaries, each containing variant knob values
            avoid_list: List of elements to avoid in the generated prompts
            max_concurrency: Maximum number of concurrent operations (unused in dummy implementation).
            
        Returns:
            List[str]: List of k simple prompt variations.
        """
        # Simulate async work with a small delay
        await asyncio.sleep(0.1)
        
        # Use the synchronous method for actual generation
        return self.make_variants_with_knobs(base_prompt, k, variant_knobs_list, avoid_list)
