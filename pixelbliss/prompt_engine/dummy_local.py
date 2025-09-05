import asyncio
from typing import List, Optional, Dict
from .base import PromptProvider

class DummyLocalProvider(PromptProvider):
    """Dummy local implementation of the PromptProvider protocol for testing."""

    def make_base_with_knobs(self, base_knobs: Dict[str, str], avoid_list: List[str] = None) -> str:
        """
        Generate a base prompt using the knobs system for testing.
        
        Args:
            base_knobs: Dictionary containing selected values for each base knob category
            avoid_list: List of elements to avoid in the generated prompt
            
        Returns:
            str: Simple template-based prompt incorporating knob values
        """
        # Create a simple template that incorporates the knobs
        knobs_text = ", ".join([f"{knob}: {value}" for knob, value in base_knobs.items()])
        avoid_text = f", avoiding {', '.join(avoid_list)}" if avoid_list else ""
        
        return f"A beautiful aesthetic image with {knobs_text}, high detail, sharp focus{avoid_text}, no blur, no text."

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
