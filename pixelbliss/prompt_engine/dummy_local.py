from typing import List
from .base import PromptProvider

class DummyLocalProvider(PromptProvider):
    """Dummy local implementation of the PromptProvider protocol for testing."""
    
    def make_base(self, category: str) -> str:
        """
        Generate a simple base prompt for testing purposes.
        
        Args:
            category: The category/theme for the prompt.
            
        Returns:
            str: Simple template-based prompt for the category.
        """
        return f"A beautiful {category} landscape with vibrant colors, high detail, sharp focus, no blur, no text."

    def make_variants_from_base(self, base_prompt: str, k: int, art_styles: List[str] = None) -> List[str]:
        """
        Generate k simple variations of a base prompt for testing.
        
        Args:
            base_prompt: The original prompt to create variations from.
            k: Number of variations to generate.
            art_styles: List of art styles to randomly select from for each variant.
            
        Returns:
            List[str]: List of k simple prompt variations.
        """
        variants = []
        for i in range(k):
            style_text = f" in {art_styles[i % len(art_styles)]} style" if art_styles else ""
            variants.append(f"{base_prompt} Variation {i+1}: with added artistic elements, enhanced lighting{style_text}.")
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
