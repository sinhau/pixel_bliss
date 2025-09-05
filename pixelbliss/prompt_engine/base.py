from typing import Protocol, List

class PromptProvider(Protocol):
    """Protocol defining the interface for prompt generation providers."""
    
    def make_base(self, category: str) -> str:
        """
        Generate a base prompt for the given category.
        
        Args:
            category: The category/theme for the prompt.
            
        Returns:
            str: Generated base prompt.
        """
        ...

    def make_variants_from_base(self, base_prompt: str, k: int) -> List[str]:
        """
        Generate k variations of a base prompt.
        
        Args:
            base_prompt: The original prompt to create variations from.
            k: Number of variations to generate.
            
        Returns:
            List[str]: List of k prompt variations.
        """
        ...

    def make_alt_text(self, base_prompt: str, variant_prompt: str) -> str:
        """
        Generate alt text for an image based on prompts.
        
        Args:
            base_prompt: The original base prompt.
            variant_prompt: The specific variant prompt used.
            
        Returns:
            str: Generated alt text describing the image.
        """
        ...
