from typing import Protocol, List, Dict, Optional

class PromptProvider(Protocol):
    """Protocol defining the interface for prompt generation providers."""
    
    def make_base(self, category: str) -> str:
        """
        Generate a base prompt for the given category (legacy method).
        
        Args:
            category: The category/theme for the prompt.
            
        Returns:
            str: Generated base prompt.
        """
        ...
    
    def make_base_with_knobs(self, base_knobs: Dict[str, str], avoid_list: List[str] = None) -> str:
        """
        Generate a base prompt using the new knobs system.
        
        Args:
            base_knobs: Dictionary containing selected values for each base knob category
                       (vibe, palette, light, texture, composition, style)
            avoid_list: List of elements to avoid in the generated prompt
            
        Returns:
            str: Generated base prompt incorporating all knob values
        """
        ...

    def make_variants_from_base(self, base_prompt: str, k: int, art_styles: List[str] = None) -> List[str]:
        """
        Generate k variations of a base prompt (legacy method).
        
        Args:
            base_prompt: The original prompt to create variations from.
            k: Number of variations to generate.
            art_styles: List of art styles to randomly select from for each variant.
            
        Returns:
            List[str]: List of k prompt variations.
        """
        ...
    
    def make_variants_with_knobs(self, base_prompt: str, k: int, variant_knobs_list: List[Dict[str, str]], avoid_list: List[str] = None) -> List[str]:
        """
        Generate k variations of a base prompt using the new knobs system.
        
        Args:
            base_prompt: The original prompt to create variations from.
            k: Number of variations to generate.
            variant_knobs_list: List of k dictionaries, each containing variant knob values
                               (tone_curve, color_grade, surface_fx)
            avoid_list: List of elements to avoid in the generated prompts
            
        Returns:
            List[str]: List of k prompt variations incorporating knob values
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
