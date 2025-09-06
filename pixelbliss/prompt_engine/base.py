from typing import Protocol, List, Dict, Optional

class PromptProvider(Protocol):
    """Protocol defining the interface for prompt generation providers using knobs system."""
    
    def make_base_with_knobs(self, base_knobs: Dict[str, str], avoid_list: List[str] = None, theme: str = None) -> str:
        """
        Generate a base prompt using the knobs system with theme integration.
        
        Args:
            base_knobs: Dictionary containing selected values for each base knob category
                       (vibe, palette, light, texture, composition, style)
            avoid_list: List of elements to avoid in the generated prompt
            theme: Theme/category hint that describes what the image will be about
            
        Returns:
            str: Generated base prompt incorporating theme and all knob values
        """
        ...
    
    def make_variants_with_knobs(self, base_prompt: str, k: int, variant_knobs_list: List[Dict[str, str]], avoid_list: List[str] = None) -> List[str]:
        """
        Generate k variations of a base prompt using the knobs system.
        
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
    
    def make_twitter_blurb(self, theme: str, image_path: str) -> str:
        """
        Generate a short, engaging blurb for Twitter posts using multimodal capabilities.
        
        Args:
            theme: The theme/category hint used for generation.
            image_path: Path to the generated image file.
            
        Returns:
            str: Generated blurb (haiku, philosophical quote, or short poem) 
                 that complements the image and theme, under 280 characters.
        """
        ...
