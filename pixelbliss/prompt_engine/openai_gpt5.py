import os
from typing import List
from openai import OpenAI
from .base import PromptProvider

class OpenAIGPT5Provider(PromptProvider):
    """OpenAI GPT-5 implementation of the PromptProvider protocol."""
    
    def __init__(self, model: str = "gpt-5"):
        """
        Initialize the OpenAI GPT-5 provider.
        
        Args:
            model: The OpenAI model to use. Defaults to "gpt-5".
        """
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def make_base(self, category: str) -> str:
        """
        Generate a base prompt for the given category using OpenAI GPT-5.
        
        Args:
            category: The category/theme for the prompt.
            
        Returns:
            str: Generated base prompt optimized for aesthetic wallpaper generation.
        """
        system_prompt = (
            "You are a creative AI that generates aesthetic image prompts for wallpaper art. "
            "Create prompts that are highly visual, artistic, very detailed, and suitable for AI image generation. "
            "They should induce aesthetic and eudaimonic pleasure, so the overall style of the image should induce this. "
            "Rules: No real people, no logos, no NSFW content. Include negative prompts like 'blurry, low quality, text'."
        )
        user_prompt = f"Generate a detailed, creative prompt for an aesthetic {category} themed image suitable for a wallpaper. Adhere strictly to the provided system prompts."

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        return response.choices[0].message.content.strip()

    def make_variants_from_base(self, base_prompt: str, k: int) -> List[str]:
        """
        Generate k variations of a base prompt using OpenAI GPT-5.
        
        Args:
            base_prompt: The original prompt to create variations from.
            k: Number of variations to generate.
            
        Returns:
            List[str]: List of k prompt variations with different styles and elements.
        """
        system_prompt = (
            "You are a creative AI that generates variations of image prompts. "
            "Given a base prompt, create variations by changing styles, colors, compositions, or adding artistic elements. "
            "Try to add some random thematic element that induces aesthetic and eudaimonic pleasure "
            "Keep them aesthetic and wallpaper-friendly. "
            "Rules: No real people, no logos, no NSFW. Include negative prompts."
        )
        variants = []
        for _ in range(k):
            user_prompt = f"Base prompt: {base_prompt}\n\nGenerate one creative variation of this prompt."

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
            )
            variant = response.choices[0].message.content.strip()
            variants.append(variant)
        return variants

    def make_alt_text(self, base_prompt: str, variant_prompt: str) -> str:
        """
        Generate alt text for an image based on prompts using OpenAI GPT-5.
        
        Args:
            base_prompt: The original base prompt used for image generation.
            variant_prompt: The specific variant prompt used for the final image.
            
        Returns:
            str: Concise, descriptive alt text for accessibility.
        """
        system_prompt = (
            "You are an AI that generates concise, descriptive alt text for images. "
            "Alt text should be 1-2 sentences, describing the visual elements without mentioning AI generation."
        )
        user_prompt = f"Generate alt text for an image based on this prompt: {variant_prompt}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=100  # Shorter for alt text
        )
        return response.choices[0].message.content.strip()
