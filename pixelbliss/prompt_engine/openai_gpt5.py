import os
from typing import List
from openai import OpenAI
from .base import PromptProvider

class OpenAIGPT5Provider(PromptProvider):
    def __init__(self, model: str = "gpt-5"):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def make_base(self, category: str) -> str:
        system_prompt = (
            "You are a creative AI that generates aesthetic image prompts for wallpaper art. "
            "Create prompts that are highly visual, artistic, and suitable for AI image generation. "
            "Focus on abstract, scenic, or conceptual themes. "
            "Rules: No real people, no logos, no NSFW content. Include negative prompts like 'blurry, low quality, text'."
        )
        user_prompt = f"Generate a detailed, creative prompt for an aesthetic {category} themed image suitable for a wallpaper."

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        return response.choices[0].message.content.strip()

    def make_variants_from_base(self, base_prompt: str, k: int) -> List[str]:
        system_prompt = (
            "You are a creative AI that generates variations of image prompts. "
            "Given a base prompt, create variations by changing styles, colors, compositions, or adding artistic elements. "
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
