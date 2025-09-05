from typing import List
from .base import PromptProvider

class DummyLocalProvider(PromptProvider):
    def make_base(self, category: str) -> str:
        return f"A beautiful {category} landscape with vibrant colors, high detail, sharp focus, no blur, no text."

    def make_variants_from_base(self, base_prompt: str, k: int) -> List[str]:
        variants = []
        for i in range(k):
            variants.append(f"{base_prompt} Variation {i+1}: with added artistic elements, enhanced lighting.")
        return variants

    def make_alt_text(self, base_prompt: str, variant_prompt: str) -> str:
        return "A stunning aesthetic image featuring vibrant colors and intricate details."
