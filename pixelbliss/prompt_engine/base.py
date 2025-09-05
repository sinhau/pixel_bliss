from typing import Protocol, List

class PromptProvider(Protocol):
    def make_base(self, category: str) -> str:
        ...

    def make_variants_from_base(self, base_prompt: str, k: int) -> List[str]:
        ...

    def make_alt_text(self, base_prompt: str, variant_prompt: str) -> str:
        ...
