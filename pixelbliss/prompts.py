from .config import Config
from .prompt_engine.openai_gpt5 import OpenAIGPT5Provider
from .prompt_engine.dummy_local import DummyLocalProvider
from .prompt_engine.base import PromptProvider

def get_provider(cfg: Config) -> PromptProvider:
    if cfg.prompt_generation.provider == "openai":
        return OpenAIGPT5Provider(
            model=cfg.prompt_generation.model,
            temperature=cfg.prompt_generation.temperature,
            max_tokens=cfg.prompt_generation.max_tokens
        )
    elif cfg.prompt_generation.provider == "dummy":
        return DummyLocalProvider()
    else:
        raise ValueError(f"Unknown prompt provider: {cfg.prompt_generation.provider}")

def make_base(category: str, cfg: Config) -> str:
    provider = get_provider(cfg)
    return provider.make_base(category)

def make_variants_from_base(base_prompt: str, k: int, cfg: Config) -> list[str]:
    provider = get_provider(cfg)
    return provider.make_variants_from_base(base_prompt, k)

def make_alt_text(base_prompt: str, variant_prompt: str, cfg: Config) -> str:
    provider = get_provider(cfg)
    return provider.make_alt_text(base_prompt, variant_prompt)
