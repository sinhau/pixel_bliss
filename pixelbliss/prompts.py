from .config import Config
from .prompt_engine.openai_gpt5 import OpenAIGPT5Provider
from .prompt_engine.dummy_local import DummyLocalProvider
from .prompt_engine.base import PromptProvider

def get_provider(cfg: Config) -> PromptProvider:
    """
    Get the appropriate prompt provider based on configuration.
    
    Args:
        cfg: Configuration object containing prompt generation settings.
        
    Returns:
        PromptProvider: Instance of the configured prompt provider.
        
    Raises:
        ValueError: If the configured provider is not supported.
    """
    if cfg.prompt_generation.provider == "openai":
        return OpenAIGPT5Provider(
            model=cfg.prompt_generation.model,
        )
    elif cfg.prompt_generation.provider == "dummy":
        return DummyLocalProvider()
    else:
        raise ValueError(f"Unknown prompt provider: {cfg.prompt_generation.provider}")

def make_base(category: str, cfg: Config) -> str:
    """
    Generate a base prompt for the given category.
    
    Args:
        category: The category/theme for the prompt (e.g., "sci-fi", "nature").
        cfg: Configuration object containing prompt generation settings.
        
    Returns:
        str: Generated base prompt for the category.
    """
    provider = get_provider(cfg)
    return provider.make_base(category)

def make_variants_from_base(base_prompt: str, k: int, cfg: Config) -> list[str]:
    """
    Generate k variations of a base prompt.
    
    Args:
        base_prompt: The original prompt to create variations from.
        k: Number of variations to generate.
        cfg: Configuration object containing prompt generation settings.
        
    Returns:
        list[str]: List of k prompt variations.
    """
    provider = get_provider(cfg)
    return provider.make_variants_from_base(base_prompt, k)

def make_alt_text(base_prompt: str, variant_prompt: str, cfg: Config) -> str:
    """
    Generate alt text for an image based on prompts.
    
    Args:
        base_prompt: The original base prompt used for image generation.
        variant_prompt: The specific variant prompt used for the final image.
        cfg: Configuration object containing prompt generation settings.
        
    Returns:
        str: Generated alt text describing the image.
    """
    provider = get_provider(cfg)
    return provider.make_alt_text(base_prompt, variant_prompt)
