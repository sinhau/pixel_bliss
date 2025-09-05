import asyncio
import time
from .config import Config
from .prompt_engine.openai_gpt5 import OpenAIGPT5Provider
from .prompt_engine.dummy_local import DummyLocalProvider
from .prompt_engine.base import PromptProvider
from .prompt_engine.knobs import KnobSelector
from .logging_config import get_logger

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

def make_base(category: str, cfg: Config, progress_logger=None) -> str:
    """
    Generate a base prompt using the knobs system.
    
    Args:
        category: The category/theme hint for knob selection.
        cfg: Configuration object containing prompt generation settings.
        progress_logger: Optional progress logger for tracking generation progress.
        
    Returns:
        str: Generated base prompt using diverse aesthetic knobs.
    """
    return make_base_with_knobs(category, cfg, progress_logger)

def make_variants_from_base(base_prompt: str, k: int, cfg: Config, progress_logger=None) -> list[str]:
    """
    Generate k variations of a base prompt using the knobs system.
    
    Args:
        base_prompt: The original prompt to create variations from.
        k: Number of variations to generate.
        cfg: Configuration object containing prompt generation settings.
        progress_logger: Optional progress logger for tracking generation progress.
        
    Returns:
        list[str]: List of k prompt variations using diverse aesthetic knobs.
    """
    return make_variants_with_knobs(base_prompt, k, cfg, progress_logger)

def make_base_with_knobs(category: str, cfg: Config, progress_logger=None) -> str:
    """
    Generate a base prompt using the new knobs system.
    
    Args:
        category: The category/theme hint for knob selection (optional, for future use).
        cfg: Configuration object containing prompt generation settings.
        progress_logger: Optional progress logger for tracking generation progress.
        
    Returns:
        str: Generated base prompt using diverse aesthetic knobs.
    """
    logger = get_logger('prompts')
    provider = get_provider(cfg)
    
    # Select base knobs
    base_knobs = KnobSelector.select_base_knobs(category)
    avoid_list = KnobSelector.get_avoid_list()
    
    # Log the start of base prompt generation
    if progress_logger:
        progress_logger.log_base_prompt_generation(f"knobs:{category}", cfg.prompt_generation.provider, cfg.prompt_generation.model)
        progress_logger.log_base_knobs_selected(base_knobs)
    else:
        logger.info(f"Base knobs selected: {base_knobs}")
    
    start_time = time.time()
    try:
        # Use knobs-based generation
        base_prompt = provider.make_base_with_knobs(base_knobs, avoid_list)
        generation_time = time.time() - start_time
        
        # Log successful generation
        if progress_logger:
            progress_logger.log_base_prompt_success(base_prompt, generation_time)
        else:
            logger.info(f"Base prompt generated with knobs in {generation_time:.2f}s: {base_prompt[:80]}...")
            logger.debug(f"Used knobs: {base_knobs}")
        
        return base_prompt
        
    except Exception as e:
        generation_time = time.time() - start_time
        logger.error(f"Knobs-based base prompt generation failed after {generation_time:.2f}s: {e}")
        raise

def make_variants_with_knobs(base_prompt: str, k: int, cfg: Config, progress_logger=None) -> list[str]:
    """
    Generate k variations of a base prompt using the new knobs system.
    
    Args:
        base_prompt: The original prompt to create variations from.
        k: Number of variations to generate.
        cfg: Configuration object containing prompt generation settings.
        progress_logger: Optional progress logger for tracking generation progress.
        
    Returns:
        list[str]: List of k prompt variations using diverse aesthetic knobs.
    """
    logger = get_logger('prompts')
    provider = get_provider(cfg)
    
    # Generate variant knobs for each variation
    variant_knobs_list = []
    for _ in range(k):
        if cfg.prompt_generation.variant_strategy == "single":
            # Vary only one knob to maintain identity
            variant_knobs = KnobSelector.select_single_variant_knob()
        else:
            # Vary all knobs for maximum diversity
            variant_knobs = KnobSelector.select_variant_knobs()
        variant_knobs_list.append(variant_knobs)
    
    avoid_list = KnobSelector.get_avoid_list()
    
    # Log the start of variant prompt generation
    if progress_logger:
        progress_logger.log_variant_prompt_generation_start(k, cfg.prompt_generation.provider, cfg.prompt_generation.model, False)
        progress_logger.log_variant_knobs_selected(variant_knobs_list, cfg.prompt_generation.variant_strategy)
    else:
        logger.info(f"Variant knobs selected (strategy: {cfg.prompt_generation.variant_strategy}):")
        for i, variant_knobs in enumerate(variant_knobs_list, 1):
            logger.info(f"  Variant {i}: {variant_knobs}")
    
    start_time = time.time()
    try:
        # Use knobs-based generation
        variant_prompts = provider.make_variants_with_knobs(base_prompt, k, variant_knobs_list, avoid_list)
        generation_time = time.time() - start_time
        
        # Log successful generation
        if progress_logger:
            progress_logger.log_variant_prompt_success(variant_prompts, generation_time)
        else:
            logger.info(f"Generated {len(variant_prompts)} prompt variants with knobs in {generation_time:.2f}s")
            for i, (variant, knobs) in enumerate(zip(variant_prompts, variant_knobs_list), 1):
                logger.debug(f"Variant {i}: {variant[:60]}... (knobs: {knobs})")
        
        return variant_prompts
        
    except Exception as e:
        generation_time = time.time() - start_time
        error_msg = str(e)
        if progress_logger:
            progress_logger.log_variant_prompt_error(error_msg, generation_time)
        else:
            logger.error(f"Knobs-based variant prompt generation failed after {generation_time:.2f}s: {error_msg}")
        raise

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

async def make_variants_from_base_async(base_prompt: str, k: int, cfg: Config, progress_logger=None) -> list[str]:
    """
    Generate k variations of a base prompt in parallel using the knobs system.
    
    Args:
        base_prompt: The original prompt to create variations from.
        k: Number of variations to generate.
        cfg: Configuration object containing prompt generation settings.
        progress_logger: Optional progress logger for tracking generation progress.
        
    Returns:
        list[str]: List of k prompt variations using diverse aesthetic knobs.
    """
    logger = get_logger('prompts')
    provider = get_provider(cfg)
    
    # Generate variant knobs for each variation
    variant_knobs_list = []
    for _ in range(k):
        if cfg.prompt_generation.variant_strategy == "single":
            # Vary only one knob to maintain identity
            variant_knobs = KnobSelector.select_single_variant_knob()
        else:
            # Vary all knobs for maximum diversity
            variant_knobs = KnobSelector.select_variant_knobs()
        variant_knobs_list.append(variant_knobs)
    
    avoid_list = KnobSelector.get_avoid_list()
    
    # Log the start of variant prompt generation
    if progress_logger:
        progress_logger.log_variant_prompt_generation_start(k, cfg.prompt_generation.provider, cfg.prompt_generation.model, True)
        progress_logger.log_variant_knobs_selected(variant_knobs_list, cfg.prompt_generation.variant_strategy)
    else:
        logger.info(f"Variant knobs selected (strategy: {cfg.prompt_generation.variant_strategy}):")
        for i, variant_knobs in enumerate(variant_knobs_list, 1):
            logger.info(f"  Variant {i}: {variant_knobs}")
    
    start_time = time.time()
    try:
        # Check if provider supports async knobs generation
        if hasattr(provider, 'make_variants_with_knobs_async'):
            variant_prompts = await provider.make_variants_with_knobs_async(
                base_prompt, k, variant_knobs_list, avoid_list, 
                cfg.prompt_generation.max_concurrency, progress_logger
            )
        else:
            # Fallback to sync method wrapped in asyncio.to_thread
            variant_prompts = await asyncio.to_thread(
                provider.make_variants_with_knobs, base_prompt, k, variant_knobs_list, avoid_list
            )
        
        generation_time = time.time() - start_time
        
        # Log successful generation
        if progress_logger:
            progress_logger.log_variant_prompt_success(variant_prompts, generation_time)
        else:
            logger.info(f"Generated {len(variant_prompts)} prompt variants with knobs async in {generation_time:.2f}s")
            for i, (variant, knobs) in enumerate(zip(variant_prompts, variant_knobs_list), 1):
                logger.debug(f"Variant {i}: {variant[:60]}... (knobs: {knobs})")
        
        return variant_prompts
        
    except Exception as e:
        generation_time = time.time() - start_time
        error_msg = str(e)
        if progress_logger:
            progress_logger.log_variant_prompt_error(error_msg, generation_time)
        else:
            logger.error(f"Async knobs-based variant prompt generation failed after {generation_time:.2f}s: {error_msg}")
        raise
