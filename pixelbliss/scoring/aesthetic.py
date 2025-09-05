import replicate
import random
import asyncio
from pixelbliss.config import Config

def aesthetic_dummy_local(image_url: str, cfg: Config) -> float:
    """
    Generate a dummy aesthetic score for testing purposes.
    
    Args:
        image_url: URL of the image to score (unused in dummy mode)
        cfg: Configuration object containing aesthetic scoring settings
    
    Returns:
        float: Random aesthetic score between 0.0 and 1.0
    """
    # Generate a consistent random score based on image_url hash for reproducibility
    seed = hash(image_url) if image_url else random.randint(0, 1000000)
    random.seed(abs(seed) % 1000000)
    
    # Generate a score in the configured range, then normalize to [0,1]
    score_min = cfg.aesthetic_scoring.score_min
    score_max = cfg.aesthetic_scoring.score_max
    
    # Generate random score in the configured range
    raw_score = random.uniform(score_min, score_max)
    
    # Normalize to [0,1] range
    if score_max == score_min:
        normalized_score = 0.5
    else:
        normalized_score = (raw_score - score_min) / (score_max - score_min)
    
    # Clamp to [0,1] range
    return min(max(normalized_score, 0.0), 1.0)

def aesthetic_replicate(image_url: str, cfg: Config) -> float:
    """
    Score image aesthetics using Replicate API.
    
    Args:
        image_url: URL of the image to score
        cfg: Configuration object containing aesthetic scoring settings
    
    Returns:
        float: Aesthetic score between 0.0 and 1.0
    """
    try:
        output = replicate.run(
            cfg.aesthetic_scoring.model,
            input={"image": image_url}
        )
        
        # Handle different output formats from Replicate models
        if isinstance(output, dict):
            score = output.get('score', output.get('aesthetic_score', 0.5))
        elif isinstance(output, list) and len(output) > 0:
            score = output[0] if isinstance(output[0], (int, float)) else 0.5
        elif isinstance(output, (int, float)):
            score = output
        else:
            raise Exception(f"Unsupported aesthetic score output format: {output}")
            
        # Normalize score to [0,1] range using configured min/max
        score = float(score)
        score_min = cfg.aesthetic_scoring.score_min
        score_max = cfg.aesthetic_scoring.score_max
        
        # Linear normalization from [score_min, score_max] to [0, 1]
        if score_max == score_min:
            # Avoid division by zero
            normalized_score = 0.5
        else:
            normalized_score = (score - score_min) / (score_max - score_min)
        
        # Clamp to [0,1] range in case score is outside expected range
        return min(max(normalized_score, 0.0), 1.0)
        
    except Exception as e:
        from ..logging_config import get_logger
        logger = get_logger('scoring.aesthetic')
        logger.error(f"Replicate aesthetic scoring failed: {e}")
        return 0.5

def aesthetic(image_url: str, cfg: Config) -> float:
    """
    Score image aesthetics using the configured provider.
    
    Args:
        image_url: URL of the image to score
        cfg: Configuration object containing aesthetic scoring settings
    
    Returns:
        float: Aesthetic score between 0.0 and 1.0
    """
    provider = cfg.aesthetic_scoring.provider
    
    if provider == "dummy_local":
        return aesthetic_dummy_local(image_url, cfg)
    elif provider == "replicate":
        return aesthetic_replicate(image_url, cfg)
    else:
        raise NotImplementedError(f"Unknown provider {provider}")

async def aesthetic_dummy_local_async(image_url: str, cfg: Config) -> float:
    """
    Generate a dummy aesthetic score for testing purposes (async version).
    
    Args:
        image_url: URL of the image to score (unused in dummy mode)
        cfg: Configuration object containing aesthetic scoring settings
    
    Returns:
        float: Random aesthetic score between 0.0 and 1.0
    """
    # Run the synchronous function in a thread pool
    return await asyncio.to_thread(aesthetic_dummy_local, image_url, cfg)

async def aesthetic_replicate_async(image_url: str, cfg: Config) -> float:
    """
    Score image aesthetics using Replicate API (async version).
    
    Args:
        image_url: URL of the image to score
        cfg: Configuration object containing aesthetic scoring settings
    
    Returns:
        float: Aesthetic score between 0.0 and 1.0
    """
    # Run the synchronous function in a thread pool
    return await asyncio.to_thread(aesthetic_replicate, image_url, cfg)

async def aesthetic_async(image_url: str, cfg: Config) -> float:
    """
    Score image aesthetics using the configured provider (async version).
    
    Args:
        image_url: URL of the image to score
        cfg: Configuration object containing aesthetic scoring settings
    
    Returns:
        float: Aesthetic score between 0.0 and 1.0
    """
    provider = cfg.aesthetic_scoring.provider
    
    if provider == "dummy_local":
        return await aesthetic_dummy_local_async(image_url, cfg)
    elif provider == "replicate":
        return await aesthetic_replicate_async(image_url, cfg)
    else:
        raise NotImplementedError(f"Unknown provider {provider}")

async def score_candidates_parallel(candidates: list, cfg: Config, progress_logger=None) -> list:
    """
    Score multiple candidates in parallel using async aesthetic scoring.
    
    Args:
        candidates: List of candidate dictionaries with image_url
        cfg: Configuration object containing aesthetic scoring and async settings
        progress_logger: Optional progress logger for tracking scoring progress
    
    Returns:
        list: Updated candidates with aesthetic scores added
    """
    # Create semaphore for concurrency control if specified
    semaphore = None
    if cfg.image_generation.max_concurrency:
        semaphore = asyncio.Semaphore(cfg.image_generation.max_concurrency)
    
    # Start progress tracking
    if progress_logger:
        progress_logger.start_operation("aesthetic_scoring", len(candidates), "parallel aesthetic scoring")
    
    async def score_single_candidate(candidate):
        async def _score():
            try:
                image_url = candidate.get("image_url")
                if image_url:
                    score = await aesthetic_async(image_url, cfg)
                else:
                    score = 0.5
                candidate["aesthetic"] = score
                
                # Update progress
                if progress_logger:
                    progress_logger.update_operation_progress("aesthetic_scoring")
                
                return candidate
            except Exception as e:
                # Update progress even on failure
                if progress_logger:
                    progress_logger.update_operation_progress("aesthetic_scoring")
                raise e
        
        if semaphore:
            async with semaphore:
                return await _score()
        else:
            return await _score()
    
    # Score all candidates in parallel
    tasks = [score_single_candidate(c) for c in candidates]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle results and exceptions
    scored_candidates = []
    failed_count = 0
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # Use fallback score for failed candidates
            candidates[i]["aesthetic"] = 0.5
            scored_candidates.append(candidates[i])
            failed_count += 1
        else:
            scored_candidates.append(result)
    
    # Finish progress tracking
    if progress_logger:
        success = failed_count == 0
        progress_logger.finish_operation("aesthetic_scoring", success)
        if failed_count > 0:
            progress_logger.warning(f"{failed_count} aesthetic scores used fallback values")
    
    return scored_candidates
