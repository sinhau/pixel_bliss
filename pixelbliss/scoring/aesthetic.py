import replicate
import random
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
        print(f"Replicate aesthetic scoring failed: {e}")
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
        print(f"Unknown aesthetic scoring provider: {provider}, falling back to dummy")
        return aesthetic_dummy_local(image_url, cfg)
