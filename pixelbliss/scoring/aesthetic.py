import replicate
from pixelbliss.config import Config

def aesthetic(image_url: str, cfg: Config) -> float:
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
