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
            
        # Normalize score to [0,1] range
        score = float(score)
        
        # Handle common score ranges from different aesthetic models
        if score >= 0 and score <= 1:
            # Already in [0,1] range
            normalized_score = score
        elif score >= 0 and score <= 10:
            # Scale from [0,10] to [0,1]
            normalized_score = score / 10.0
        elif score >= 1 and score <= 5:
            # Scale from [1,5] to [0,1]
            normalized_score = (score - 1) / 4.0
        elif score >= -1 and score <= 1:
            # Scale from [-1,1] to [0,1]
            normalized_score = (score + 1) / 2.0
        else:
            # For any other range, use sigmoid-like normalization
            # This maps any real number to [0,1] range
            import math
            normalized_score = 1 / (1 + math.exp(-score))
        
        return min(max(normalized_score, 0.0), 1.0)
        
    except Exception as e:
        print(f"Replicate aesthetic scoring failed: {e}")
        return 0.5
