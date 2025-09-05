from PIL import Image
import replicate
from typing import Union
from pixelbliss.config import Config

def aesthetic(image_input: Union[str, Image.Image], cfg: Config = None) -> float:
    """
    Score image aesthetics using Replicate API or fallback method.
    
    Args:
        image_input: Either an image URL (str) or PIL Image object
        cfg: Configuration object containing aesthetic scoring settings
    
    Returns:
        float: Aesthetic score between 0.0 and 1.0
    """
    try:
        # If config is provided and we have a URL, use Replicate
        if cfg and isinstance(image_input, str):
            return _score_with_replicate(image_input, cfg)
        
        # Fallback to local scoring for PIL Images or when no config
        elif isinstance(image_input, Image.Image):
            return _score_local_fallback(image_input)
        
        # If we have a URL but no config, try with default settings
        elif isinstance(image_input, str):
            # Create default config for aesthetic scoring
            from pixelbliss.config import AestheticScoring
            default_aesthetic_config = AestheticScoring()
            return _score_with_replicate(image_input, type('Config', (), {'aesthetic_scoring': default_aesthetic_config})())
        
        else:
            return 0.5
            
    except Exception as e:
        print(f"Aesthetic scoring failed: {e}")
        return 0.5

def _score_with_replicate(image_url: str, cfg: Config) -> float:
    """Score image using Replicate aesthetic model."""
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
            score = 0.5
            
        # Ensure score is in [0,1] range
        return min(max(float(score), 0.0), 1.0)
        
    except Exception as e:
        print(f"Replicate aesthetic scoring failed: {e}")
        return 0.5

def _score_local_fallback(image: Image.Image) -> float:
    """
    Fallback local aesthetic scoring for PIL Images.
    This is a simple heuristic-based approach.
    """
    try:
        # Simple heuristic based on image properties
        # This is a placeholder - you could implement more sophisticated local scoring
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Basic aesthetic heuristics
        width, height = image.size
        aspect_ratio = width / height
        
        # Prefer certain aspect ratios (golden ratio, common photo ratios)
        golden_ratio = 1.618
        common_ratios = [1.0, 4/3, 3/2, 16/9, golden_ratio, 1/golden_ratio]
        
        ratio_score = 0.5
        for ratio in common_ratios:
            if abs(aspect_ratio - ratio) < 0.1:
                ratio_score = 0.8
                break
        
        # Size preference (larger images often look better)
        size_score = min(1.0, (width * height) / (1920 * 1080))
        
        # Combine scores
        final_score = (ratio_score * 0.6 + size_score * 0.4)
        
        return min(max(final_score, 0.0), 1.0)
        
    except Exception as e:
        print(f"Local aesthetic scoring failed: {e}")
        return 0.5
