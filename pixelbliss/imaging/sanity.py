from ..config import Config

def passes_floors(brightness_value: float, entropy_value: float, cfg: Config) -> bool:
    """
    Check if an image passes the minimum quality thresholds.
    
    Args:
        brightness_value: The brightness score of the image.
        entropy_value: The entropy score of the image.
        cfg: Configuration object containing minimum thresholds.
        
    Returns:
        bool: True if the image passes all quality floors, False otherwise.
    """
    return (
        cfg.ranking.entropy_min <= entropy_value and
        cfg.ranking.brightness_min <= brightness_value <= cfg.ranking.brightness_max
    )
