from ..config import Config

def passes_floors(brightness_value: float, entropy_value: float, cfg: Config) -> bool:
    return (
        cfg.ranking.entropy_min <= entropy_value and
        cfg.ranking.brightness_min <= brightness_value <= cfg.ranking.brightness_max
    )
