import os
import yaml
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from pathlib import Path

class PromptGeneration(BaseModel):
    provider: str = "openai"
    model: str = "gpt-5"
    num_prompt_variants: int = 3
    async_enabled: bool = True
    max_concurrency: Optional[int] = None  # None means no limit, defaults to number of variants
    use_knobs: bool = True  # Use new knobs system instead of legacy categories/art_styles
    variant_strategy: str = "single"  # "single" = vary one knob, "multiple" = vary all knobs

class ImageGeneration(BaseModel):
    provider_order: List[str] = ["fal", "replicate"]
    model_fal: List[str] = ["black-forest-labs/flux-1.1"]
    model_replicate: List[str] = ["black-forest-labs/flux"]
    retries_per_image: int = 2
    async_enabled: bool = True
    max_concurrency: Optional[int] = None  # None means no limit, defaults to number of variants

class Ranking(BaseModel):
    w_brightness: float = 0.25
    w_entropy: float = 0.25
    w_aesthetic: float = 0.50
    w_local_quality: float = 0.20
    entropy_min: float = 3.5
    brightness_min: float = 10
    brightness_max: float = 245
    phash_distance_min: int = 6

class Upscale(BaseModel):
    enabled: bool = True
    provider: str = "replicate"
    model: str = "real-esrgan-4x"
    factor: int = 2

class WallpaperVariant(BaseModel):
    name: str
    w: int
    h: int

class AestheticScoring(BaseModel):
    provider: str = "replicate"
    model: str = "laion/aesthetic-predictor:v2-14"
    score_min: float = 0.0
    score_max: float = 1.0

class LocalQuality(BaseModel):
    resize_long: int = 768
    min_side: int = 512
    ar_min: float = 0.5
    ar_max: float = 2.0
    sharpness_min: float = 120.0
    sharpness_good: float = 600.0
    clip_max: float = 0.20

class Alerts(BaseModel):
    enabled: bool = True
    webhook_url_env: str = "ALERT_WEBHOOK_URL"

class Discord(BaseModel):
    enabled: bool = False
    bot_token_env: str = "DISCORD_BOT_TOKEN"
    user_id_env: str = "DISCORD_USER_ID"
    timeout_sec: int = 900
    batch_size: int = 10

class Config(BaseModel):
    timezone: str = "America/Los_Angeles"
    categories: List[str] = ["sci-fi", "tech", "mystic", "geometry", "nature", "neo-noir", "watercolor", "cosmic-minimal"]
    art_styles: List[str] = ["Realism", "Impressionism", "Watercolor", "Oil painting", "Digital Art"]
    category_selection_method: str = "time"
    rotation_minutes: int = 180
    prompt_generation: PromptGeneration = Field(default_factory=PromptGeneration)
    image_generation: ImageGeneration = Field(default_factory=ImageGeneration)
    ranking: Ranking = Field(default_factory=Ranking)
    aesthetic_scoring: AestheticScoring = Field(default_factory=AestheticScoring)
    local_quality: LocalQuality = Field(default_factory=LocalQuality)
    upscale: Upscale = Field(default_factory=Upscale)
    wallpaper_variants: List[WallpaperVariant] = []
    alerts: Alerts = Field(default_factory=Alerts)
    discord: Discord = Field(default_factory=Discord)

def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from a YAML file and return a Config object.
    
    Args:
        config_path: Path to the YAML configuration file. Defaults to "config.yaml".
        
    Returns:
        Config: Parsed configuration object with all settings.
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r') as f:
        data = yaml.safe_load(f)

    # Load environment variables for sensitive data
    if 'alerts' in data and 'webhook_url_env' in data['alerts']:
        env_var = data['alerts']['webhook_url_env']
        data['alerts']['webhook_url'] = os.getenv(env_var, "")

    return Config(**data)
