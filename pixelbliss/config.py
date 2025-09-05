import os
import yaml
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from pathlib import Path

class PromptGeneration(BaseModel):
    provider: str = "openai"
    model: str = "gpt-5"
    temperature: float = 0.8
    max_tokens: int = 400

class Generation(BaseModel):
    num_prompt_variants: int = 3
    images_per_variant: int = 3
    provider_order: List[str] = ["fal", "replicate"]
    model_fal: str = "black-forest-labs/flux-1.1"
    model_replicate: str = "black-forest-labs/flux"
    retries_per_image: int = 2

class Ranking(BaseModel):
    w_brightness: float = 0.25
    w_entropy: float = 0.25
    w_aesthetic: float = 0.50
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

class Alerts(BaseModel):
    enabled: bool = True
    webhook_url_env: str = "ALERT_WEBHOOK_URL"

class Config(BaseModel):
    timezone: str = "America/Los_Angeles"
    categories: List[str] = ["sci-fi", "tech", "mystic", "geometry", "nature", "neo-noir", "watercolor", "cosmic-minimal"]
    rotation_minutes: int = 180
    prompt_generation: PromptGeneration = Field(default_factory=PromptGeneration)
    generation: Generation = Field(default_factory=Generation)
    ranking: Ranking = Field(default_factory=Ranking)
    upscale: Upscale = Field(default_factory=Upscale)
    wallpaper_variants: List[WallpaperVariant] = []
    alerts: Alerts = Field(default_factory=Alerts)

def load_config(config_path: str = "config.yaml") -> Config:
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
