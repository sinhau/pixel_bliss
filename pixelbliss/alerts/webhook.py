import os
import requests
from ..config import Config

def send_success(category: str, model: str, tweet_url: str, image_url: str, cfg: Config) -> None:
    """
    Send a success notification webhook when an image is successfully posted.
    
    Args:
        category: The category/theme of the generated image.
        model: The AI model used for image generation.
        tweet_url: URL of the posted tweet.
        image_url: URL of the generated image.
        cfg: Configuration object containing alerts settings.
    """
    if not cfg.alerts.enabled:
        return
    
    webhook_url = os.getenv(cfg.alerts.webhook_url_env)
    if not webhook_url:
        return
    message = f"[PixelBliss] Posted {category} via {model} → {tweet_url}\nImage: {image_url}"
    requests.post(webhook_url, json={"content": message})

def send_failure(reason: str, cfg: Config, details: str = "") -> None:
    """
    Send a failure notification webhook when an error occurs.
    
    Args:
        reason: Brief description of the failure reason.
        cfg: Configuration object containing alerts settings.
        details: Optional additional details about the failure. Defaults to "".
    """
    if not cfg.alerts.enabled:
        return
    
    webhook_url = os.getenv(cfg.alerts.webhook_url_env)
    if not webhook_url:
        return
    message = f"[PixelBliss] FAIL: {reason}"
    if details:
        message += f"\n{details}"
    requests.post(webhook_url, json={"content": message})
