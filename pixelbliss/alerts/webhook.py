import os
import requests

def send_success(category: str, model: str, tweet_url: str, image_url: str) -> None:
    """
    Send a success notification webhook when an image is successfully posted.
    
    Args:
        category: The category/theme of the generated image.
        model: The AI model used for image generation.
        tweet_url: URL of the posted tweet.
        image_url: URL of the generated image.
    """
    webhook_url = os.getenv("ALERT_WEBHOOK_URL")
    if not webhook_url:
        return
    message = f"[PixelBliss] Posted {category} via {model} → {tweet_url}\nImage: {image_url}"
    requests.post(webhook_url, json={"content": message})

def send_failure(reason: str, details: str = "") -> None:
    """
    Send a failure notification webhook when an error occurs.
    
    Args:
        reason: Brief description of the failure reason.
        details: Optional additional details about the failure. Defaults to "".
    """
    webhook_url = os.getenv("ALERT_WEBHOOK_URL")
    if not webhook_url:
        return
    message = f"[PixelBliss] FAIL: {reason}"
    if details:
        message += f"\n{details}"
    requests.post(webhook_url, json={"content": message})
