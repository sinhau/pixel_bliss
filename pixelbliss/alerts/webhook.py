import os
import requests

def send_success(category: str, model: str, tweet_url: str, image_url: str) -> None:
    webhook_url = os.getenv("ALERT_WEBHOOK_URL")
    if not webhook_url:
        return
    message = f"[PixelBliss] Posted {category} via {model} → {tweet_url}\nImage: {image_url}"
    requests.post(webhook_url, json={"content": message})

def send_failure(reason: str, details: str = "") -> None:
    webhook_url = os.getenv("ALERT_WEBHOOK_URL")
    if not webhook_url:
        return
    message = f"[PixelBliss] FAIL: {reason}"
    if details:
        message += f"\n{details}"
    requests.post(webhook_url, json={"content": message})
