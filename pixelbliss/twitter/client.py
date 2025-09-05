import os
import tweepy
from typing import List

def get_client():
    return tweepy.Client(
        consumer_key=os.getenv("X_API_KEY"),
        consumer_secret=os.getenv("X_API_SECRET"),
        access_token=os.getenv("X_ACCESS_TOKEN"),
        access_token_secret=os.getenv("X_ACCESS_TOKEN_SECRET")
    )

def upload_media(paths: List[str]) -> List[str]:
    auth = tweepy.OAuth1UserHandler(
        consumer_key=os.getenv("X_API_KEY"),
        consumer_secret=os.getenv("X_API_SECRET"),
        access_token=os.getenv("X_ACCESS_TOKEN"),
        access_token_secret=os.getenv("X_ACCESS_TOKEN_SECRET")
    )
    api = tweepy.API(auth)
    media_ids = []
    for path in paths:
        media = api.media_upload(path)
        media_ids.append(media.media_id_string)
    return media_ids

def set_alt_text(media_id: str, alt: str) -> None:
    client = get_client()
    client.create_media_metadata(media_id, alt_text={"text": alt})

def create_tweet(text: str, media_ids: List[str]) -> str:
    client = get_client()
    response = client.create_tweet(text=text, media_ids=media_ids)
    return response.data['id']
