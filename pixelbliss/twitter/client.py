import os
import tweepy
from typing import List
from ..imaging.compression import prepare_for_twitter_upload

def get_client():
    """
    Create and return a configured Twitter/X API client.
    
    Returns:
        tweepy.Client: Configured Twitter API client.
    """
    return tweepy.Client(
        consumer_key=os.getenv("X_API_KEY"),
        consumer_secret=os.getenv("X_API_SECRET"),
        access_token=os.getenv("X_ACCESS_TOKEN"),
        access_token_secret=os.getenv("X_ACCESS_TOKEN_SECRET")
    )

def upload_media(paths: List[str]) -> List[str]:
    """
    Upload media files to Twitter/X and return media IDs.
    Automatically compresses images to stay under 5MB limit while maintaining quality.
    
    Args:
        paths: List of file paths to upload.
        
    Returns:
        List[str]: List of media ID strings from Twitter.
    """
    # Compress images if needed to stay under Twitter's 5MB limit
    processed_paths = prepare_for_twitter_upload(paths)
    
    auth = tweepy.OAuth1UserHandler(
        consumer_key=os.getenv("X_API_KEY"),
        consumer_secret=os.getenv("X_API_SECRET"),
        access_token=os.getenv("X_ACCESS_TOKEN"),
        access_token_secret=os.getenv("X_ACCESS_TOKEN_SECRET")
    )
    api = tweepy.API(auth)
    media_ids = []
    for path in processed_paths:
        media = api.media_upload(path)
        media_ids.append(media.media_id_string)
    return media_ids

def set_alt_text(media_id: str, alt: str) -> None:
    """
    Set alt text for uploaded media on Twitter/X.
    
    Args:
        media_id: The media ID string from Twitter.
        alt: Alt text description for accessibility.
    """
    client = get_client()
    client.create_media_metadata(media_id, alt_text={"text": alt})

def create_tweet(text: str, media_ids: List[str]) -> str:
    """
    Create a tweet with text and attached media.
    
    Args:
        text: Tweet text content.
        media_ids: List of media ID strings to attach.
        
    Returns:
        str: The ID of the created tweet.
    """
    client = get_client()
    response = client.create_tweet(text=text, media_ids=media_ids)
    return response.data['id']
