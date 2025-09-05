import os
import tweepy
from typing import List, Optional
from ..imaging.compression import prepare_for_twitter_upload

def get_client():
    """
    Create and return a configured Twitter/X API v2 client using OAuth 2.0.
    
    Returns:
        tweepy.Client: Configured Twitter API v2 client.
    """
    return tweepy.Client(
        bearer_token=os.getenv("X_BEARER_TOKEN"),
        consumer_key=os.getenv("X_API_KEY"),
        consumer_secret=os.getenv("X_API_SECRET"),
        access_token=os.getenv("X_ACCESS_TOKEN"),
        access_token_secret=os.getenv("X_ACCESS_TOKEN_SECRET"),
        wait_on_rate_limit=True
    )

def upload_media(paths: List[str]) -> List[str]:
    """
    Upload media files to Twitter/X using v2 API and return media IDs.
    Automatically compresses images to stay under 5MB limit while maintaining quality.
    
    Args:
        paths: List of file paths to upload.
        
    Returns:
        List[str]: List of media ID strings from Twitter.
        
    Raises:
        Exception: If media upload fails.
    """
    # Compress images if needed to stay under Twitter's 5MB limit
    processed_paths = prepare_for_twitter_upload(paths)
    
    if not processed_paths:
        return []
    
    # Use OAuth 2.0 client for v2 media upload
    client = get_client()
    media_ids = []
    
    for path in processed_paths:
        try:
            # Upload media using v2 API
            media = client.media_upload(filename=path)
            media_ids.append(str(media.media_id))
        except Exception as e:
            # If v2 upload fails, fallback to v1.1 for compatibility
            # This provides backward compatibility during transition period
            auth = tweepy.OAuth1UserHandler(
                consumer_key=os.getenv("X_API_KEY"),
                consumer_secret=os.getenv("X_API_SECRET"),
                access_token=os.getenv("X_ACCESS_TOKEN"),
                access_token_secret=os.getenv("X_ACCESS_TOKEN_SECRET")
            )
            api = tweepy.API(auth)
            media = api.media_upload(path)
            media_ids.append(media.media_id_string)
    
    return media_ids

def set_alt_text(media_id: str, alt: str) -> None:
    """
    Set alt text for uploaded media on Twitter/X using v2 API.
    
    Args:
        media_id: The media ID string from Twitter.
        alt: Alt text description for accessibility.
        
    Raises:
        Exception: If setting alt text fails.
    """
    try:
        # Try v2 API approach first
        client = get_client()
        # Note: v2 API handles alt text differently - it's set during upload or via separate metadata endpoint
        # For now, we'll use the v1.1 approach as fallback since v2 metadata endpoint may not be fully supported in tweepy yet
        raise NotImplementedError("Using v1.1 fallback for alt text")
    except:
        # Fallback to v1.1 API for alt text (still supported)
        auth = tweepy.OAuth1UserHandler(
            consumer_key=os.getenv("X_API_KEY"),
            consumer_secret=os.getenv("X_API_SECRET"),
            access_token=os.getenv("X_ACCESS_TOKEN"),
            access_token_secret=os.getenv("X_ACCESS_TOKEN_SECRET")
        )
        api = tweepy.API(auth)
        api.create_media_metadata(media_id, alt_text={"text": alt})

def create_tweet(text: str, media_ids: List[str]) -> str:
    """
    Create a tweet with text and attached media using Twitter API v2.
    
    Args:
        text: Tweet text content.
        media_ids: List of media ID strings to attach.
        
    Returns:
        str: The ID of the created tweet.
        
    Raises:
        Exception: If tweet creation fails.
    """
    client = get_client()
    
    # Convert media_ids to integers as required by v2 API
    media_ids_int = [int(media_id) for media_id in media_ids] if media_ids else None
    
    # Create tweet using v2 API endpoint
    response = client.create_tweet(text=text, media_ids=media_ids_int)
    
    # v2 API returns response in different format
    if hasattr(response, 'data') and response.data:
        return str(response.data['id'])
    else:
        # Handle case where response format might be different
        return str(response.id) if hasattr(response, 'id') else str(response)

def get_user_info(username: Optional[str] = None) -> dict:
    """
    Get user information using Twitter API v2.
    
    Args:
        username: Twitter username (without @). If None, gets authenticated user info.
        
    Returns:
        dict: User information including id, name, username, etc.
        
    Raises:
        Exception: If user lookup fails.
    """
    client = get_client()
    
    if username:
        # Get user by username
        user = client.get_user(username=username, user_fields=['id', 'name', 'username', 'public_metrics'])
    else:
        # Get authenticated user (me)
        user = client.get_me(user_fields=['id', 'name', 'username', 'public_metrics'])
    
    if user.data:
        return {
            'id': user.data.id,
            'name': user.data.name,
            'username': user.data.username,
            'public_metrics': getattr(user.data, 'public_metrics', {})
        }
    else:
        raise Exception(f"User not found: {username}")

def get_tweet_info(tweet_id: str) -> dict:
    """
    Get tweet information using Twitter API v2.
    
    Args:
        tweet_id: The ID of the tweet to retrieve.
        
    Returns:
        dict: Tweet information including text, metrics, etc.
        
    Raises:
        Exception: If tweet lookup fails.
    """
    client = get_client()
    
    tweet = client.get_tweet(
        tweet_id, 
        tweet_fields=['created_at', 'public_metrics', 'author_id', 'text'],
        expansions=['author_id'],
        user_fields=['username', 'name']
    )
    
    if tweet.data:
        result = {
            'id': tweet.data.id,
            'text': tweet.data.text,
            'created_at': tweet.data.created_at,
            'public_metrics': getattr(tweet.data, 'public_metrics', {}),
            'author_id': tweet.data.author_id
        }
        
        # Add author info if available in includes
        if hasattr(tweet, 'includes') and tweet.includes and 'users' in tweet.includes:
            author = tweet.includes['users'][0]
            result['author'] = {
                'id': author.id,
                'username': author.username,
                'name': author.name
            }
        
        return result
    else:
        raise Exception(f"Tweet not found: {tweet_id}")
