import os
import tweepy
from typing import List, Optional
from ..imaging.compression import prepare_for_twitter_upload
from ..logging_config import get_logger

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
    logger = get_logger('twitter.upload')
    
    # Compress images if needed to stay under Twitter's 5MB limit
    processed_paths = prepare_for_twitter_upload(paths)
    
    logger.info(f"📤 TWITTER MEDIA UPLOAD - Final parameters:")
    logger.info(f"  • Original paths: {paths}")
    logger.info(f"  • Processed paths: {processed_paths}")
    
    if not processed_paths:
        logger.warning("No processed paths available for upload")
        return []
    
    # Use OAuth 2.0 client for v2 media upload
    client = get_client()
    media_ids = []
    
    for i, path in enumerate(processed_paths, 1):
        logger.info(f"  • Uploading file {i}/{len(processed_paths)}: {path}")
        try:
            # Upload media using v2 API
            media = client.media_upload(filename=path)
            media_id = str(media.media_id)
            media_ids.append(media_id)
            logger.info(f"    ✓ Upload successful - Media ID: {media_id}")
        except Exception as e:
            logger.warning(f"    ⚠ v2 API upload failed: {e}, trying v1.1 fallback")
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
            media_id = media.media_id_string
            media_ids.append(media_id)
            logger.info(f"    ✓ v1.1 fallback successful - Media ID: {media_id}")
    
    logger.info(f"📤 MEDIA UPLOAD COMPLETE - Final media IDs: {media_ids}")
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
    logger = get_logger('twitter.alt_text')
    
    logger.info(f"🏷️ TWITTER ALT TEXT - Final parameters:")
    logger.info(f"  • Media ID: {media_id}")
    logger.info(f"  • Alt text length: {len(alt)} characters")
    logger.info(f"  • Alt text content: {alt[:200]}{'...' if len(alt) > 200 else ''}")
    
    try:
        # Try v2 API approach first
        client = get_client()
        # Note: v2 API handles alt text differently - it's set during upload or via separate metadata endpoint
        # For now, we'll use the v1.1 approach as fallback since v2 metadata endpoint may not be fully supported in tweepy yet
        raise NotImplementedError("Using v1.1 fallback for alt text")
    except:
        logger.info("  • Using v1.1 API fallback for alt text")
        # Fallback to v1.1 API for alt text (still supported)
        auth = tweepy.OAuth1UserHandler(
            consumer_key=os.getenv("X_API_KEY"),
            consumer_secret=os.getenv("X_API_SECRET"),
            access_token=os.getenv("X_ACCESS_TOKEN"),
            access_token_secret=os.getenv("X_ACCESS_TOKEN_SECRET")
        )
        api = tweepy.API(auth)
        api.create_media_metadata(media_id, alt_text={"text": alt})
        logger.info(f"🏷️ ALT TEXT SET SUCCESSFULLY for media ID: {media_id}")

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
    logger = get_logger('twitter.tweet')
    
    logger.info(f"🐦 TWITTER TWEET CREATION - Final parameters:")
    logger.info(f"  • Tweet text length: {len(text)} characters")
    logger.info(f"  • Tweet text content: {text[:280]}{'...' if len(text) > 280 else ''}")
    logger.info(f"  • Media IDs (strings): {media_ids}")
    
    client = get_client()
    
    # Convert media_ids to integers as required by v2 API
    media_ids_int = [int(media_id) for media_id in media_ids] if media_ids else None
    logger.info(f"  • Media IDs (integers): {media_ids_int}")
    
    logger.info("  • Sending tweet to Twitter API v2...")
    
    # Create tweet using v2 API endpoint
    response = client.create_tweet(text=text, media_ids=media_ids_int)
    
    # v2 API returns response in different format
    if hasattr(response, 'data') and response.data:
        tweet_id = str(response.data['id'])
        logger.info(f"🐦 TWEET CREATED SUCCESSFULLY - Tweet ID: {tweet_id}")
        return tweet_id
    else:
        # Handle case where response format might be different
        tweet_id = str(response.id) if hasattr(response, 'id') else str(response)
        logger.info(f"🐦 TWEET CREATED SUCCESSFULLY - Tweet ID: {tweet_id}")
        return tweet_id

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
