"""
Twitter/X API integration module.

This module provides functionality for posting to Twitter/X, including:
- Creating authenticated API clients
- Uploading media files
- Setting alt text for accessibility
- Creating tweets with media attachments
"""

from . import client

# Expose client functions at module level for backward compatibility
from .client import (
    get_client,
    upload_media,
    set_alt_text,
    create_tweet
)

__all__ = [
    'client',
    'get_client',
    'upload_media', 
    'set_alt_text',
    'create_tweet'
]
