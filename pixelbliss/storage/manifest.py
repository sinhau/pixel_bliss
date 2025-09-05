import json
import os
from pathlib import Path
from typing import List, Dict

MANIFEST_PATH = "manifest/index.json"

def _load_manifest() -> List[Dict]:
    """
    Load the manifest data from the JSON file.
    
    Returns:
        List[Dict]: List of manifest entries, or empty list if file doesn't exist.
    """
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, 'r') as f:
            return json.load(f)
    return []

def _save_manifest(data: List[Dict]) -> None:
    """
    Save the manifest data to the JSON file.
    
    Args:
        data: List of manifest entries to save.
    """
    Path(MANIFEST_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(data, f, indent=2)

def append(entry: Dict) -> None:
    """
    Append a new entry to the manifest.
    
    Args:
        entry: Dictionary containing the manifest entry data.
    """
    data = _load_manifest()
    data.append(entry)
    _save_manifest(data)

def update_tweet_id(item_id: str, tweet_id: str) -> None:
    """
    Update the tweet ID for a specific manifest item.
    
    Args:
        item_id: The ID of the manifest item to update.
        tweet_id: The tweet ID to set.
    """
    data = _load_manifest()
    for item in data:
        if item.get('id') == item_id:
            item['tweet_id'] = tweet_id
            break
    _save_manifest(data)

def load_recent_hashes(limit: int = 200) -> List[str]:
    """
    Load recent perceptual hashes from the manifest for duplicate detection.
    
    Args:
        limit: Maximum number of recent entries to consider. Defaults to 200.
        
    Returns:
        List[str]: List of perceptual hash strings from recent entries.
    """
    data = _load_manifest()
    hashes = [item.get('phash') for item in data[-limit:] if item.get('phash')]
    return hashes
