import json
import os
from pathlib import Path
from typing import List, Dict

MANIFEST_PATH = "manifest/index.json"

def _load_manifest() -> List[Dict]:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, 'r') as f:
            return json.load(f)
    return []

def _save_manifest(data: List[Dict]) -> None:
    Path(MANIFEST_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(data, f, indent=2)

def append(entry: Dict) -> None:
    data = _load_manifest()
    data.append(entry)
    _save_manifest(data)

def update_tweet_id(item_id: str, tweet_id: str) -> None:
    data = _load_manifest()
    for item in data:
        if item.get('id') == item_id:
            item['tweet_id'] = tweet_id
            break
    _save_manifest(data)

def load_recent_hashes(limit: int = 200) -> List[str]:
    data = _load_manifest()
    hashes = [item.get('phash') for item in data[-limit:] if item.get('phash')]
    return hashes
