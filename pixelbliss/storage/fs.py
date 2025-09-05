import os
import json
from pathlib import Path
from typing import Dict
from PIL import Image

def save_images(dir_path: str, images: Dict[str, Image.Image]) -> Dict[str, str]:
    """
    Save multiple images to a directory with PNG format.
    
    Args:
        dir_path: Directory path to save images to.
        images: Dictionary mapping image names to PIL Image objects.
        
    Returns:
        Dict[str, str]: Dictionary mapping image names to their public paths.
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    public_paths = {}
    for name, img in images.items():
        path = os.path.join(dir_path, f"{name}.png")
        img.save(path, 'PNG')
        public_paths[name] = f"/{path}"
    return public_paths

def save_meta(dir_path: str, meta: Dict) -> str:
    """
    Save metadata as a JSON file in the specified directory.
    
    Args:
        dir_path: Directory path to save the metadata file to.
        meta: Dictionary containing metadata to save.
        
    Returns:
        str: Path to the saved metadata file.
    """
    path = os.path.join(dir_path, "meta.json")
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)
    return path
