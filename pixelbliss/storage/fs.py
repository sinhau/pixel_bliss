import os
import json
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image

def save_images(dir_path: str, images: Dict[str, Image.Image], base_img: Image.Image = None) -> Dict[str, str]:
    """
    Save multiple images to a directory with PNG format.
    
    Args:
        dir_path: Directory path to save images to.
        images: Dictionary mapping image names to PIL Image objects.
        base_img: Optional base image to save alongside variants.
        
    Returns:
        Dict[str, str]: Dictionary mapping image names to their public paths.
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    public_paths = {}
    
    # Save the base image if provided
    if base_img is not None:
        base_path = os.path.join(dir_path, "base_img.png")
        base_img.save(base_path, 'PNG')
        public_paths["base_img"] = f"/{base_path}"
    
    # Save all variant images
    for name, img in images.items():
        path = os.path.join(dir_path, f"{name}.png")
        img.save(path, 'PNG')
        public_paths[name] = f"/{path}"
    return public_paths

def save_candidate_images(dir_path: str, candidates: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Save all original candidate images to a 'candidates' subfolder.
    
    Args:
        dir_path: Base directory path to save images to.
        candidates: List of candidate dictionaries containing 'image' and other metadata.
        
    Returns:
        Dict[str, str]: Dictionary mapping candidate indices to their public paths.
    """
    candidates_dir = os.path.join(dir_path, "candidates")
    Path(candidates_dir).mkdir(parents=True, exist_ok=True)
    
    candidate_paths = {}
    
    for i, candidate in enumerate(candidates):
        if 'image' in candidate:
            filename = f"candidate_{i+1:03d}.png"
            path = os.path.join(candidates_dir, filename)
            candidate['image'].save(path, 'PNG')
            candidate_paths[f"candidate_{i+1:03d}"] = f"/{path}"
    
    return candidate_paths

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
