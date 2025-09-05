import os
import json
from pathlib import Path
from typing import Dict
from PIL import Image

def save_images(dir_path: str, images: Dict[str, Image.Image]) -> Dict[str, str]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    public_paths = {}
    for name, img in images.items():
        path = os.path.join(dir_path, f"{name}.png")
        img.save(path, 'PNG')
        public_paths[name] = f"/{path}"
    return public_paths

def save_meta(dir_path: str, meta: Dict) -> str:
    path = os.path.join(dir_path, "meta.json")
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)
    return path
