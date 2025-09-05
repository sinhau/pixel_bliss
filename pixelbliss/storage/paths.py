import re
from pathlib import Path

def make_slug(category: str, base_prompt: str) -> str:
    # Create a slug from category and base_prompt
    combined = f"{category}_{base_prompt}"
    slug = re.sub(r'[^\w\-_]', '_', combined)  # Replace non-alphanumeric with _
    slug = re.sub(r'_+', '_', slug)  # Collapse multiple _
    return slug[:50]  # Limit length

def output_dir(date_str: str, slug: str) -> str:
    return f"outputs/{date_str}/{slug}"
