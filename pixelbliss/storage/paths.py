import re
from pathlib import Path

def make_slug(category: str, base_prompt: str) -> str:
    """
    Create a filesystem-safe slug from category and base prompt.
    
    Args:
        category: The category/theme string.
        base_prompt: The base prompt string.
        
    Returns:
        str: Sanitized slug limited to 50 characters.
    """
    # Create a slug from category and base_prompt
    combined = f"{category}_{base_prompt}"
    slug = re.sub(r'[^\w\-_]', '_', combined)  # Replace non-alphanumeric with _
    slug = re.sub(r'_+', '_', slug)  # Collapse multiple _
    return slug[:50]  # Limit length

def output_dir(date_str: str, slug: str) -> str:
    """
    Generate the output directory path for a given date and slug.
    
    Args:
        date_str: Date string in YYYY-MM-DD format.
        slug: Filesystem-safe slug string.
        
    Returns:
        str: Output directory path.
    """
    return f"outputs/{date_str}/{slug}"
