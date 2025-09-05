from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict
import math
import os

def create_candidates_collage(candidates: List[Dict], max_width: int = 1920, max_height: int = 1080) -> Image.Image:
    """
    Create a collage of all candidate images sorted by their final scores.
    
    Args:
        candidates: List of candidate dictionaries with 'image', 'final', 'aesthetic', 
                   'brightness', 'entropy' keys, sorted by final score (highest first).
        max_width: Maximum width of the collage in pixels.
        max_height: Maximum height of the collage in pixels.
        
    Returns:
        Image.Image: PIL Image containing the collage with scores overlaid.
    """
    if not candidates:
        # Return a blank image if no candidates
        return Image.new('RGB', (max_width, max_height), color='black')
    
    num_images = len(candidates)
    
    # Calculate grid dimensions - try to make it roughly square
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)
    
    # Calculate individual image size to fit within the collage
    img_width = max_width // cols
    img_height = max_height // rows
    
    # Create the collage canvas
    collage = Image.new('RGB', (max_width, max_height), color='black')
    
    # Try to load a font for text overlay
    try:
        # Try to use a system font
        font_size = max(12, min(img_width // 15, img_height // 20))
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
    except (OSError, IOError):
        try:
            # Fallback to default font
            font = ImageFont.load_default()
        except:
            font = None
    
    for idx, candidate in enumerate(candidates):
        # Calculate position in grid
        col = idx % cols
        row = idx // cols
        
        # Calculate position in collage
        x = col * img_width
        y = row * img_height
        
        # Resize candidate image to fit in grid cell
        img = candidate['image'].copy()
        img.thumbnail((img_width - 10, img_height - 60), Image.LANCZOS)  # Leave space for text
        
        # Center the image in its cell
        img_x = x + (img_width - img.width) // 2
        img_y = y + (img_height - img.height - 50) // 2  # Leave space at bottom for text
        
        # Paste the image
        collage.paste(img, (img_x, img_y))
        
        # Add score overlay
        if font:
            draw = ImageDraw.Draw(collage)
            
            # Prepare score text
            final_score = candidate.get('final', 0)
            aesthetic_score = candidate.get('aesthetic', 0)
            brightness_score = candidate.get('brightness', 0)
            entropy_score = candidate.get('entropy', 0)
            
            # Create score text
            score_text = f"#{idx + 1} Final: {final_score:.3f}"
            detail_text = f"A:{aesthetic_score:.2f} B:{brightness_score:.1f} E:{entropy_score:.2f}"
            
            # Position text at bottom of cell
            text_y = y + img_height - 45
            detail_y = y + img_height - 25
            
            # Draw text with black outline for visibility
            text_x = x + 5
            
            # Draw outline
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:
                        draw.text((text_x + dx, text_y + dy), score_text, font=font, fill='black')
                        draw.text((text_x + dx, detail_y + dy), detail_text, font=font, fill='black')
            
            # Draw main text
            draw.text((text_x, text_y), score_text, font=font, fill='white')
            draw.text((text_x, detail_y), detail_text, font=font, fill='white')
            
            # Add ranking indicator (crown for #1)
            if idx == 0:
                # Draw a simple crown symbol for the winner
                crown_x = x + img_width - 30
                crown_y = y + 5
                draw.text((crown_x, crown_y), "👑", font=font, fill='gold')
    
    return collage

def save_collage(candidates: List[Dict], output_dir: str, filename: str = "candidates_collage.jpg") -> str:
    """
    Create and save a candidates collage to the specified directory.
    
    Args:
        candidates: List of candidate dictionaries sorted by final score.
        output_dir: Directory to save the collage in.
        filename: Name of the collage file.
        
    Returns:
        str: Path to the saved collage file.
    """
    # Sort candidates by final score (highest first)
    sorted_candidates = sorted(candidates, key=lambda x: x.get('final', 0), reverse=True)
    
    # Create the collage
    collage = create_candidates_collage(sorted_candidates)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the collage
    collage_path = os.path.join(output_dir, filename)
    collage.save(collage_path, 'JPEG', quality=85)
    
    return collage_path
