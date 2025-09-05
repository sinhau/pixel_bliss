from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict
import os


def add_candidate_number_to_image(image: Image.Image, candidate_number: int, font_size: int = None) -> Image.Image:
    """
    Add a candidate number overlay to an image.
    
    Args:
        image: PIL Image to add number to
        candidate_number: The candidate number to display (1-based)
        font_size: Optional font size. If None, will be calculated based on image size
        
    Returns:
        Image.Image: New PIL Image with the candidate number overlaid
    """
    # Create a copy to avoid modifying the original
    img_copy = image.copy()
    
    # Calculate font size based on image dimensions if not provided
    if font_size is None:
        # Use a font size that's proportional to the image size
        min_dimension = min(img_copy.size)
        font_size = max(24, min_dimension // 15)  # Minimum 24px, scale with image size
    
    # Ensure font size is positive
    font_size = max(1, font_size)
    
    # Try to load a system font
    font = None
    font_paths = [
        "/System/Library/Fonts/Arial.ttf",  # macOS
        "/System/Library/Fonts/Helvetica.ttc",  # macOS alternative
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
        "/Windows/Fonts/arial.ttf",  # Windows
        "/Windows/Fonts/calibri.ttf",  # Windows alternative
    ]
    
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        except (OSError, IOError):
            continue
    
    # Fallback to default font if no system font found
    if font is None:
        try:
            font = ImageFont.load_default()
        except:
            # If even default font fails, we'll draw without font
            font = None
    
    # Create drawing context
    draw = ImageDraw.Draw(img_copy)
    
    # Prepare the number text
    number_text = str(candidate_number)
    
    # Calculate text size and position
    if font:
        # Get text bounding box
        bbox = draw.textbbox((0, 0), number_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    else:
        # Estimate text size without font
        text_width = len(number_text) * 10
        text_height = 15
    
    # Position in top-left corner with some padding
    padding = max(10, font_size // 3)
    text_x = padding
    text_y = padding
    
    # Create a semi-transparent background for better visibility
    bg_padding = max(5, font_size // 6)
    bg_x1 = text_x - bg_padding
    bg_y1 = text_y - bg_padding
    bg_x2 = text_x + text_width + bg_padding
    bg_y2 = text_y + text_height + bg_padding
    
    # Draw background rectangle with semi-transparent black
    overlay = Image.new('RGBA', img_copy.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(0, 0, 0, 180))
    
    # Composite the overlay onto the image
    if img_copy.mode != 'RGBA':
        img_copy = img_copy.convert('RGBA')
    img_copy = Image.alpha_composite(img_copy, overlay)
    
    # Convert back to RGB if needed
    if img_copy.mode == 'RGBA':
        rgb_img = Image.new('RGB', img_copy.size, (255, 255, 255))
        rgb_img.paste(img_copy, mask=img_copy.split()[-1])
        img_copy = rgb_img
    
    # Draw the text on the final image
    draw = ImageDraw.Draw(img_copy)
    
    # Draw text with white color for good contrast
    if font:
        draw.text((text_x, text_y), number_text, font=font, fill='white')
    else:
        draw.text((text_x, text_y), number_text, fill='white')
    
    return img_copy


def add_candidate_numbers_to_images(candidates: List[Dict]) -> List[Dict]:
    """
    Add candidate numbers to all images in a list of candidates.
    
    Args:
        candidates: List of candidate dictionaries with 'image' (PIL.Image) key
        
    Returns:
        List[Dict]: New list of candidates with numbered images
    """
    numbered_candidates = []
    
    for i, candidate in enumerate(candidates):
        # Create a copy of the candidate dict
        numbered_candidate = candidate.copy()
        
        # Add the candidate number to the image (1-based numbering)
        candidate_number = i + 1
        numbered_image = add_candidate_number_to_image(candidate['image'], candidate_number)
        numbered_candidate['image'] = numbered_image
        
        numbered_candidates.append(numbered_candidate)
    
    return numbered_candidates
