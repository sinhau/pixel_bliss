from PIL import Image
import math
import numpy as np

def brightness(image: Image.Image) -> float:
    """
    Calculate the average brightness of an image.
    
    Args:
        image: PIL Image object to analyze.
        
    Returns:
        float: Average brightness value (0-255 scale).
    """
    # Convert to grayscale and calculate average brightness
    gray = image.convert('L')
    return np.array(gray).mean()

def entropy(image: Image.Image) -> float:
    """
    Calculate the entropy (information content) of an image.
    
    Higher entropy indicates more visual complexity and detail.
    
    Args:
        image: PIL Image object to analyze.
        
    Returns:
        float: Entropy value (typically 0-8 for 8-bit images).
    """
    # Calculate image entropy
    gray = image.convert('L')
    hist = np.histogram(np.array(gray), bins=256, range=(0, 256))[0]
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))  # Avoid log(0)
    return entropy
