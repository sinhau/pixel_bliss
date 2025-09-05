from PIL import Image
import math
import numpy as np

def brightness(image: Image.Image) -> float:
    # Convert to grayscale and calculate average brightness
    gray = image.convert('L')
    return np.array(gray).mean()

def entropy(image: Image.Image) -> float:
    # Calculate image entropy
    gray = image.convert('L')
    hist = np.histogram(np.array(gray), bins=256, range=(0, 256))[0]
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))  # Avoid log(0)
    return entropy
