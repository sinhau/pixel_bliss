from PIL import Image
import imagehash

def phash_hex(image: Image.Image) -> str:
    """
    Generate a perceptual hash (pHash) of an image as a hex string.
    
    Args:
        image: PIL Image object to hash.
        
    Returns:
        str: Hexadecimal string representation of the perceptual hash.
    """
    phash = imagehash.phash(image)
    return str(phash)

def is_duplicate(phash_hex_str: str, recent_hashes: list, distance_min: int) -> bool:
    """
    Check if an image is a duplicate based on perceptual hash comparison.
    
    Args:
        phash_hex_str: Hexadecimal string of the image's perceptual hash.
        recent_hashes: List of recent perceptual hashes to compare against.
        distance_min: Minimum Hamming distance required to consider images different.
        
    Returns:
        bool: True if the image is considered a duplicate, False otherwise.
    """
    phash = imagehash.hex_to_hash(phash_hex_str)
    for recent in recent_hashes:
        recent_phash = imagehash.hex_to_hash(recent)
        if phash - recent_phash < distance_min:
            return True
    return False
