from PIL import Image
import imagehash

def phash_hex(image: Image.Image) -> str:
    phash = imagehash.phash(image)
    return str(phash)

def is_duplicate(phash_hex_str: str, recent_hashes: list, distance_min: int) -> bool:
    phash = imagehash.hex_to_hash(phash_hex_str)
    for recent in recent_hashes:
        recent_phash = imagehash.hex_to_hash(recent)
        if phash - recent_phash < distance_min:
            return True
    return False
