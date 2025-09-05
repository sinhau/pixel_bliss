from PIL import Image
import requests
from io import BytesIO

def aesthetic(image: Image.Image) -> float:
    try:
        # Hypothetical hosted aesthetic scorer API
        # Assume it accepts image upload and returns score
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)

        response = requests.post(
            "https://aesthetic-scorer.example.com/score",  # Placeholder URL
            files={'image': buffer}
        )
        if response.status_code == 200:
            score = response.json().get('score', 0.5)
            return min(max(score, 0.0), 1.0)  # Clamp to [0,1]
        else:
            return 0.5
    except Exception as e:
        print(f"Aesthetic scoring failed: {e}")
        return 0.5
