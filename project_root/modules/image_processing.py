import logging
import mss
from PIL import Image
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Handles image processing tasks."""
    def capture_screen(self, bbox=None):
        """Captures a screen frame."""
        with mss.mss() as sct:
            monitor = sct.monitors[1] if not bbox else {"top": bbox[1], "left": bbox[0], "width": bbox[2] - bbox[0], "height": bbox[3] - bbox[1]}
            sct_img = sct.grab(monitor)
            return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

    def encode_image_to_base64(self, image):
        """Encodes a PIL Image to base64."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
