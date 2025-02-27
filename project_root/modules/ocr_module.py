import logging
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)

class OCRError(Exception):
    """Custom exception for OCR errors."""
    pass

class OCR:
    """Handles Optical Character Recognition."""
    async def extract_text(self, image_file: str) -> str:
        """Extracts text from an image using pytesseract."""
        try:
            logger.info(f"Extracting text from image: {image_file}")
            text = pytesseract.image_to_string(Image.open(image_file))
            logger.info(f"Text extracted successfully from {image_file}")
            return text.strip()
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_file}")
            raise OCRError(f"Image file not found: {image_file}")
        except Exception as e:
            logger.error(f"Error during OCR: {e}")
            raise OCRError(f"OCR failed: {e}")
