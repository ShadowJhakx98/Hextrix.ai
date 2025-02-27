import logging
from .ocr_module import OCR
from .gesture_recognition import GestureRecognition
from .audio_streaming import AudioStreamHandler
from .image_processing import ImageProcessor

logger = logging.getLogger(__name__)

class MultimodalManager:
    """Manages multimodal input/output processing."""
    def __init__(self, ocr: OCR, gesture_recognition: GestureRecognition, image_processor: ImageProcessor):
        self.ocr = ocr
        self.gesture_recognition = gesture_recognition
        self.image_processor = image_processor

    def process_input(self, mode: str, **kwargs):
        """Process multimodal inputs like gestures, images, or audio."""
        if mode == "gesture":
            return self.gesture_recognition.detect_gesture()
        elif mode == "image":
            image_path = kwargs.get("image_path")
            return self.ocr.extract_text(image_path)
        else:
            raise ValueError(f"Unknown multimodal mode: {mode}")
