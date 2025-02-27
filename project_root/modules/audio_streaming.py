import logging
import wave
from pathlib import Path
from IPython.display import Audio
from google.generativeai import GenerativeModel

logger = logging.getLogger(__name__)

class AudioStreamHandler:
    """Handles live audio streaming and interaction with an AI model."""
    def __init__(self, output_dir: str, sample_rate: int, channels: int, sample_width: int, model_name: str):
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.model_name = model_name
        self.index = 0

    async def send_and_receive_audio(self, session, text_message: str):
        file_name = Path(self.output_dir) / f"audio_{self.index}.wav"
        self.index += 1

        try:
            with wave.open(file_name, 'wb') as wav:
                wav.setnchannels(self.channels)
                wav.setsampwidth(self.sample_width)
                wav.setframerate(self.sample_rate)

                response_stream = session.send_message(text_message, stream=True)
                async for chunk in response_stream:
                    if chunk.parts and chunk.parts[0].inline_data.mime_type == 'audio/wav':
                        wav.writeframes(chunk.parts[0].inline_data.data)

            Audio(filename=str(file_name), autoplay=True)
            logger.info("Audio streamed and saved successfully.")
        except Exception as e:
            logger.error(f"Error during audio streaming: {e}")
            raise
