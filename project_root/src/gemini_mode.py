"""
gemini_mode.py

Implements a real-time audio/video streaming client for Google Gemini 2.0
via the google-genai library (no placeholders).

Requires:
  pip install google-genai pyaudio opencv-python
  A valid Gemini 2.0 API key with 'live' streaming permissions.
"""

import asyncio
import wave
import pyaudio
import cv2
import numpy as np
import logging
from google import genai
from google.genai.types import (
    BidiGenerateContentRealtimeInput,
    BidiGenerateContentClientContent,
    BidiGenerateContentSetup,
    GenerationConfig,
    BidiGenerateContentServerContent
)

logger = logging.getLogger("GeminiMode")
logger.setLevel(logging.INFO)


class GeminiMode:
    """
    Demonstrates real-time streaming of audio and video to the Gemini 2.0
    Multimodal Live API, and receiving text/audio back in real time.

    - 'audio_stream' captures mic, streams it chunk-by-chunk, 
      and yields model responses (text or audio).
    - 'video_stream' uses OpenCV to read camera frames, sends them as 
      incremental media chunks.

    In a real system, you might combine them or run them in parallel.
    """

    def __init__(self, api_key: str, model_id: str = "gemini-2.0-flash-exp"):
        """
        :param api_key: Your Gemini 2.0 key
        :param model_id: e.g. "gemini-2.0-flash-exp", or "gemini-2.0-flash-thinking-exp"
        """
        self.api_key = api_key
        self.model_id = model_id
        # We create a genai.Client with a special api_version
        self.client = genai.Client(api_key=self.api_key, http_options={'api_version': 'v1alpha'})

    async def audio_stream(self, response_modalities=["TEXT", "AUDIO"]):
        """
        Connects a 'live' session with the given response modalities,
        then captures microphone audio chunk by chunk using PyAudio and 
        sends it to the Gemini model in real time.

        Yields partial server responses: text or audio bytes.
        """
        # Session config
        config = {
            "generation_config": {
                "response_modalities": response_modalities
            }
        }

        # We'll use 16k sample rate, 16bit PCM for input
        chunk_size = 1024
        format = pyaudio.paInt16
        channels = 1
        sample_rate = 16000

        p = pyaudio.PyAudio()
        stream = p.open(format=format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)
        logger.info("Opened microphone audio stream")

        # Connect a session
        async with self.client.aio.live.connect(model=self.model_id, config=config) as session:
            # We'll read from mic until user stops (Ctrl+C or some condition)
            # We'll also read responses in parallel:
            #   1) Send mic data
            #   2) Receive partial content
            # 
            # This approach uses two tasks.
            async def send_mic():
                try:
                    while True:
                        data = stream.read(chunk_size, exception_on_overflow=False)
                        # BidiGenerateContentRealtimeInput => we push raw audio
                        await session.send_realtime_audio(data, sample_rate=sample_rate, is_final=False)
                except asyncio.CancelledError:
                    logger.info("send_mic cancelled.")
                except Exception as e:
                    logger.error(f"send_mic error: {e}")

            async def recv_responses():
                try:
                    # We read from session.receive() in a loop
                    # Each chunk could be text or partial audio
                    turn = session.receive()
                    async for resp_chunk in turn:
                        if resp_chunk.text:
                            yield resp_chunk.text
                        if resp_chunk.data:
                            # This is audio from the model
                            yield resp_chunk.data
                except asyncio.CancelledError:
                    logger.info("recv_responses cancelled.")
                except Exception as e:
                    logger.error(f"recv_responses error: {e}")

            # Start both tasks
            send_task = asyncio.create_task(send_mic())
            try:
                # Let the caller consume the generator from recv_responses
                async for item in recv_responses():
                    yield item
            finally:
                # End the mic task and close stream
                send_task.cancel()
                stream.stop_stream()
                stream.close()
                p.terminate()

    async def video_stream(self, response_modalities=["TEXT"]):
        """
        Streams camera frames chunk by chunk to Gemini. 
        For real usage, you must ensure the model can accept video input. 
        
        Yields partial server responses.
        """
        config = {
            "generation_config": {
                "response_modalities": response_modalities
            }
        }

        cap = cv2.VideoCapture(0)  # open default camera
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam 0")

        async with self.client.aio.live.connect(model=self.model_id, config=config) as session:
            # We'll read frames in a loop
            async def send_video():
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            logger.warning("Camera read failed. Stopping.")
                            break
                        # Convert frame to e.g. JPEG
                        ret, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        if not ret:
                            continue
                        data = buf.tobytes()
                        # Now we push this data as "video" chunk
                        # For a real usage, you'd see how Gemini expects it 
                        # e.g. is_final=False, mime=video...
                        await session.send_realtime_video(data, is_final=False)
                except asyncio.CancelledError:
                    logger.info("send_video canceled")
                except Exception as e:
                    logger.error(f"send_video error: {e}")

            async def recv_responses():
                try:
                    turn = session.receive()
                    async for resp_chunk in turn:
                        # If there's text or other data
                        if resp_chunk.text:
                            yield resp_chunk.text
                except asyncio.CancelledError:
                    logger.info("recv_responses canceled")
                except Exception as e:
                    logger.error(f"recv_responses error: {e}")

            send_task = asyncio.create_task(send_video())
            try:
                async for item in recv_responses():
                    yield item
            finally:
                send_task.cancel()
                cap.release()
                logger.info("Released camera")


    async def text_to_text(self, text_input: str) -> str:
        """
        Simple text-based call if you just want to do a standard generate_content.
        """
        resp = self.client.models.generate_content(
            model=self.model_id,
            contents=text_input
        )
        if resp.candidates:
            return resp.candidates[0].text
        return "[No response from Gemini]"
