"""
Enhanced Audio Streaming Module with Multi-Format Support
Supports MP3, WAV, FLAC, AAC, OGG, and WebM with automatic transcoding
"""

import numpy as np
from pydub import AudioSegment
from io import BytesIO
import logging
import webrtcvad
import whisper
import ffmpeg
import json

logger = logging.getLogger("AudioStreaming")
logger.setLevel(logging.INFO)

SUPPORTED_FORMATS = ['wav', 'mp3', 'ogg', 'flac', 'aac', 'webm']

class AudioStreamProcessor:
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.vad = webrtcvad.Vad(3)
        self.asr_model = whisper.load_model("base")
        self.format_cache = {}

    def process_chunk(self, raw_data: bytes, mime_type: str = None) -> dict:
        """Process audio chunk with format detection and conversion"""
        # Detect format if not provided
        format = mime_type.split('/')[1] if mime_type else self.detect_format(raw_data)
        
        # Convert to standardized format
        pcm_data = self.convert_to_pcm(raw_data, format)
        
        # Process audio
        return {
            'vad': self.detect_voice_activity(pcm_data),
            'transcript': self.transcribe_audio(pcm_data),
            'waveform': self.generate_waveform(pcm_data)
        }

    def convert_to_pcm(self, data: bytes, input_format: str) -> bytes:
        """Convert any supported format to 16-bit PCM"""
        if input_format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {input_format}")

        # Use cached converter if available
        if input_format in self.format_cache:
            return self.format_cache[input_format](data)

        # FFmpeg-based conversion pipeline
        try:
            process = (
                ffmpeg
                .input('pipe:', format=input_format)
                .output('pipe:', 
                       format='s16le',  # 16-bit little-endian PCM
                       acodec='pcm_s16le',
                       ac=self.channels,
                       ar=self.sample_rate)
                .run_async(pipe_ini=True, pipe_out=True)
            )
            
            def converter(chunk):
                process.stdin.write(chunk)
                return process.stdout.read(4096)

            self.format_cache[input_format] = converter
            return converter(data)
            
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg conversion error: {e.stderr.decode()}")
            raise

    def detect_format(self, data: bytes) -> str:
        """Auto-detect format using magic bytes"""
        if data.startswith(b'RIFF'):
            return 'wav'
        elif data.startswith(b'ID3') or data[-128:].startswith(b'TAG'):
            return 'mp3'
        elif data.startswith(b'OggS'):
            return 'ogg'
        elif data.startswith(b'fLaC'):
            return 'flac'
        else:
            # Fallback to FFmpeg probe
            try:
                info = ffmpeg.probe(BytesIO(data))
                return info['format']['format_name']
            except:
                return 'wav'  # Default assumption

    def detect_voice_activity(self, pcm_data: bytes) -> bool:
        """Voice activity detection using WebRTC VAD"""
        return self.vad.is_speech(pcm_data, self.sample_rate)

    def transcribe_audio(self, pcm_data: bytes) -> str:
        """Whisper ASR transcription with real-time optimization"""
        audio_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        result = self.asr_model.transcribe(
            audio_array,
            language='en',
            fp16=False,
            temperature=0.0  # For deterministic output
        )
        return result['text']

    def generate_waveform(self, pcm_data: bytes) -> dict:
        """Generate waveform visualization data"""
        samples = np.frombuffer(pcm_data, dtype=np.int16)
        return {
            'rms': np.sqrt(np.mean(samples**2)),
            'peaks': {
                'max': int(np.max(samples)),
                'min': int(np.min(samples))
            },
            'samples': samples[::100].tolist()  # Downsample for visualization
        }

class AudioStreamingServer:
    def __init__(self, config):
        self.processor = AudioStreamProcessor()
        self.config = config
        self.active_streams = {}

    async def handle_stream(self, websocket):
        """Handle WebSocket audio stream with format negotiation"""
        async for message in websocket:
            packet = json.loads(message)
            audio_data = bytes.fromhex(packet['data'])
            
            try:
                result = self.processor.process_chunk(
                    audio_data,
                    packet.get('format')
                )
                
                await websocket.send(json.dumps({
                    'status': 'processed',
                    'vad': result['vad'],
                    'transcript': result['transcript'],
                    'waveform': result['waveform']
                }))
                
            except Exception as e:
                await websocket.send(json.dumps({
                    'status': 'error',
                    'message': str(e)
                }))

# Usage Example
async def main():
    server = AudioStreamingServer(config={
        'max_streams': 100,
        'sample_rate': 16000,
        'channels': 1
    })
    
    # Start WebSocket server
    async with websockets.serve(server.handle_stream, "localhost", 8765):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
