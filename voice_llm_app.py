#!/usr/bin/env python3
"""
Voice-to-LLM Application MVP
Captures speech, detects silence, transcribes with Whisper, and sends to Together.ai LLM
"""

import os
import sys
import time

# Check Python version requirement
if sys.version_info[:2] != (3, 9):
    print("❌ ERROR: This application requires Python 3.9 exactly")
    print(f"   Current version: Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print("\n💡 To install Python 3.9:")
    print("   Using pyenv: pyenv install 3.9.18 && pyenv local 3.9.18")
    print("   Using conda: conda create -n voice-llm python=3.9 && conda activate voice-llm")
    print("   Or download from: https://www.python.org/downloads/release/python-3918/")
    sys.exit(1)
import wave
import threading
import queue
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass
from contextlib import contextmanager

import pyaudio
import numpy as np
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
@dataclass
class AudioConfig:
    CHUNK_SIZE: int = 1024
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    AUDIO_FORMAT: int = pyaudio.paInt16
    SILENCE_THRESHOLD: int = 500
    SILENCE_DURATION: float = 2.5
    MIN_AUDIO_LENGTH: float = 1.0
    BYTES_PER_SAMPLE: int = 2


@dataclass
class APIConfig:
    TOGETHER_API_URL: str = "https://api.together.xyz/v1/chat/completions"
    DEFAULT_MODEL: str = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
    TOGETHER_API_KEY: str = os.getenv("TOGETHER_API_KEY", "")
    MAX_TOKENS: int = 512
    TEMPERATURE: float = 0.7


class AudioCapture:
    """Handles microphone input and audio buffering"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        
    def start(self):
        """Start audio capture stream"""
        try:
            self.stream = self.audio.open(
                format=self.config.AUDIO_FORMAT,
                channels=self.config.CHANNELS,
                rate=self.config.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.config.CHUNK_SIZE
            )
            self.is_recording = True
            print("🎤 Microphone initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize microphone: {e}")
            return False
    
    def read_chunk(self) -> Optional[bytes]:
        """Read a single audio chunk from the stream"""
        if not self.stream or not self.is_recording:
            return None
        try:
            return self.stream.read(self.config.CHUNK_SIZE, exception_on_overflow=False)
        except Exception as e:
            print(f"⚠️ Error reading audio chunk: {e}")
            return None
    
    def add_to_buffer(self, chunk: bytes):
        """Add audio chunk to buffer thread-safely"""
        with self.buffer_lock:
            self.audio_buffer.append(chunk)
    
    def get_buffer_copy(self) -> List[bytes]:
        """Get a copy of the current buffer"""
        with self.buffer_lock:
            return self.audio_buffer.copy()
    
    def clear_buffer(self):
        """Clear the audio buffer"""
        with self.buffer_lock:
            self.audio_buffer.clear()
    
    def stop(self):
        """Stop audio capture and cleanup"""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("🔇 Microphone stopped")


class SilenceDetector:
    """Monitors audio levels and detects silence periods"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.silence_start = None
        self.is_speaking = False
        
    def calculate_rms(self, audio_chunk: bytes) -> float:
        """Calculate root mean square (RMS) of audio chunk"""
        try:
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            if len(audio_array) == 0:
                return 0
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            return rms
        except Exception as e:
            print(f"⚠️ Error calculating RMS: {e}")
            return 0
    
    def process_chunk(self, audio_chunk: bytes) -> Tuple[bool, bool]:
        """
        Process audio chunk and detect silence
        Returns: (is_silent, silence_detected_for_duration)
        """
        rms = self.calculate_rms(audio_chunk)
        is_silent = rms < self.config.SILENCE_THRESHOLD
        
        if not is_silent:
            # Speech detected
            self.silence_start = None
            if not self.is_speaking:
                self.is_speaking = True
                print("🗣️ Speech detected...")
            return False, False
        
        # Silence detected
        current_time = time.time()
        
        if self.silence_start is None:
            self.silence_start = current_time
            return True, False
        
        # Check if silence duration exceeded threshold
        silence_duration = current_time - self.silence_start
        if silence_duration >= self.config.SILENCE_DURATION and self.is_speaking:
            self.is_speaking = False
            self.silence_start = None
            return True, True
        
        return True, False


class WhisperProcessor:
    """Handles speech-to-text conversion using Whisper"""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        
    def load_model(self):
        """Load Whisper model"""
        try:
            import whisper
            print("📥 Loading Whisper model...")
            self.model = whisper.load_model("base")
            self.model_loaded = True
            print("✅ Whisper model loaded successfully")
            return True
        except ImportError:
            print("❌ Whisper not installed. Install with: uv pip install openai-whisper")
            return False
        except Exception as e:
            print(f"❌ Failed to load Whisper model: {e}")
            return False
    
    def transcribe(self, audio_buffer: List[bytes], config: AudioConfig) -> Optional[str]:
        """Transcribe audio buffer to text"""
        if not self.model_loaded:
            print("⚠️ Whisper model not loaded")
            return None
        
        # Check minimum audio length
        total_samples = len(audio_buffer) * config.CHUNK_SIZE
        duration = total_samples / config.SAMPLE_RATE
        
        if duration < config.MIN_AUDIO_LENGTH:
            print(f"⏱️ Audio too short ({duration:.1f}s), skipping transcription")
            return None
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            # Write WAV file
            with wave.open(temp_path, 'wb') as wav_file:
                wav_file.setnchannels(config.CHANNELS)
                wav_file.setsampwidth(config.BYTES_PER_SAMPLE)
                wav_file.setframerate(config.SAMPLE_RATE)
                wav_file.writeframes(b''.join(audio_buffer))
            
            # Transcribe
            print("🔄 Transcribing speech...")
            result = self.model.transcribe(temp_path, language="en")
            text = result["text"].strip()
            
            if text:
                print(f"📝 Transcribed: {text}")
                return text
            else:
                print("⚠️ No speech detected in audio")
                return None
                
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None
        finally:
            # Cleanup temp file
            try:
                os.unlink(temp_path)
            except:
                pass


class TogetherClient:
    """Manages LLM API calls to Together.ai"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.client = httpx.Client(timeout=30.0)
        
    def send_message(self, text: str) -> Optional[str]:
        """Send text to Together.ai LLM and get response"""
        if not self.config.TOGETHER_API_KEY:
            print("❌ TOGETHER_API_KEY not set in environment variables")
            return None
        
        headers = {
            "Authorization": f"Bearer {self.config.TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.DEFAULT_MODEL,
            "messages": [
                {"role": "user", "content": text}
            ],
            "max_tokens": self.config.MAX_TOKENS,
            "temperature": self.config.TEMPERATURE,
            "stream": False
        }
        
        try:
            print("🤖 Sending to LLM...")
            response = self.client.post(
                self.config.TOGETHER_API_URL,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                llm_response = data["choices"][0]["message"]["content"]
                return llm_response
            else:
                print("⚠️ Unexpected response format from LLM")
                return None
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                print("❌ Invalid API key. Please check your TOGETHER_API_KEY")
            else:
                print(f"❌ HTTP error: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            print(f"❌ LLM API error: {e}")
            return None
    
    def close(self):
        """Close HTTP client"""
        self.client.close()


class VoiceLLMApp:
    """Main application coordinating all components"""
    
    def __init__(self):
        self.audio_config = AudioConfig()
        self.api_config = APIConfig()
        self.audio_capture = AudioCapture(self.audio_config)
        self.silence_detector = SilenceDetector(self.audio_config)
        self.whisper_processor = WhisperProcessor()
        self.together_client = TogetherClient(self.api_config)
        self.running = False
        
    def initialize(self) -> bool:
        """Initialize all components"""
        print("\n🚀 Initializing Voice-to-LLM Application...")
        print("=" * 50)
        
        # Check API key
        if not self.api_config.TOGETHER_API_KEY:
            print("❌ Error: TOGETHER_API_KEY environment variable not set")
            print("Please set it using: export TOGETHER_API_KEY='your_key_here'")
            return False
        
        # Initialize audio
        if not self.audio_capture.start():
            return False
        
        # Load Whisper model
        if not self.whisper_processor.load_model():
            return False
        
        print("=" * 50)
        print("✅ All components initialized successfully!")
        print("\n📢 Instructions:")
        print("  • Speak into your microphone")
        print("  • Pause for 2-3 seconds when done")
        print("  • Wait for the LLM response")
        print("  • Press Ctrl+C to exit\n")
        print("=" * 50)
        
        return True
    
    def process_audio_buffer(self, audio_buffer: List[bytes]):
        """Process captured audio: transcribe and send to LLM"""
        # Transcribe audio
        text = self.whisper_processor.transcribe(audio_buffer, self.audio_config)
        
        if not text:
            print("⚠️ No text to process")
            return
        
        # Send to LLM
        response = self.together_client.send_message(text)
        
        if response:
            print("\n" + "=" * 50)
            print("🤖 LLM Response:")
            print("-" * 50)
            print(response)
            print("=" * 50 + "\n")
        else:
            print("⚠️ No response from LLM")
    
    def run(self):
        """Main application loop"""
        if not self.initialize():
            return
        
        self.running = True
        print("🎧 Listening... (speak now)")
        
        try:
            while self.running:
                # Read audio chunk
                chunk = self.audio_capture.read_chunk()
                if chunk is None:
                    time.sleep(0.01)
                    continue
                
                # Add to buffer
                self.audio_capture.add_to_buffer(chunk)
                
                # Check for silence
                is_silent, trigger_processing = self.silence_detector.process_chunk(chunk)
                
                if trigger_processing:
                    print("🔕 Silence detected, processing...")
                    
                    # Get audio buffer and process
                    audio_buffer = self.audio_capture.get_buffer_copy()
                    self.process_audio_buffer(audio_buffer)
                    
                    # Clear buffer for next recording
                    self.audio_capture.clear_buffer()
                    print("🎧 Listening... (speak now)")
                    
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        self.audio_capture.stop()
        self.together_client.close()
        print("✅ Cleanup complete. Goodbye!")


def main():
    """Entry point"""
    app = VoiceLLMApp()
    app.run()


if __name__ == "__main__":
    main()