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
import tempfile
import select
import termios
import tty
import signal
import atexit
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

import pyaudio
import numpy as np
import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

# Load environment variables
load_dotenv()

# Configuration
class RecordingMode(Enum):
    VAD = "vad"  # Voice Activity Detection
    SPACE_TOGGLE = "space_toggle"  # Space Bar Toggle Recording

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
    PTT_KEY: str = ' '  # Space bar for push-to-talk


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
            from faster_whisper import WhisperModel
            print("📥 Loading faster-whisper model...")
            self.model = WhisperModel("base", device="cpu", compute_type="int8")
            self.model_loaded = True
            print("✅ faster-whisper model loaded successfully")
            return True
        except ImportError:
            print("❌ faster-whisper not installed. Install with: uv pip install faster-whisper")
            return False
        except Exception as e:
            print(f"❌ Failed to load faster-whisper model: {e}")
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
            console = Console()
            with Progress(
                SpinnerColumn(),
                TextColumn("🔄 Transcribing speech..."),
                console=console
            ) as progress:
                task = progress.add_task("Transcribing...", total=1)
                segments, info = self.model.transcribe(temp_path, language="en")
                progress.update(task, completed=1)
            
            # Extract text from segments
            text = "".join([segment.text for segment in segments]).strip()
            
            if text:
                # Use Rich to display transcribed text nicely
                console.print(
                    Panel(
                        Text(text, style="green"),
                        title="📝 Transcribed",
                        title_align="left",
                        border_style="green"
                    )
                )
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
            # Use Rich to show a nice loading indicator
            console = Console()
            with Progress(
                SpinnerColumn(),
                TextColumn("🤖 Sending to LLM..."),
                console=console
            ) as progress:
                task = progress.add_task("Sending...", total=1)
                response = self.client.post(
                    self.config.TOGETHER_API_URL,
                    json=payload,
                    headers=headers
                )
                progress.update(task, completed=1)
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


class KeyboardListener:
    """Non-blocking keyboard input handler for push-to-talk"""
    
    def __init__(self):
        self.old_settings = None
        self.is_pressed = False
        
    def setup(self):
        """Set terminal to cbreak mode for immediate key detection"""
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        
    def restore(self):
        """Restore terminal settings"""
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def check_key_event(self) -> Optional[str]:
        """Check for key events (non-blocking)"""
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            key = sys.stdin.read(1)
            return key
        return None


class VoiceLLMApp:
    """Main application coordinating all components"""
    
    def __init__(self):
        self.audio_config = AudioConfig()
        self.api_config = APIConfig()
        self.audio_capture = AudioCapture(self.audio_config)
        self.silence_detector = SilenceDetector(self.audio_config)
        self.whisper_processor = WhisperProcessor()
        self.together_client = TogetherClient(self.api_config)
        self.keyboard_listener = KeyboardListener()
        self.running = False
        self.recording_mode = RecordingMode.VAD
        self.cleanup_registered = False
        
        # Initialize Rich console
        self.console = Console()
        
        # Register signal handlers
        self.register_signal_handlers()
        
    def initialize(self) -> bool:
        """Initialize all components"""
        self.console.print("\n")
        self.console.print(
            Panel(
                Text("🚀 Initializing Voice-to-LLM Application", style="bold cyan"),
                border_style="cyan"
            )
        )
        
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
        
        # Select recording mode
        self.select_recording_mode()
        
        self.console.print(
            Panel(
                Text("✅ All components initialized successfully!", style="bold green"),
                border_style="green"
            )
        )
        self.console.print("\n📢 Instructions:", style="bold yellow")
        
        if self.recording_mode == RecordingMode.VAD:
            instructions = [
                "• Mode: Voice Activity Detection (VAD)",
                "• Speak into your microphone",
                "• Pause for 2-3 seconds when done",
                "• Wait for the LLM response",
                "• Press Ctrl+C to exit"
            ]
        else:
            instructions = [
                "• Mode: Space Bar Toggle Recording",
                "• Press SPACE to START recording",
                "• Press SPACE again to STOP and send to LLM",
                "• Wait for the LLM response",
                "• Press 'Q' or Ctrl+C to exit"
            ]
        
        self.console.print(
            Panel(
                "\n".join(instructions),
                title="Instructions",
                border_style="yellow",
                padding=(1, 2)
            )
        )
        
        return True
    
    def select_recording_mode(self):
        """Allow user to select recording mode"""
        self.console.print("\n")
        self.console.print(
            Panel(
                "1. Voice Activity Detection (VAD) - Automatic silence detection\n"
                "2. Space Bar Toggle Recording - Press SPACE to start/stop recording",
                title="🎙️ Select Recording Mode",
                border_style="magenta",
                padding=(1, 2)
            )
        )
        
        while True:
            try:
                choice = input("\nEnter choice (1 or 2): ").strip()
                if choice == "1":
                    self.recording_mode = RecordingMode.VAD
                    break
                elif choice == "2":
                    self.recording_mode = RecordingMode.SPACE_TOGGLE
                    break
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            except KeyboardInterrupt:
                print("\n\n⚠️  Setup cancelled by user.")
                self.shutdown()
                sys.exit(0)
            except EOFError:
                print("\n\n⚠️  Input stream closed.")
                self.shutdown()
                sys.exit(1)
    
    def process_audio_buffer(self, audio_buffer: List[bytes]):
        """Process captured audio: transcribe and send to LLM"""
        # Transcribe audio
        text = self.whisper_processor.transcribe(audio_buffer, self.audio_config)
        
        if not text:
            self.console.print("⚠️ No text to process", style="yellow")
            return
        
        # Send to LLM
        response = self.together_client.send_message(text)
        
        if response:
            # Create markdown object for the response
            markdown_response = Markdown(response)
            
            # Display in a panel with proper styling
            self.console.print("\n")
            self.console.print(
                Panel(
                    markdown_response,
                    title="🤖 LLM Response",
                    title_align="left",
                    border_style="blue",
                    padding=(1, 2)
                )
            )
            self.console.print("\n")
        else:
            self.console.print("⚠️ No response from LLM", style="yellow")
    
    def run(self):
        """Main application loop"""
        if not self.initialize():
            return
        
        self.running = True
        
        if self.recording_mode == RecordingMode.VAD:
            self.run_vad_mode()
        else:
            self.run_space_toggle_mode()
    
    def run_vad_mode(self):
        """Run with Voice Activity Detection mode"""
        self.console.print("🎧 Listening... (speak now)", style="bold green")
        
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
                    self.console.print("🔕 Silence detected, processing...", style="yellow")
                    
                    # Get audio buffer and process
                    audio_buffer = self.audio_capture.get_buffer_copy()
                    self.process_audio_buffer(audio_buffer)
                    
                    # Clear buffer for next recording
                    self.audio_capture.clear_buffer()
                    self.console.print("🎧 Listening... (speak now)", style="bold green")
                    
        except KeyboardInterrupt:
            pass  # Signal handler will take care of shutdown
        except Exception as e:
            print(f"\n❌ Error in VAD mode: {e}")
            self.shutdown()
    
    def run_space_toggle_mode(self):
        """Run with space bar toggle recording mode"""
        self.console.print(f"\n🎧 Ready! Press SPACE to start/stop recording...\n", style="bold green")
        
        # Setup keyboard listener
        try:
            self.keyboard_listener.setup()
        except Exception as e:
            print(f"❌ Failed to setup keyboard listener: {e}")
            return
        
        try:
            is_recording = False
            ptt_buffer = []
            
            while self.running:
                # Check for key events
                key = self.keyboard_listener.check_key_event()
                
                if key == 'q' or key == 'Q':  # Allow 'q' to quit
                    print("\n\n📤 Quit command received...")
                    break
                elif key == '\x03':  # Ctrl+C to quit
                    raise KeyboardInterrupt
                elif key == ' ':  # Space bar to toggle recording
                    # Toggle recording state
                    if not is_recording:
                        # Start recording
                        is_recording = True
                        ptt_buffer = []
                        self.console.print("\n🔴 Recording... (press SPACE again to stop)", style="bold red")
                    else:
                        # Stop recording and process
                        is_recording = False
                        self.console.print("⏹️  Recording stopped, processing...\n", style="yellow")
                        
                        if ptt_buffer:
                            self.process_audio_buffer(ptt_buffer)
                        
                        ptt_buffer = []
                        self.console.print(f"\n🎧 Ready! Press SPACE to start recording... (Press 'Q' to quit)\n", style="bold green")
                
                # Read audio chunk if available
                chunk = self.audio_capture.read_chunk()
                if chunk and is_recording:
                    ptt_buffer.append(chunk)
                
                time.sleep(0.01)
                    
        except KeyboardInterrupt:
            pass  # Signal handler will take care of shutdown
        except Exception as e:
            print(f"\n❌ Error in PTT mode: {e}")
        finally:
            # Ensure terminal is restored even if error occurs
            try:
                self.keyboard_listener.restore()
            except:
                pass
    
    def register_signal_handlers(self):
        """Register signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            if hasattr(self, 'console'):
                self.console.print("\n\n⚠️  Interrupt received, shutting down gracefully...", style="yellow")
            else:
                print("\n\n⚠️  Interrupt received, shutting down gracefully...")
            self.shutdown()
            sys.exit(0)
        
        # Register handlers for common termination signals
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
        
        # Register cleanup at exit
        atexit.register(self.cleanup)
    
    def shutdown(self):
        """Initiate graceful shutdown"""
        if hasattr(self, 'console'):
            self.console.print("🛑 Stopping recording...", style="yellow")
        else:
            print("🛑 Stopping recording...")
        self.running = False
        
        # Give threads time to finish
        time.sleep(0.5)
        
        # Perform cleanup
        self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        if self.cleanup_registered:
            return  # Avoid double cleanup
        
        self.cleanup_registered = True
        self.running = False
        
        try:
            # Restore terminal settings first if in space toggle mode
            if self.recording_mode == RecordingMode.SPACE_TOGGLE and self.keyboard_listener.old_settings:
                self.keyboard_listener.restore()
                if hasattr(self, 'console'):
                    self.console.print("✅ Terminal settings restored", style="green")
                else:
                    print("✅ Terminal settings restored")
            
            # Stop audio capture
            if hasattr(self, 'audio_capture') and self.audio_capture.stream:
                self.audio_capture.stop()
                if hasattr(self, 'console'):
                    self.console.print("✅ Audio capture stopped", style="green")
                else:
                    print("✅ Audio capture stopped")
            
            # Close API client
            if hasattr(self, 'together_client'):
                self.together_client.close()
                if hasattr(self, 'console'):
                    self.console.print("✅ API client closed", style="green")
                else:
                    print("✅ API client closed")
                
        except Exception as e:
            if hasattr(self, 'console'):
                self.console.print(f"⚠️  Error during cleanup: {e}", style="red")
            else:
                print(f"⚠️  Error during cleanup: {e}")
        
        if hasattr(self, 'console'):
            self.console.print(
                Panel(
                    Text("✅ Cleanup complete. Goodbye!", style="bold green"),
                    border_style="green"
                )
            )
        else:
            print("✅ Cleanup complete. Goodbye!")


def main():
    """Entry point"""
    app = None
    try:
        app = VoiceLLMApp()
        app.run()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if app:
            app.shutdown()
        sys.exit(1)
    except SystemExit:
        # Let system exit normally
        raise
    finally:
        # Ensure cleanup happens
        if app and not app.cleanup_registered:
            app.cleanup()


if __name__ == "__main__":
    main()