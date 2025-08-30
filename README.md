# Voice-to-LLM Application

A **terminal-only** Python application that captures speech from your microphone, detects when you stop speaking, transcribes it using OpenAI Whisper, and sends it to Together.ai's LLM for intelligent responses.

## Features

- 🎤 Real-time audio capture from microphone
- 🔕 Automatic silence detection (2-3 seconds pause triggers processing)
- 📝 Speech-to-text using OpenAI Whisper (base model)
- 🤖 LLM integration with Together.ai (Llama 3 by default)
- 💻 **Pure terminal/command-line interface - no GUI**
- ⚡ Continuous operation - speak, get response, repeat

## Prerequisites

### ⚠️ Python Version Requirement
**This application requires Python 3.9 exactly.** Other versions will not work.

Install Python 3.9:
```bash
# Using pyenv (recommended)
pyenv install 3.9.18
pyenv local 3.9.18

# Using conda
conda create -n voice-llm python=3.9
conda activate voice-llm

# Or download from: https://www.python.org/downloads/release/python-3918/
```

### System Dependencies

**macOS:**
```bash
brew install portaudio ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev ffmpeg python3-dev
```

**Windows:**
- Download and install [PortAudio](http://www.portaudio.com/download.html)
- Download and install [FFmpeg](https://ffmpeg.org/download.html)

### API Key

You need a Together.ai API key:
1. Sign up at [Together.ai](https://api.together.xyz/)
2. Get your API key from the dashboard
3. Set it as an environment variable (see Setup below)

## Installation

### Option 1: Using UV (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd speech-brain

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Install Whisper separately (due to Python version constraints)
uv pip install openai-whisper
```

### Option 2: Using pip

```bash
# Clone the repository
git clone <repository-url>
cd speech-brain

# Create virtual environment (requires Python 3.9)
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install openai-whisper
```

## Setup

1. **Configure API Key:**

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Together.ai API key
# Or set it directly:
export TOGETHER_API_KEY="your_api_key_here"
```

2. **Test microphone access:**

```bash
# Quick test to ensure microphone works
python -c "import pyaudio; p = pyaudio.PyAudio(); print(f'Found {p.get_device_count()} audio devices')"
```

## Usage

### Running the Application

```bash
# With uv (recommended)
uv run python voice_llm_app.py

# With virtual environment (if using pip)
source venv/bin/activate  # On Windows: venv\Scripts\activate
python voice_llm_app.py
```

**Important**: Always use the same Python environment where you installed the dependencies. If you used `uv`, use `uv run`. If you used pip with a virtual environment, activate it first.

### How to Use

1. **Start the application** - You'll see initialization messages
2. **Speak clearly** into your microphone when you see "🎧 Listening..."
3. **Pause for 2-3 seconds** when you're done speaking
4. **Wait for transcription** - The app will show your transcribed text
5. **Receive LLM response** - The response will appear in the terminal
6. **Continue conversation** - The app automatically starts listening again
7. **Exit** - Press `Ctrl+C` to stop the application

### Example Session

```
🚀 Initializing Voice-to-LLM Application...
==================================================
📥 Loading Whisper model...
✅ Whisper model loaded successfully
🎤 Microphone initialized successfully
==================================================
✅ All components initialized successfully!

📢 Instructions:
  • Speak into your microphone
  • Pause for 2-3 seconds when done
  • Wait for the LLM response
  • Press Ctrl+C to exit

==================================================
🎧 Listening... (speak now)
🗣️ Speech detected...
🔕 Silence detected, processing...
🔄 Transcribing speech...
📝 Transcribed: What is the capital of France?
🤖 Sending to LLM...

==================================================
🤖 LLM Response:
--------------------------------------------------
The capital of France is Paris.
==================================================

🎧 Listening... (speak now)
```

## Configuration

You can customize behavior by modifying values in `voice_llm_app.py`:

```python
# Audio settings
CHUNK_SIZE = 1024          # Audio buffer size
SAMPLE_RATE = 16000        # 16kHz for Whisper compatibility
SILENCE_THRESHOLD = 500    # Adjust based on your environment
SILENCE_DURATION = 2.5     # Seconds of silence to trigger processing
MIN_AUDIO_LENGTH = 1.0     # Minimum audio length to process

# LLM settings
DEFAULT_MODEL = "meta-llama/Llama-3-8b-chat-hf"
MAX_TOKENS = 512
TEMPERATURE = 0.7
```

## Running Tests

```bash
# With uv
uv run python -m pytest test_voice_llm.py -v

# Or with unittest
uv run python test_voice_llm.py

# Standard Python
python -m unittest test_voice_llm.py
```

## Troubleshooting

### "Microphone not found" Error
- Ensure your microphone is connected and recognized by the system
- Check microphone permissions in system settings
- Try listing devices: `python -c "import pyaudio; p = pyaudio.PyAudio(); print([p.get_device_info_by_index(i)['name'] for i in range(p.get_device_count())])"`

### "TOGETHER_API_KEY not set" Error
- Make sure you've set the environment variable: `export TOGETHER_API_KEY="your_key"`
- Or create a `.env` file with your key

### "Whisper model failed to load" Error
- Ensure you have enough disk space (~1GB for base model)
- Check internet connection (first run downloads the model)
- Try manually downloading: `python -c "import whisper; whisper.load_model('base')"`

### Audio Too Quiet/Loud
- Adjust `SILENCE_THRESHOLD` in the code (lower = more sensitive)
- Test with different microphone positions
- Check system audio input levels

### Python Version Issues
- **This application requires Python 3.9 exactly**
- Check version: `python --version`
- If wrong version: Install Python 3.9 using pyenv, conda, or direct download
- The app will exit with clear error message if wrong Python version is detected

## Project Structure

```
speech-brain/
├── voice_llm_app.py       # Main application
├── test_voice_llm.py      # Unit tests
├── requirements.txt       # pip dependencies
├── pyproject.toml         # uv project configuration
├── .env.example           # Environment variable template
├── .env                   # Your API keys (create this)
└── README.md              # This file
```

## Architecture

The application uses an object-oriented design with clear separation of concerns:

- **AudioCapture**: Manages microphone input and audio buffering
- **SilenceDetector**: Monitors audio levels and detects speech pauses
- **WhisperProcessor**: Handles speech-to-text conversion
- **TogetherClient**: Manages LLM API communication
- **VoiceLLMApp**: Coordinates all components in the main loop

## Performance Notes

- **Whisper transcription**: ~1-3 seconds for 10-second audio clips
- **LLM response time**: ~2-5 seconds depending on response length
- **Memory usage**: ~500MB-1GB (mostly Whisper model)
- **CPU usage**: Moderate during transcription, low while listening

## Security

- API keys are never logged or displayed
- Audio is processed locally (only text is sent to LLM)
- Temporary audio files are immediately deleted after processing
- No data is stored permanently

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues or questions, please open an issue on the GitHub repository.