#!/bin/bash

echo "🚀 Voice-to-LLM Application Setup"
echo "=================================="

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10 or 3.11"
    exit 1
fi

# Check Python version - REQUIRE 3.11
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Current Python version: $PYTHON_VERSION"

if [ "$PYTHON_VERSION" != "3.9" ]; then
    echo "❌ ERROR: This application requires Python 3.9 exactly"
    echo "   Current version: Python $PYTHON_VERSION"
    echo ""
    echo "💡 To install Python 3.9:"
    echo "   Using pyenv: pyenv install 3.9.18 && pyenv local 3.9.18"
    echo "   Using conda: conda create -n voice-llm python=3.9 && conda activate voice-llm"
    echo "   Or download from: https://www.python.org/downloads/release/python-3918/"
    echo ""
    echo "After installing Python 3.9, run this setup script again."
    exit 1
fi

echo "✓ Python 3.9 detected"

# Check for system dependencies
echo ""
echo "Checking system dependencies..."

# macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! command -v brew &> /dev/null; then
        echo "⚠️ Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    if ! brew list portaudio &> /dev/null; then
        echo "📦 Installing PortAudio..."
        brew install portaudio
    else
        echo "✓ PortAudio installed"
    fi
    
    if ! brew list ffmpeg &> /dev/null; then
        echo "📦 Installing FFmpeg..."
        brew install ffmpeg
    else
        echo "✓ FFmpeg installed"
    fi
fi

# Linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "📦 Installing system dependencies (may require sudo)..."
    sudo apt-get update
    sudo apt-get install -y portaudio19-dev ffmpeg python3-dev
fi

# Check for uv
echo ""
if command -v uv &> /dev/null; then
    echo "✓ uv is installed"
    echo ""
    echo "Installing Python dependencies with uv..."
    uv sync
    echo "Installing Whisper..."
    uv pip install openai-whisper
else
    echo "⚠️ uv not found. Using pip instead..."
    echo ""
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
    echo "Installing Whisper..."
    pip install openai-whisper
fi

# Check for .env file
echo ""
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "⚠️ Please edit .env and add your TOGETHER_API_KEY"
else
    echo "✓ .env file exists"
fi

# Test imports
echo ""
echo "Testing imports..."

if command -v uv &> /dev/null; then
    uv run python -c "import pyaudio; print('✓ PyAudio imported successfully')" 2>/dev/null || echo "❌ PyAudio import failed"
    uv run python -c "import numpy; print('✓ NumPy imported successfully')" 2>/dev/null || echo "❌ NumPy import failed"
    uv run python -c "import httpx; print('✓ httpx imported successfully')" 2>/dev/null || echo "❌ httpx import failed" 
    uv run python -c "import whisper; print('✓ Whisper imported successfully')" 2>/dev/null || echo "❌ Whisper import failed"
else
    source venv/bin/activate
    python -c "import pyaudio; print('✓ PyAudio imported successfully')" 2>/dev/null || echo "❌ PyAudio import failed"
    python -c "import numpy; print('✓ NumPy imported successfully')" 2>/dev/null || echo "❌ NumPy import failed"
    python -c "import httpx; print('✓ httpx imported successfully')" 2>/dev/null || echo "❌ httpx import failed"
    python -c "import whisper; print('✓ Whisper imported successfully')" 2>/dev/null || echo "❌ Whisper import failed"
fi

echo ""
echo "=================================="
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your TOGETHER_API_KEY"
echo "2. Run the demo: python demo.py"
echo "3. Run the app: python voice_llm_app.py"
echo ""
echo "For more info, see README.md"