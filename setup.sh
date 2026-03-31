#!/bin/bash
set -e

echo "=== Smart_Cam Setup ==="

# Check macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: Smart_Cam requires macOS (Continuity Camera)."
    exit 1
fi

# Install ffmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo "Installing ffmpeg..."
    if command -v brew &>/dev/null; then
        brew install ffmpeg
    else
        echo "Error: Homebrew not found. Install it from https://brew.sh then re-run."
        exit 1
    fi
else
    echo "ffmpeg already installed: $(which ffmpeg)"
fi

# Create venv
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "Installing Python dependencies..."
source venv/bin/activate
pip install -r requirements.txt --quiet

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit config.py to set your API key (ANTHROPIC_API_KEY or OPENAI_API_KEY)"
echo "  2. Mount your iPhone on a stand pointing at your desk"
echo "  3. Run: source venv/bin/activate && python smartcam.py"
echo ""
echo "To list available cameras: python smartcam.py --list-cameras"
