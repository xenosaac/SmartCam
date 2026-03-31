# Smart_Cam

Hands-free exercise answer scanner. Point your iPhone at a paper exercise sheet, answers appear on your Mac screen automatically.

## How it works

1. Your iPhone acts as a webcam via Apple Continuity Camera (no app needed)
2. `smartcam.py` captures frames every 2 seconds via ffmpeg
3. Perceptual hashing detects when you flip to a new page
4. The new page is sent to a vision LLM (Claude or GPT-5.4)
5. Answers appear in an always-on-top window on your Mac

No touching the phone or Mac during operation. Just flip pages.

## Requirements

- macOS 13+ (Ventura or later)
- iPhone with iOS 16+ (same Apple ID as Mac)
- Python 3.9+
- ffmpeg

## Quick start

```bash
# 1. Setup
./setup.sh

# 2. Configure your API key in config.py
#    (Anthropic, OpenAI, or CLIProxyAPI)

# 3. Run
source venv/bin/activate
python smartcam.py
```

## Usage

```bash
# List available cameras
python smartcam.py --list-cameras

# Use a specific camera
python smartcam.py --camera 2

# Default: auto-detects iPhone via Continuity Camera
python smartcam.py
```

## Backend options

Edit `BACKEND` in `config.py`:

| Backend | Cost | Reliability |
|---------|------|-------------|
| `anthropic` | ~$0.01-0.05/question | Official API, reliable |
| `openai` | ~$0.01-0.05/question | Official API, reliable |
| `cliproxyapi` | Free (uses Claude Max subscription) | Unofficial, may break |

## Tips

- **Disable auto-lock** on your iPhone: Settings > Display > Auto-Lock > Never
- **Close other camera apps** (FaceTime, Zoom) before running
- **USB connection** is more reliable than wireless for Continuity Camera
- Logs are at `/tmp/smartcam.log` for debugging

## Running tests

```bash
source venv/bin/activate
pip install pytest
pytest tests/ -v
```
