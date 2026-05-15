# Smart_Cam Configuration
# Edit this file to configure your backend and capture settings.

import os

# Backend configuration — pick ONE by setting BACKEND
BACKEND = "openai"                         # "anthropic", "openai", or "cliproxyapi"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-5.5"
OPENAI_REASONING_EFFORT = "medium"          # gpt-5.x only: minimal | low | medium | high

CLIPROXYAPI_URL = "http://localhost:8080"   # unofficial proxy (free, may break)
CLIPROXYAPI_MODEL = "claude-opus-4-6"

# Capture settings
CAPTURE_INTERVAL = 2                       # seconds between frame captures
SSIM_THRESHOLD = 0.92                      # below this = content changed
                                           # camera shake ≈ 0.95-0.99, PDF scroll ≈ 0.80-0.90
SSIM_SETTLE = 0.90                         # consecutive frames must be above this to confirm
                                           # the page has stopped moving
                                           # to consider the page "settled"
BRIGHTNESS_MIN = 30                        # skip frames darker than this (0-255 avg)

# Display settings
DISPLAY_OPACITY = 0.85
DISPLAY_BG = "#1a1a1a"
DISPLAY_FONT_SIZE = 16
DISPLAY_WIDTH = 500
DISPLAY_HEIGHT = 400

# System prompt for the LLM
SYSTEM_PROMPT = (
    "You are looking at a photo taken by a camera pointed at a page or screen. "
    "Your job:\n"
    "1. If there are QUESTIONS or EXERCISES visible: solve every one. "
    "Return the answers numbered to match. Show your work briefly for math/science.\n"
    "2. If there is a READING PASSAGE, TEXTBOOK PAGE, or LECTURE NOTES with no questions: "
    "provide a concise summary of the key points (3-5 bullets).\n"
    "3. If the image is unclear or partially visible, describe what you can see.\n"
    "Be concise. Use plain text."
)

LOG_FILE = "/tmp/smartcam.log"
FRAME_PATH = "/tmp/smartcam_frame.jpg"
