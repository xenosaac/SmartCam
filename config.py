# Smart_Cam Configuration
# Edit this file to configure your backend and capture settings.

# Backend configuration — pick ONE by setting BACKEND
BACKEND = "cliproxyapi"                    # "anthropic", "openai", or "cliproxyapi"

ANTHROPIC_API_KEY = ""                     # from console.anthropic.com (paid, reliable)
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

OPENAI_API_KEY = ""                        # from platform.openai.com (paid, reliable)
OPENAI_MODEL = "gpt-4o"

CLIPROXYAPI_URL = "http://localhost:8080"   # unofficial proxy (free, may break)
CLIPROXYAPI_MODEL = "claude-sonnet-4-20250514"

# Capture settings
CAPTURE_INTERVAL = 2                       # seconds between frame captures
HASH_THRESHOLD = 5                         # hamming distance to detect page change
DEBOUNCE_SIMILARITY = 3                    # max hamming between consecutive frames
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
    "You are looking at a photo of an exercise sheet. "
    "Solve every visible problem. Return ONLY the answers, numbered to match "
    "the problems on the sheet. If any problem is partially cut off at the edge "
    "of the image, skip it and note 'partially visible, skipped.' Be concise."
)

LOG_FILE = "/tmp/smartcam.log"
FRAME_PATH = "/tmp/smartcam_frame.jpg"
