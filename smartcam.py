#!/usr/bin/env python3
"""Smart_Cam: Hands-free exercise answer scanner.

iPhone camera (via Continuity Camera) captures exercise sheets,
sends them to a vision LLM, displays answers on Mac screen.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import requests
from PIL import Image

import config


def _ssim(a, b):
    # Lazy import: skimage triggers a large scipy load cascade (30-90s cold).
    # Keep CLI paths like --list-cameras fast.
    from skimage.metrics import structural_similarity
    return structural_similarity(a, b)

# ---------------------------------------------------------------------------
# Logging — state transitions only, not per-frame spam
# ---------------------------------------------------------------------------

logger = logging.getLogger("smartcam")
logger.setLevel(logging.INFO)
_fh = logging.FileHandler(config.LOG_FILE, mode="a")
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(_fh)
_sh = logging.StreamHandler()
_sh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(_sh)

# ---------------------------------------------------------------------------
# Camera discovery
# ---------------------------------------------------------------------------


def discover_cameras() -> list[tuple[int, str]]:
    """Return list of (index, name) for available AVFoundation cameras."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", "dummy"],
            capture_output=True, text=True, timeout=10,
        )
        stderr = result.stderr
    except FileNotFoundError:
        logger.error("ffmpeg not found. Run ./setup.sh first.")
        return []
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg device discovery timed out.")
        return []

    cameras = []
    in_video = False
    for line in stderr.splitlines():
        if "AVFoundation video devices:" in line:
            in_video = True
            continue
        if "AVFoundation audio devices:" in line:
            break
        if in_video:
            m = re.search(r"\[(\d+)\]\s+(.+)", line)
            if m:
                cameras.append((int(m.group(1)), m.group(2).strip()))
    return cameras


def find_iphone_camera(cameras: list[tuple[int, str]]) -> int | None:
    """Find the first camera whose name suggests it's an iPhone."""
    for idx, name in cameras:
        if "iphone" in name.lower():
            return idx
    return None


def list_cameras_and_exit():
    cameras = discover_cameras()
    if not cameras:
        print("No cameras found. Is ffmpeg installed?")
        sys.exit(1)
    print("Available cameras:")
    for idx, name in cameras:
        tag = " <-- iPhone" if "iphone" in name.lower() else ""
        print(f"  [{idx}] {name}{tag}")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Frame capture (Phase 1: one-shot ffmpeg)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Persistent camera process (imagesnap -t keeps Continuity Camera alive)
# ---------------------------------------------------------------------------

_camera_proc = None
_camera_frame_dir = "/tmp/smartcam_frames"


def start_camera_process():
    """Start imagesnap in time-lapse mode. It keeps the camera open and writes
    a new JPEG every CAPTURE_INTERVAL seconds."""
    global _camera_proc
    os.makedirs(_camera_frame_dir, exist_ok=True)
    # Clean old frames
    for f in os.listdir(_camera_frame_dir):
        os.remove(os.path.join(_camera_frame_dir, f))

    _camera_proc = subprocess.Popen(
        [
            "imagesnap", "-t", str(config.CAPTURE_INTERVAL),
            "-w", "1.0",
            os.path.join(_camera_frame_dir, "frame"),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    logger.info("Camera process started (PID %d)", _camera_proc.pid)


def stop_camera_process():
    """Stop the persistent camera process."""
    global _camera_proc
    if _camera_proc and _camera_proc.poll() is None:
        _camera_proc.terminate()
        _camera_proc.wait(timeout=5)
        logger.info("Camera process stopped")
    _camera_proc = None


def capture_frame(device_index: int) -> Image.Image | None:
    """Read the latest frame written by the persistent imagesnap process."""
    global _camera_proc

    # Start camera process if not running
    if _camera_proc is None or _camera_proc.poll() is not None:
        try:
            start_camera_process()
            time.sleep(2)  # wait for first frame
        except FileNotFoundError:
            logger.error("imagesnap not found. Install: brew install imagesnap")
            return None

    # Find the latest frame file that we haven't already processed
    try:
        files = sorted(
            [f for f in os.listdir(_camera_frame_dir) if f.startswith("frame")],
            key=lambda f: os.path.getmtime(os.path.join(_camera_frame_dir, f)),
        )
    except OSError:
        return None

    if not files:
        return None

    latest = os.path.join(_camera_frame_dir, files[-1])

    # Skip if we already processed this exact file
    last_file = getattr(capture_frame, "_last_file", None)
    if latest == last_file:
        # Wait for a new frame from imagesnap
        return "SAME_FRAME"

    try:
        age = time.time() - os.path.getmtime(latest)
        if age > config.CAPTURE_INTERVAL * 5:
            return None  # stale frame, camera may have stopped

        img = Image.open(latest)
        img.load()  # force read before file gets overwritten
        capture_frame._last_file = latest

        # Clean up old frames (keep only last 3)
        for f in files[:-3]:
            try:
                os.remove(os.path.join(_camera_frame_dir, f))
            except OSError:
                pass

        return img
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------


def check_brightness(img: Image.Image) -> bool:
    """Return True if image is bright enough."""
    gray = img.convert("L")
    avg = sum(gray.getdata()) / (gray.width * gray.height)
    return avg >= config.BRIGHTNESS_MIN


def resize_for_api(img: Image.Image, max_width: int = 1024) -> Image.Image:
    """Resize image to max_width, preserving aspect ratio."""
    if img.width <= max_width:
        return img
    ratio = max_width / img.width
    new_size = (max_width, int(img.height * ratio))
    return img.resize(new_size, Image.LANCZOS)


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 JPEG string."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# LLM API call
# ---------------------------------------------------------------------------


def call_llm(image_b64: str) -> str:
    """Send image to configured LLM backend and return the answer text."""
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {"type": "text", "text": "Solve the problems in this image."},
            ],
        },
    ]

    if config.BACKEND == "anthropic":
        return _call_anthropic(image_b64)
    elif config.BACKEND == "openai":
        return _call_openai_compatible(
            url="https://api.openai.com/v1/chat/completions",
            api_key=config.OPENAI_API_KEY,
            model=config.OPENAI_MODEL,
            messages=messages,
        )
    elif config.BACKEND == "cliproxyapi":
        return _call_openai_compatible(
            url=f"{config.CLIPROXYAPI_URL}/v1/chat/completions",
            api_key="not-needed",
            model=config.CLIPROXYAPI_MODEL,
            messages=messages,
        )
    else:
        return f"Unknown backend: {config.BACKEND}"


def _call_anthropic(image_b64: str) -> str:
    """Call Anthropic API directly (uses their native format)."""
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": config.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": config.ANTHROPIC_MODEL,
            "max_tokens": 4096,
            "system": config.SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": "Solve the problems in this image."},
                    ],
                }
            ],
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["content"][0]["text"]


def _call_openai_compatible(url: str, api_key: str, model: str, messages: list) -> str:
    """Call any OpenAI-compatible API."""
    payload: dict = {"model": model, "messages": messages}
    if model.startswith("gpt-5"):
        payload["max_completion_tokens"] = 4096
        effort = getattr(config, "OPENAI_REASONING_EFFORT", None)
        if effort:
            payload["reasoning_effort"] = effort
    else:
        payload["max_tokens"] = 4096
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Capture thread
# ---------------------------------------------------------------------------


def capture_loop(answer_queue: queue.Queue, shutdown: threading.Event):
    """Main capture loop. Simple state machine:

    1. Grab latest frame
    2. If content changed from last sent: wait one more frame to confirm (debounce)
    3. Send to LLM, display answer
    4. Immediately check if content changed AGAIN while LLM was thinking
    5. If yes, send the new content right away (no extra debounce wait)
    """
    camera_index = getattr(capture_loop, "camera_index", 0)
    last_sent_gray = None  # grayscale numpy array of last sent frame

    logger.info("Capture loop started (camera index %d)", camera_index)

    loop_count = 0
    while not shutdown.is_set():
        loop_count += 1

        # --- Step 1: Get a fresh frame ---
        img = _get_frame(camera_index, answer_queue, shutdown)
        if img is None or shutdown.is_set():
            continue

        if not check_brightness(img):
            answer_queue.put("Waiting for clear image...")
            shutdown.wait(config.CAPTURE_INTERVAL)
            continue

        # Convert to small grayscale for fast SSIM comparison
        small = img.resize((320, 240), Image.LANCZOS).convert("L")
        current_gray = np.array(small)

        # --- Step 2: Check if content changed (SSIM) ---
        if last_sent_gray is not None:
            score = _ssim(last_sent_gray, current_gray)
            if score > config.SSIM_THRESHOLD:
                # No meaningful change
                if loop_count % 20 == 0:
                    logger.info("[loop %d] No change (SSIM=%.3f)", loop_count, score)
                shutdown.wait(config.CAPTURE_INTERVAL)
                continue
            logger.info("[loop %d] PAGE CHANGE (SSIM=%.3f)", loop_count, score)

            # Debounce: grab one more frame, confirm it's stable
            shutdown.wait(config.CAPTURE_INTERVAL)
            if shutdown.is_set():
                break
            img2 = _get_frame(camera_index, answer_queue, shutdown)
            if img2 is None:
                continue
            small2 = img2.resize((320, 240), Image.LANCZOS).convert("L")
            gray2 = np.array(small2)
            settle_score = _ssim(current_gray, gray2)
            if settle_score < config.SSIM_SETTLE:
                # Still changing
                logger.info("[loop %d] Still changing (settle=%.3f)", loop_count, settle_score)
                continue
            img = img2
            current_gray = gray2
            logger.info("[loop %d] Page settled (settle=%.3f)", loop_count, settle_score)
        else:
            logger.info("[loop %d] First frame ever", loop_count)

        # --- Step 3: Send to LLM ---
        logger.info("[loop %d] Sending to LLM...", loop_count)
        answer_queue.put("Analyzing...")
        resized = resize_for_api(img)
        b64 = image_to_base64(resized)

        t0 = time.time()
        try:
            answer = call_llm(b64)
            elapsed = time.time() - t0
            logger.info("[loop %d] LLM RESPONSE (%.1fs): %s...", loop_count, elapsed, answer[:80])
            last_sent_gray = current_gray
            answer_queue.put(answer)
            logger.info("[loop %d] Answer pushed to queue (qsize=%d)", loop_count, answer_queue.qsize())
        except Exception as e:
            logger.error("[loop %d] LLM ERROR: %s", loop_count, e)
            body = ""
            if hasattr(e, "response"):
                try:
                    body = e.response.text
                except Exception:
                    pass
            answer_queue.put(f"LLM error:\n{body or e}")
            last_sent_gray = current_gray
            shutdown.wait(config.CAPTURE_INTERVAL)
            continue

        # Step 4: No immediate re-check. Let the next loop iteration handle
        # any new page changes naturally. This prevents "answer flashing"
        # where camera shake (distance 4-8) triggers an immediate re-send.
        logger.info("[loop %d] Answer displayed, waiting for next cycle", loop_count)

        shutdown.wait(config.CAPTURE_INTERVAL)

    logger.info("Capture loop stopped")


def _get_frame(camera_index, answer_queue, shutdown):
    """Helper: grab a frame, handle errors. Returns Image or None."""
    result = capture_frame(camera_index)

    if result == "PERMISSION_DENIED":
        answer_queue.put(
            "Camera permission needed.\n"
            "Grant in System Settings > Privacy & Security > Camera"
        )
        shutdown.wait(5)
        return None

    if result == "SAME_FRAME":
        # Don't log every SAME_FRAME (too noisy), just wait
        shutdown.wait(0.5)
        return None

    if result is None:
        logger.warning("Capture failed — re-discovering cameras")
        cameras = discover_cameras()
        new_idx = find_iphone_camera(cameras)
        if new_idx is not None:
            capture_loop.camera_index = new_idx
            logger.info("Re-discovered camera at index %d", new_idx)
        shutdown.wait(3)
        return None

    return result


# ---------------------------------------------------------------------------
# Qt display (main thread)
# ---------------------------------------------------------------------------


def run_display(answer_queue: queue.Queue, shutdown: threading.Event):
    """Run the PyQt6 always-on-top answer window with live camera preview."""
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QFont, QPixmap, QImage
    from PyQt6.QtWidgets import (
        QApplication, QLabel, QScrollArea, QHBoxLayout, QVBoxLayout, QWidget,
        QSplitter, QFrame,
    )

    app = QApplication(sys.argv)

    WINDOW_W = 900
    WINDOW_H = 500

    # Main window
    window = QWidget()
    window.setWindowTitle("Smart_Cam")
    window.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
    window.resize(WINDOW_W, WINDOW_H)
    window.setStyleSheet(f"background-color: {config.DISPLAY_BG};")
    window.setWindowOpacity(config.DISPLAY_OPACITY)

    # Horizontal layout: camera preview (left) | answers (right)
    main_layout = QHBoxLayout(window)
    main_layout.setContentsMargins(8, 8, 8, 8)
    main_layout.setSpacing(8)

    # --- Left panel: camera preview ---
    preview_frame = QFrame()
    preview_frame.setStyleSheet(
        "QFrame { background-color: #111; border: 1px solid #333; border-radius: 6px; }"
    )
    preview_layout = QVBoxLayout(preview_frame)
    preview_layout.setContentsMargins(4, 4, 4, 4)

    preview_title = QLabel("Camera Preview")
    preview_title.setStyleSheet("color: #888; font-size: 11px; border: none;")
    preview_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
    preview_layout.addWidget(preview_title)

    preview_label = QLabel("Waiting for camera...")
    preview_label.setStyleSheet("color: #555; border: none;")
    preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    preview_label.setMinimumSize(350, 280)
    preview_layout.addWidget(preview_label)

    main_layout.addWidget(preview_frame, stretch=1)

    # --- Right panel: answers ---
    answer_frame = QFrame()
    answer_frame.setStyleSheet(
        "QFrame { background-color: #1a1a1a; border: 1px solid #333; border-radius: 6px; }"
    )
    answer_layout = QVBoxLayout(answer_frame)
    answer_layout.setContentsMargins(4, 4, 4, 4)

    answer_title = QLabel("Answers")
    answer_title.setStyleSheet("color: #888; font-size: 11px; border: none;")
    answer_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
    answer_layout.addWidget(answer_title)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setStyleSheet(
        "QScrollArea { border: none; background: transparent; }"
        "QScrollBar:vertical { width: 6px; background: transparent; }"
        "QScrollBar::handle:vertical { background: #555; border-radius: 3px; }"
    )

    answer_label = QLabel("Starting... Point your camera at an exercise sheet.")
    answer_label.setWordWrap(True)
    answer_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    answer_label.setFont(QFont("Menlo", config.DISPLAY_FONT_SIZE))
    answer_label.setStyleSheet("color: #e0e0e0; padding: 8px; border: none;")
    answer_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

    scroll.setWidget(answer_label)
    answer_layout.addWidget(scroll)

    main_layout.addWidget(answer_frame, stretch=1)

    # Position: right side of screen
    screen = QApplication.primaryScreen()
    if screen:
        geo = screen.availableGeometry()
        window.move(
            geo.right() - WINDOW_W - 20,
            geo.bottom() - WINDOW_H - 20,
        )

    window.show()

    # --- Update camera preview ---
    def update_preview():
        """Load the latest frame from disk and display it."""
        try:
            files = sorted(
                [f for f in os.listdir(_camera_frame_dir) if f.startswith("frame")],
                key=lambda f: os.path.getmtime(os.path.join(_camera_frame_dir, f)),
            )
            if not files:
                return
            latest = os.path.join(_camera_frame_dir, files[-1])
            pixmap = QPixmap(latest)
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    preview_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                preview_label.setPixmap(scaled)
        except Exception:
            pass

    # --- Poll queue for new answers ---
    def check_queue():
        latest_text = None
        count = 0
        try:
            while True:
                latest_text = answer_queue.get_nowait()
                count += 1
        except queue.Empty:
            pass

        if latest_text is not None:
            logger.info("[DISPLAY] Updating answer (picked %d items from queue, showing last)", count)
            logger.info("[DISPLAY] Text preview: %s...", latest_text[:80])
            answer_label.setText(latest_text)
            answer_label.adjustSize()
            answer_label.repaint()
            scroll.verticalScrollBar().setValue(0)
            if not latest_text.startswith(("Waiting", "Camera", "Starting", "LLM", "Analyzing")):
                os.system('afplay /System/Library/Sounds/Glass.aiff &')

        if shutdown.is_set():
            app.quit()

    # Timer for answers (100ms)
    answer_timer = QTimer()
    answer_timer.timeout.connect(check_queue)
    answer_timer.start(100)

    # Timer for camera preview (500ms)
    preview_timer = QTimer()
    preview_timer.timeout.connect(update_preview)
    preview_timer.start(500)

    # Handle Ctrl-C
    signal.signal(signal.SIGINT, lambda *_: shutdown.set())

    app.exec()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Smart_Cam: Hands-free exercise scanner")
    parser.add_argument("--camera", type=int, default=None, help="Camera device index")
    parser.add_argument("--list-cameras", action="store_true", help="List available cameras and exit")
    args = parser.parse_args()

    if args.list_cameras:
        list_cameras_and_exit()

    # Validate backend config
    if config.BACKEND == "anthropic" and not config.ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not set in config.py")
        print("Set it or change BACKEND to 'openai' or 'cliproxyapi'")
        sys.exit(1)
    if config.BACKEND == "openai" and not config.OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not set in config.py")
        print("Set it or change BACKEND to 'anthropic' or 'cliproxyapi'")
        sys.exit(1)

    # Find camera
    if args.camera is not None:
        camera_index = args.camera
        logger.info("Using camera index %d (from --camera)", camera_index)
    else:
        cameras = discover_cameras()
        if not cameras:
            print("No cameras found. Is ffmpeg installed? Is your iPhone connected?")
            print("Run: python smartcam.py --list-cameras")
            sys.exit(1)

        iphone_idx = find_iphone_camera(cameras)
        if iphone_idx is not None:
            camera_index = iphone_idx
            cam_name = next(n for i, n in cameras if i == iphone_idx)
            logger.info("Auto-detected iPhone camera: [%d] %s", camera_index, cam_name)
        else:
            camera_index = cameras[0][0]
            logger.info(
                "No iPhone found, using first camera: [%d] %s",
                camera_index, cameras[0][1],
            )
            print(f"Warning: No iPhone camera detected. Using [{camera_index}] {cameras[0][1]}")
            print("Use --camera <index> to specify. Run --list-cameras to see options.")

    # Set up shared state
    answer_queue = queue.Queue()
    shutdown_event = threading.Event()
    capture_loop.camera_index = camera_index

    # Start capture thread
    capture_thread = threading.Thread(
        target=capture_loop, args=(answer_queue, shutdown_event), daemon=True
    )
    capture_thread.start()

    print(f"Smart_Cam running (backend={config.BACKEND}, camera={camera_index})")
    print("Press Ctrl+C to stop.")

    # Run Qt display on main thread (required by PyQt6)
    try:
        run_display(answer_queue, shutdown_event)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_event.set()
        stop_camera_process()
        capture_thread.join(timeout=5)
        # Cleanup temp files
        for f in [config.FRAME_PATH]:
            try:
                os.remove(f)
            except OSError:
                pass
        import shutil
        shutil.rmtree(_camera_frame_dir, ignore_errors=True)
        logger.info("Smart_Cam stopped")


if __name__ == "__main__":
    main()
