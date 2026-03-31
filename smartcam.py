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

import imagehash
import requests
from PIL import Image

import config

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


def capture_frame(device_index: int) -> Image.Image | None:
    """Capture a single frame via ffmpeg. Returns PIL Image or None on failure."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-f", "avfoundation",
                "-i", str(device_index),
                "-frames:v", "1",
                "-y", config.FRAME_PATH,
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            if "permission" in result.stderr.lower() or "denied" in result.stderr.lower():
                return "PERMISSION_DENIED"
            return None
        return Image.open(config.FRAME_PATH)
    except subprocess.TimeoutExpired:
        return None
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
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"model": model, "messages": messages, "max_tokens": 4096},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Capture thread
# ---------------------------------------------------------------------------


def capture_loop(answer_queue: queue.Queue, shutdown: threading.Event):
    """Main capture loop running in a background thread."""
    camera_index = getattr(capture_loop, "camera_index", 0)
    last_sent_hash = None
    pending_hash = None       # hash of first changed frame (for debounce)
    api_in_flight = False
    discard_pending = False

    logger.info("Capture loop started (camera index %d)", camera_index)

    while not shutdown.is_set():
        # Capture
        result = capture_frame(camera_index)

        if result == "PERMISSION_DENIED":
            answer_queue.put(
                "Camera permission needed.\n"
                "Grant in System Settings > Privacy & Security > Camera"
            )
            logger.error("Camera permission denied")
            shutdown.wait(5)
            continue

        if result is None:
            # Camera may have disconnected — re-discover
            logger.warning("Capture failed — re-discovering cameras")
            answer_queue.put("Camera disconnected — reconnecting...")
            cameras = discover_cameras()
            new_idx = find_iphone_camera(cameras)
            if new_idx is not None:
                camera_index = new_idx
                logger.info("Re-discovered camera at index %d", camera_index)
            shutdown.wait(5)
            continue

        img = result

        # Quality gate: brightness
        if not check_brightness(img):
            answer_queue.put("Waiting for clear image...")
            shutdown.wait(config.CAPTURE_INTERVAL)
            continue

        # Change detection
        current_hash = imagehash.average_hash(img)

        if last_sent_hash is not None:
            distance = current_hash - last_sent_hash
            if distance < config.HASH_THRESHOLD:
                # No change
                shutdown.wait(config.CAPTURE_INTERVAL)
                continue
            else:
                # Change detected
                if pending_hash is None:
                    # First changed frame — start debounce
                    pending_hash = current_hash
                    logger.info(
                        "Page change detected (distance=%d), waiting for settle...",
                        distance,
                    )
                    shutdown.wait(config.CAPTURE_INTERVAL)
                    continue
                else:
                    # Second frame — check if settled
                    debounce_dist = current_hash - pending_hash
                    if debounce_dist > config.DEBOUNCE_SIMILARITY:
                        # Still changing — reset debounce
                        pending_hash = current_hash
                        shutdown.wait(config.CAPTURE_INTERVAL)
                        continue
                    # Settled! Proceed to API call
                    logger.info("Page settled (debounce_dist=%d)", debounce_dist)
                    pending_hash = None
        else:
            # First frame ever — send it
            logger.info("First frame captured, sending to LLM")

        # Backpressure: if API call is in flight, mark for discard
        if api_in_flight:
            discard_pending = True
            logger.info("API call in flight — will discard stale response")
            shutdown.wait(config.CAPTURE_INTERVAL)
            continue

        # Resize and encode
        resized = resize_for_api(img)
        b64 = image_to_base64(resized)

        # API call
        api_in_flight = True
        t0 = time.time()
        try:
            answer = call_llm(b64)
            elapsed = time.time() - t0
            logger.info("LLM response received (%.1fs)", elapsed)

            if discard_pending:
                logger.info("Discarding stale response")
                discard_pending = False
            else:
                last_sent_hash = current_hash
                answer_queue.put(answer)
        except requests.exceptions.Timeout:
            logger.warning("API call timed out — retrying once")
            try:
                answer = call_llm(b64)
                elapsed = time.time() - t0
                logger.info("LLM retry response received (%.1fs)", elapsed)
                if not discard_pending:
                    last_sent_hash = current_hash
                    answer_queue.put(answer)
                discard_pending = False
            except Exception as e:
                logger.error("API retry failed: %s", e)
                answer_queue.put(f"LLM unavailable — check backend\n({e})")
                discard_pending = False
        except requests.exceptions.HTTPError as e:
            logger.error("API error: %s", e)
            try:
                body = e.response.text
            except Exception:
                body = str(e)
            answer_queue.put(f"LLM error:\n{body}")
            discard_pending = False
        except Exception as e:
            logger.error("Unexpected API error: %s", e)
            answer_queue.put(f"LLM unavailable — check backend\n({e})")
            discard_pending = False
        finally:
            api_in_flight = False

        shutdown.wait(config.CAPTURE_INTERVAL)

    logger.info("Capture loop stopped")


# ---------------------------------------------------------------------------
# Qt display (main thread)
# ---------------------------------------------------------------------------


def run_display(answer_queue: queue.Queue, shutdown: threading.Event):
    """Run the PyQt6 always-on-top answer window."""
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import (
        QApplication, QLabel, QScrollArea, QVBoxLayout, QWidget,
    )

    app = QApplication(sys.argv)

    # Main window
    window = QWidget()
    window.setWindowTitle("Smart_Cam")
    window.setWindowFlags(
        Qt.WindowType.WindowStaysOnTopHint
        | Qt.WindowType.FramelessWindowHint
        | Qt.WindowType.Tool
    )
    window.setFixedSize(config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
    window.setStyleSheet(
        f"background-color: {config.DISPLAY_BG};"
        f"border-radius: 12px;"
    )
    window.setWindowOpacity(config.DISPLAY_OPACITY)

    # Layout
    layout = QVBoxLayout(window)
    layout.setContentsMargins(16, 16, 16, 16)

    # Title bar (for dragging)
    title = QLabel("Smart_Cam")
    title.setStyleSheet("color: #666; font-size: 11px; padding-bottom: 4px;")
    title.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(title)

    # Scrollable answer area
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setStyleSheet(
        "QScrollArea { border: none; }"
        "QScrollBar:vertical { width: 6px; background: transparent; }"
        "QScrollBar::handle:vertical { background: #555; border-radius: 3px; }"
    )

    answer_label = QLabel("Starting... Point your camera at an exercise sheet.")
    answer_label.setWordWrap(True)
    answer_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    answer_label.setFont(QFont("SF Mono, Menlo, monospace", config.DISPLAY_FONT_SIZE))
    answer_label.setStyleSheet("color: #e0e0e0; padding: 8px;")
    answer_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

    scroll.setWidget(answer_label)
    layout.addWidget(scroll)

    # Position: bottom-right corner
    from PyQt6.QtWidgets import QApplication as QApp
    screen = QApp.primaryScreen()
    if screen:
        geo = screen.availableGeometry()
        window.move(
            geo.right() - config.DISPLAY_WIDTH - 20,
            geo.bottom() - config.DISPLAY_HEIGHT - 20,
        )

    window.show()

    # Poll queue for new answers
    def check_queue():
        try:
            while True:
                text = answer_queue.get_nowait()
                answer_label.setText(text)
                # Chime on new real answer (not status messages)
                if not text.startswith(("Waiting", "Camera", "Starting", "LLM")):
                    os.system('afplay /System/Library/Sounds/Glass.aiff &')
        except queue.Empty:
            pass

        if shutdown.is_set():
            app.quit()

    timer = QTimer()
    timer.timeout.connect(check_queue)
    timer.start(100)

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
        capture_thread.join(timeout=5)
        # Cleanup temp files
        for f in [config.FRAME_PATH]:
            try:
                os.remove(f)
            except OSError:
                pass
        logger.info("Smart_Cam stopped")


if __name__ == "__main__":
    main()
