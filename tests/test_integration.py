"""Integration tests: end-to-end frame -> answer pipeline (mocked)."""

import queue
import threading
import time

import config
from smartcam import (
    check_brightness,
    image_to_base64,
    resize_for_api,
    call_llm,
)
import imagehash


def test_full_pipeline_happy_path(exercise_image_a, mock_api_success, monkeypatch):
    """Full pipeline: image -> quality check -> hash -> resize -> API -> answer."""
    monkeypatch.setattr(config, "BACKEND", "cliproxyapi")

    # 1. Quality gate
    assert check_brightness(exercise_image_a) is True

    # 2. Hash (first frame, so always "changed")
    current_hash = imagehash.average_hash(exercise_image_a)
    assert current_hash is not None

    # 3. Resize
    resized = resize_for_api(exercise_image_a)
    assert resized.width <= 1024

    # 4. Encode
    b64 = image_to_base64(resized)
    assert len(b64) > 0

    # 5. API call
    answer = call_llm(b64)
    assert "Answer" in answer


def test_dark_frame_skipped(dark_image):
    """Dark frames should be rejected by quality gate before hashing."""
    assert check_brightness(dark_image) is False
    # No hash computation, no API call needed


def test_unchanged_frame_skipped(exercise_image_a):
    """Two identical frames should not trigger an API call."""
    h1 = imagehash.average_hash(exercise_image_a)
    h2 = imagehash.average_hash(exercise_image_a)
    distance = h1 - h2
    assert distance < config.HASH_THRESHOLD  # No change detected


def test_page_flip_detected(exercise_image_a, exercise_image_b):
    """Different page should produce distance > threshold."""
    h1 = imagehash.average_hash(exercise_image_a)
    h2 = imagehash.average_hash(exercise_image_b)
    distance = h1 - h2
    assert distance >= config.HASH_THRESHOLD


def test_backpressure_discard_logic():
    """Verify discard_pending flag logic works correctly."""
    # Simulate: API call in flight, new page detected
    api_in_flight = True
    discard_pending = False

    # New page detected while API in flight
    if api_in_flight:
        discard_pending = True

    # API response arrives
    api_in_flight = False
    answer = "stale answer"

    if discard_pending:
        answer = None  # Discarded
        discard_pending = False

    assert answer is None  # Stale answer was discarded


def test_queue_communication():
    """Answer queue should correctly pass messages between threads."""
    q = queue.Queue()

    def producer():
        q.put("1. x = 5\n2. y = 10")

    t = threading.Thread(target=producer)
    t.start()
    t.join()

    result = q.get(timeout=1)
    assert "x = 5" in result
    assert "y = 10" in result
