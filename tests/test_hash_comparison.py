"""Tests for perceptual hash change detection and temporal debounce."""

import imagehash
from smartcam import resize_for_api, image_to_base64
import config


def test_identical_images_zero_distance(bright_image):
    """Same image should produce hash distance 0."""
    h1 = imagehash.average_hash(bright_image)
    h2 = imagehash.average_hash(bright_image)
    assert h1 - h2 == 0


def test_different_images_high_distance(exercise_image_a, exercise_image_b):
    """Clearly different images should produce distance > HASH_THRESHOLD."""
    h1 = imagehash.average_hash(exercise_image_a)
    h2 = imagehash.average_hash(exercise_image_b)
    assert h1 - h2 > config.HASH_THRESHOLD


def test_slight_brightness_change_low_distance(bright_image):
    """Minor brightness change should produce distance < HASH_THRESHOLD."""
    from PIL import ImageEnhance
    slightly_darker = ImageEnhance.Brightness(bright_image).enhance(0.95)
    h1 = imagehash.average_hash(bright_image)
    h2 = imagehash.average_hash(slightly_darker)
    assert h1 - h2 < config.HASH_THRESHOLD


def test_debounce_similar_frames_settle(exercise_image_a):
    """Two consecutive identical frames should be within DEBOUNCE_SIMILARITY."""
    h1 = imagehash.average_hash(exercise_image_a)
    h2 = imagehash.average_hash(exercise_image_a)
    assert h1 - h2 <= config.DEBOUNCE_SIMILARITY


def test_debounce_different_frames_reject(exercise_image_a, exercise_image_b):
    """Two very different frames should exceed DEBOUNCE_SIMILARITY."""
    h1 = imagehash.average_hash(exercise_image_a)
    h2 = imagehash.average_hash(exercise_image_b)
    assert h1 - h2 > config.DEBOUNCE_SIMILARITY


def test_resize_reduces_size(large_image):
    """Resize should bring a 4000px-wide image down to 1024px."""
    resized = resize_for_api(large_image)
    assert resized.width == 1024
    assert resized.height < large_image.height


def test_resize_no_upscale(bright_image):
    """Small images should not be upscaled."""
    resized = resize_for_api(bright_image)
    assert resized.width == bright_image.width


def test_base64_encoding(bright_image):
    """Base64 encoding should produce a non-empty string."""
    b64 = image_to_base64(bright_image)
    assert len(b64) > 0
    # Should be valid base64
    import base64
    decoded = base64.b64decode(b64)
    assert len(decoded) > 0
