"""Tests for the image quality gate (brightness check)."""

from smartcam import check_brightness


def test_bright_image_passes(bright_image):
    assert check_brightness(bright_image) is True


def test_dark_image_fails(dark_image):
    assert check_brightness(dark_image) is False


def test_threshold_boundary():
    """Image with avg brightness exactly at threshold should pass."""
    from PIL import Image
    img = Image.new("RGB", (100, 100), color=(30, 30, 30))
    assert check_brightness(img) is True


def test_just_below_threshold():
    """Image with avg brightness just below threshold should fail."""
    from PIL import Image
    img = Image.new("RGB", (100, 100), color=(29, 29, 29))
    assert check_brightness(img) is False


def test_mixed_brightness():
    """Image that's half bright, half dark should pass (avg > 30)."""
    from PIL import Image
    img = Image.new("RGB", (100, 100), color=(0, 0, 0))
    for x in range(50, 100):
        for y in range(100):
            img.putpixel((x, y), (200, 200, 200))
    assert check_brightness(img) is True
