"""Shared test fixtures for Smart_Cam."""

import io
import pytest
from PIL import Image


@pytest.fixture
def bright_image():
    """A 200x200 white image (brightness ~255)."""
    return Image.new("RGB", (200, 200), color=(255, 255, 255))


@pytest.fixture
def dark_image():
    """A 200x200 nearly-black image (brightness ~10)."""
    return Image.new("RGB", (200, 200), color=(10, 10, 10))


@pytest.fixture
def large_image():
    """A 4000x3000 image simulating a 12MP camera frame."""
    return Image.new("RGB", (4000, 3000), color=(128, 128, 128))


@pytest.fixture
def exercise_image_a():
    """A synthetic image with distinct pattern A (red)."""
    img = Image.new("RGB", (200, 200), color=(200, 50, 50))
    # Draw a simple pattern to make the hash distinct
    for x in range(0, 200, 20):
        for y in range(100):
            img.putpixel((x, y), (255, 255, 255))
    return img


@pytest.fixture
def exercise_image_b():
    """A synthetic image with distinct pattern B (blue)."""
    img = Image.new("RGB", (200, 200), color=(50, 50, 200))
    for y in range(0, 200, 20):
        for x in range(100):
            img.putpixel((x, y), (255, 255, 255))
    return img


@pytest.fixture
def mock_api_success(monkeypatch):
    """Mock a successful LLM API response."""
    class MockResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {
                "choices": [{"message": {"content": "1. Answer A\n2. Answer B\n3. Answer C"}}]
            }
        @property
        def text(self):
            return '{"choices":[{"message":{"content":"1. Answer A"}}]}'

    def mock_post(*args, **kwargs):
        return MockResponse()

    import requests
    monkeypatch.setattr(requests, "post", mock_post)


@pytest.fixture
def mock_api_timeout(monkeypatch):
    """Mock an API timeout."""
    import requests

    def mock_post(*args, **kwargs):
        raise requests.exceptions.Timeout("Connection timed out")

    monkeypatch.setattr(requests, "post", mock_post)


@pytest.fixture
def mock_api_refusal(monkeypatch):
    """Mock an LLM refusal response."""
    import requests

    class MockResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {
                "choices": [{"message": {"content": "I cannot help with exam questions."}}]
            }

    def mock_post(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(requests, "post", mock_post)


@pytest.fixture
def mock_api_error(monkeypatch):
    """Mock an HTTP 500 error from the API."""
    import requests

    class MockResponse:
        status_code = 500
        text = '{"error": "Internal server error"}'
        def raise_for_status(self):
            raise requests.exceptions.HTTPError(response=self)

    def mock_post(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(requests, "post", mock_post)
