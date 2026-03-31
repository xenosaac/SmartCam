"""Tests for error handling: API failures, timeouts, refusals."""

import config
from smartcam import call_llm, image_to_base64


def test_api_timeout_raises(mock_api_timeout, bright_image):
    """API timeout should raise Timeout exception."""
    import requests
    b64 = image_to_base64(bright_image)
    try:
        call_llm(b64)
        assert False, "Should have raised Timeout"
    except requests.exceptions.Timeout:
        pass


def test_api_http_error_raises(mock_api_error, bright_image, monkeypatch):
    """HTTP 500 should raise HTTPError."""
    monkeypatch.setattr(config, "BACKEND", "cliproxyapi")
    import requests
    b64 = image_to_base64(bright_image)
    try:
        call_llm(b64)
        assert False, "Should have raised HTTPError"
    except requests.exceptions.HTTPError:
        pass


def test_llm_refusal_returns_text(mock_api_refusal, bright_image, monkeypatch):
    """LLM refusal should return the refusal text (not crash)."""
    monkeypatch.setattr(config, "BACKEND", "cliproxyapi")
    b64 = image_to_base64(bright_image)
    result = call_llm(b64)
    assert "cannot help" in result.lower()


def test_camera_discovery_no_ffmpeg(monkeypatch):
    """discover_cameras should return empty list if ffmpeg not found."""
    import subprocess

    def mock_run(*args, **kwargs):
        raise FileNotFoundError("ffmpeg not found")

    monkeypatch.setattr(subprocess, "run", mock_run)

    from smartcam import discover_cameras
    result = discover_cameras()
    assert result == []


def test_capture_frame_returns_none_on_failure(monkeypatch):
    """capture_frame should return None when ffmpeg fails."""
    import subprocess

    class MockResult:
        returncode = 1
        stderr = "Device not found"

    def mock_run(*args, **kwargs):
        return MockResult()

    monkeypatch.setattr(subprocess, "run", mock_run)

    from smartcam import capture_frame
    result = capture_frame(99)
    assert result is None


def test_capture_frame_permission_denied(monkeypatch):
    """capture_frame should return 'PERMISSION_DENIED' on permission error."""
    import subprocess

    class MockResult:
        returncode = 1
        stderr = "Permission Denied by user"

    def mock_run(*args, **kwargs):
        return MockResult()

    monkeypatch.setattr(subprocess, "run", mock_run)

    from smartcam import capture_frame
    result = capture_frame(0)
    assert result == "PERMISSION_DENIED"
