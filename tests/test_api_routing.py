"""Tests for backend API routing and call logic."""

import config
from smartcam import call_llm, image_to_base64


def test_cliproxyapi_backend(mock_api_success, bright_image, monkeypatch):
    """CLIProxyAPI backend should call localhost:8080."""
    monkeypatch.setattr(config, "BACKEND", "cliproxyapi")

    called_urls = []
    original_post = None
    import requests

    class MockResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"message": {"content": "Answer"}}]}

    def tracking_post(url, **kwargs):
        called_urls.append(url)
        return MockResponse()

    monkeypatch.setattr(requests, "post", tracking_post)

    b64 = image_to_base64(bright_image)
    result = call_llm(b64)
    assert result == "Answer"
    assert any("localhost:8080" in u for u in called_urls)


def test_openai_backend(monkeypatch, bright_image):
    """OpenAI backend should call api.openai.com."""
    monkeypatch.setattr(config, "BACKEND", "openai")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")

    called_urls = []
    import requests

    class MockResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"message": {"content": "OpenAI answer"}}]}

    def tracking_post(url, **kwargs):
        called_urls.append(url)
        assert "Bearer test-key" in kwargs.get("headers", {}).get("Authorization", "")
        return MockResponse()

    monkeypatch.setattr(requests, "post", tracking_post)

    b64 = image_to_base64(bright_image)
    result = call_llm(b64)
    assert result == "OpenAI answer"
    assert any("api.openai.com" in u for u in called_urls)


def test_anthropic_backend(monkeypatch, bright_image):
    """Anthropic backend should call api.anthropic.com with x-api-key header."""
    monkeypatch.setattr(config, "BACKEND", "anthropic")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-ant-test")

    called_urls = []
    called_headers = []
    import requests

    class MockResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"content": [{"text": "Anthropic answer"}]}

    def tracking_post(url, **kwargs):
        called_urls.append(url)
        called_headers.append(kwargs.get("headers", {}))
        return MockResponse()

    monkeypatch.setattr(requests, "post", tracking_post)

    b64 = image_to_base64(bright_image)
    result = call_llm(b64)
    assert result == "Anthropic answer"
    assert any("api.anthropic.com" in u for u in called_urls)
    assert called_headers[0]["x-api-key"] == "sk-ant-test"


def test_unknown_backend(monkeypatch, bright_image):
    """Unknown backend should return an error message."""
    monkeypatch.setattr(config, "BACKEND", "nonexistent")
    b64 = image_to_base64(bright_image)
    result = call_llm(b64)
    assert "Unknown backend" in result
