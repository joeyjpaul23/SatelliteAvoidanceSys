"""Static UI contract tests (ui_console)."""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from aegis.api import app

_STANDALONE_AI = re.compile(r"(?<![A-Za-z])AI(?![A-Za-z])", re.IGNORECASE)
_AI_PHRASES = (
    "artificial intelligence",
    "ai-powered",
    "ai space",
)


def _client() -> TestClient:
    return TestClient(app)


def test_root_returns_html_globe_without_ai_marketing() -> None:
    response = _client().get("/")

    assert response.status_code == 200
    content_type = response.headers.get("content-type", "").lower()
    assert "html" in content_type

    body = response.text
    assert "globe" in body.lower()

    lowered = body.lower()
    for phrase in _AI_PHRASES:
        assert phrase not in lowered
    assert _STANDALONE_AI.search(body) is None


def test_scene_js_is_served() -> None:
    response = _client().get("/js/scene.js")
    assert response.status_code == 200
