"""Pytest fixtures for AEGIS ingest contract tests.

``pyproject.toml`` already sets ``pythonpath = ["src"]``. This file only
isolates environment state so tests do not inherit a developer machine's
synthetic opt-in flag.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_synthetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic ingest is opt-in; never inherit ``AEGIS_ALLOW_SYNTHETIC``."""
    monkeypatch.delenv("AEGIS_ALLOW_SYNTHETIC", raising=False)
