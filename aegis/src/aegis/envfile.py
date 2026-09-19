"""Load local secrets from the repository's ``.env`` for command-line entry points.

Only entry points call :func:`load_env_file` (``python -m aegis.api``, the
pipeline CLI, the Space-Track CLI); importing the library never touches the
environment. Variables already set win, so systemd's ``EnvironmentFile`` on
the VM and anything exported in the shell take precedence. The file is
gitignored; values are never printed.

``AEGIS_DOTENV=0`` disables loading (the test suite sets it so a developer's
real Space-Track credentials can never reach a test).
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["REPO_ENV_FILE", "load_env_file"]

#: ``<repo>/.env``: this file is ``<repo>/aegis/src/aegis/envfile.py``.
REPO_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


def _parse_line(line: str) -> tuple[str, str] | None:
    text = line.strip()
    if not text or text.startswith("#") or "=" not in text:
        return None
    if text.startswith("export "):
        text = text[len("export ") :].lstrip()
    key, _, value = text.partition("=")
    key = key.strip()
    if not key.isidentifier():
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1]
    return key, value


def load_env_file(path: str | Path | None = None) -> list[str]:
    """Set unset variables from ``path`` (default: the repo ``.env``).

    Returns the names that were set. A missing file is not an error.
    """
    if os.environ.get("AEGIS_DOTENV", "1") == "0":
        return []
    env_path = Path(path) if path is not None else REPO_ENV_FILE
    if not env_path.is_file():
        return []
    loaded: list[str] = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded
