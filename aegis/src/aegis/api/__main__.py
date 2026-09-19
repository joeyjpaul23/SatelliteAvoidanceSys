"""``python -m aegis.api`` serves the console on 127.0.0.1:8000."""

from __future__ import annotations

import uvicorn

from ..envfile import load_env_file
from .app import app


def main() -> None:
    load_env_file()
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
