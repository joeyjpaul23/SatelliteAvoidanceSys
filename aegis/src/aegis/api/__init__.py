"""REST surface and the operations console it serves."""

from aegis import __version__

from .app import app

__all__ = ["app", "__version__"]
