"""Screening-layer errors."""

from __future__ import annotations

__all__ = ["ScreeningError"]


class ScreeningError(RuntimeError):
    """Raised when screening cannot produce a defensible geometric result.

    Distinct from an empty result. No conjunctions in the window is success
    with an empty list; this error means propagation or refinement failed
    and the caller must not invent a close approach.
    """
