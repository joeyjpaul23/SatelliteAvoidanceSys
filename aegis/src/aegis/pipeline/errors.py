"""Pipeline-layer errors."""

from __future__ import annotations

__all__ = ["PipelineError"]


class PipelineError(RuntimeError):
    """Raised when the end-to-end pipeline cannot run.

    Distinct from an empty catalog or a plan with unresolved conjunctions.
    Those are successful results. This error means ingest, source selection,
    or configuration is invalid -- including a CelesTrak failure, which is
    never papered over with a synthetic catalog.
    """
