"""Exceptions for the artifact and experiment storage layer."""

from __future__ import annotations

__all__ = ["StoreError", "StoreUnavailableError", "ArtifactKeyError"]


class StoreError(RuntimeError):
    """Base class for every storage failure.

    Distinct from an empty result (an empty ``list()`` or a scenario with no
    recorded runs is a successful query). This is reserved for the store
    itself being unable to do what was asked -- corruption, a missing key on
    ``get``, or a backend that refused the operation.
    """


class StoreUnavailableError(StoreError):
    """Raised when a requested backend cannot be used at all.

    Covers a missing cloud SDK, a missing bucket/DSN environment variable,
    and a credentials failure at client construction time. This codebase has
    a documented "no silent fallback" policy (docs/LIMITATIONS.md): a caller
    that asked for S3 or GCS or Postgres and cannot get it must see this
    exception, never a quiet downgrade to local storage. The only sanctioned
    downgrade path is the explicit ``open_artifact_store(fallback=True)``,
    which still raises this from the failed backend construction and records
    why before falling back.
    """


class ArtifactKeyError(StoreError):
    """Raised when an artifact key fails validation.

    Keys double as filesystem-safe paths and, for cloud backends, object
    names. A leading slash, a ``..`` segment, an empty segment, or a control
    character would make the key backend-dependent or exploitable as a path
    traversal -- rejected before it ever reaches a filesystem or SDK call.
    """
