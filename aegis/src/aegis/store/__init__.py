"""Offline-first, cloud-optional artifact and experiment storage for AEGIS-FO.

Every default works with zero configuration, writing under this checkout's
own ``artifacts/`` directory (see :mod:`aegis.store.config`). Cloud backends
and Postgres are opt-in and fail loud, never fall back silently -- see
``docs/LIMITATIONS.md``.
"""

from __future__ import annotations

from .artifacts import (
    ArtifactRef,
    ArtifactStore,
    GCSArtifactStore,
    LocalArtifactStore,
    S3ArtifactStore,
    open_artifact_store,
    validate_key,
)
from .config import StoreConfig
from .datasets import DatasetManifest, DatasetReader, DatasetWriter, ShardInfo
from .db import ExperimentStore
from .errors import ArtifactKeyError, StoreError, StoreUnavailableError

__all__ = [
    "ArtifactRef",
    "ArtifactStore",
    "LocalArtifactStore",
    "S3ArtifactStore",
    "GCSArtifactStore",
    "open_artifact_store",
    "validate_key",
    "StoreConfig",
    "StoreError",
    "StoreUnavailableError",
    "ArtifactKeyError",
    "ExperimentStore",
    "DatasetWriter",
    "DatasetReader",
    "DatasetManifest",
    "ShardInfo",
]
