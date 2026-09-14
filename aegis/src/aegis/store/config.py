"""Environment-driven configuration for artifact and experiment storage.

Offline-first: every default works with zero environment variables set,
writing under this checkout's own ``artifacts/`` directory. Cloud backends
and Postgres are opt-in via explicit environment variables so that a bare
checkout plus a test run never reaches the network on its own -- selecting
a cloud backend without the matching SDK or bucket is a loud
:class:`~aegis.store.errors.StoreUnavailableError`, never a silent
downgrade. See ``docs/LIMITATIONS.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["StoreConfig"]

#: Root of the ``aegis/`` checkout (the directory holding pyproject.toml),
#: derived from this file's location rather than hard-coded, so the default
#: artifact root follows the checkout wherever it is cloned.
_AEGIS_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ROOT = _AEGIS_ROOT / "artifacts"
_DEFAULT_EXPERIMENT_DB = _DEFAULT_ROOT / "experiments.sqlite3"

_VALID_BACKENDS = ("local", "s3", "gcs")


@dataclass(frozen=True)
class StoreConfig:
    """Resolved storage configuration.

    Construct directly for tests (each field has a sane offline default) or
    via :meth:`from_env` to read the environment variables the deployed
    system actually uses. ``backend`` selects which :class:`ArtifactStore`
    :func:`~aegis.store.artifacts.open_artifact_store` builds; ``root`` is
    only consulted by the local backend but is always resolved, so a
    fallback from a cloud backend always has somewhere to land.
    """

    backend: str = "local"
    root: Path = field(default_factory=lambda: _DEFAULT_ROOT)
    s3_bucket: str | None = None
    gcs_bucket: str | None = None
    prefix: str = ""
    experiment_db: Path = field(default_factory=lambda: _DEFAULT_EXPERIMENT_DB)
    postgres_dsn: str | None = None

    def __post_init__(self) -> None:
        if self.backend not in _VALID_BACKENDS:
            raise ValueError(
                f"unknown store backend {self.backend!r}; expected one of {_VALID_BACKENDS}"
            )

    @classmethod
    def from_env(cls, *, root: str | Path | None = None) -> StoreConfig:
        """Build from ``AEGIS_STORE_*`` environment variables.

        ``root`` lets a caller override ``AEGIS_STORE_ROOT`` explicitly
        (tests do this to point at a temp directory) without touching the
        environment. Every other field is env-only because there is no
        legitimate reason to override a bucket name or DSN from code --
        that is exactly the kind of implicit override this module exists to
        avoid.
        """
        backend = os.environ.get("AEGIS_STORE_BACKEND", "local").strip().lower() or "local"

        root_value = root if root is not None else os.environ.get("AEGIS_STORE_ROOT")
        resolved_root = Path(root_value).expanduser() if root_value else _DEFAULT_ROOT

        experiment_db_value = os.environ.get("AEGIS_EXPERIMENT_DB")
        resolved_experiment_db = (
            Path(experiment_db_value).expanduser()
            if experiment_db_value
            else (resolved_root / "experiments.sqlite3")
        )

        return cls(
            backend=backend,
            root=resolved_root,
            s3_bucket=os.environ.get("AEGIS_S3_BUCKET") or None,
            gcs_bucket=os.environ.get("AEGIS_GCS_BUCKET") or None,
            prefix=os.environ.get("AEGIS_STORE_PREFIX", ""),
            experiment_db=resolved_experiment_db,
            postgres_dsn=os.environ.get("AEGIS_POSTGRES_DSN") or None,
        )
