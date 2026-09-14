"""Content-addressed artifact storage: local disk, S3, and GCS.

Why content-addressed
----------------------
Experiments in this codebase re-run the same scenario family under many
planners and re-write near-identical checkpoints across training epochs.
Addressing blobs by their sha256 digest rather than by caller-chosen key
means identical content is written to disk (or uploaded) at most once, no
matter how many keys point at it or how many times ``put`` is called with
the same bytes -- the digest comparison in :meth:`LocalArtifactStore.put`
(and its S3/GCS analogues) is what makes ``put`` idempotent for identical
content without ever rewriting a multi-megabyte artifact. The key namespace
stays exactly what a caller expects (``run_id/planner/checkpoint.pt``); only
the physical storage is deduplicated underneath it.

Cloud SDKs are imported lazily inside each backend's ``__init__`` so that
importing this module -- or using :class:`LocalArtifactStore` -- never
requires ``boto3`` or ``google-cloud-storage`` to be installed. Selecting a
cloud backend without its SDK or bucket raises
:class:`~aegis.store.errors.StoreUnavailableError` with an actionable
message; it never downgrades to local storage on its own. See
``docs/LIMITATIONS.md``.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from .config import StoreConfig
from .errors import ArtifactKeyError, StoreError, StoreUnavailableError

__all__ = [
    "ArtifactRef",
    "ArtifactStore",
    "LocalArtifactStore",
    "S3ArtifactStore",
    "GCSArtifactStore",
    "open_artifact_store",
    "validate_key",
]

_MAX_KEY_LENGTH = 1024


def validate_key(key: str) -> None:
    """Reject a key that would be unsafe as a path or object name.

    Rules (per the step-14 contract): no leading slash, no ``..`` segment,
    no empty segment, no control character, and a length cap shared by
    every backend so a key valid for local storage is also valid once a run
    is later replayed against S3 or GCS.
    """
    if not isinstance(key, str) or key == "":
        raise ArtifactKeyError("artifact key must be a non-empty string")
    if len(key) > _MAX_KEY_LENGTH:
        raise ArtifactKeyError(
            f"artifact key exceeds {_MAX_KEY_LENGTH} characters ({len(key)}): {key!r}"
        )
    if key.startswith("/"):
        raise ArtifactKeyError(f"artifact key must not start with '/': {key!r}")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in key):
        raise ArtifactKeyError(f"artifact key contains a control character: {key!r}")
    for segment in key.split("/"):
        if segment == "":
            raise ArtifactKeyError(f"artifact key contains an empty segment: {key!r}")
        if segment == "..":
            raise ArtifactKeyError(f"artifact key contains a '..' segment: {key!r}")


@dataclass(frozen=True)
class ArtifactRef:
    """What :meth:`ArtifactStore.put` hands back for a stored object."""

    key: str
    uri: str
    digest: str
    bytes_written: int
    content_type: str | None = None


@runtime_checkable
class ArtifactStore(Protocol):
    """Backend-independent artifact storage.

    Every implementation is content-addressed under the hood (see module
    docstring) but keyed the same way to callers: ``put``/``get`` deal only
    in logical keys, never in digests.
    """

    def put(self, key: str, data: bytes, *, content_type: str | None = None) -> ArtifactRef: ...

    def get(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...

    def list(self, prefix: str = "") -> list[str]: ...

    def uri(self, key: str) -> str: ...


class LocalArtifactStore:
    """Content-addressed blob store rooted at a local directory.

    Layout under ``<root>/<prefix>``::

        blobs/<digest[:2]>/<digest>     -- immutable, written at most once
        refs/<key>                     -- tiny JSON pointer: {digest, content_type, bytes}

    This mirrors how git separates ref names from object storage: the blob
    directory is write-once and trivially deduplicated (across keys and
    across repeated ``put`` calls of the same bytes), while the ref file is
    the only thing that changes when a key is repointed at new content.
    Both blob and ref writes go through a temp-file-then-rename so a reader
    never observes a partially written file even without a lock.
    """

    def __init__(self, root: str | Path, *, prefix: str = "") -> None:
        self.root = Path(root)
        self.prefix = prefix.strip("/")
        self.notes: list[str] = []
        base = (self.root / self.prefix) if self.prefix else self.root
        self._blobs_dir = base / "blobs"
        self._refs_dir = base / "refs"
        self._blobs_dir.mkdir(parents=True, exist_ok=True)
        self._refs_dir.mkdir(parents=True, exist_ok=True)

    def _blob_path(self, digest: str) -> Path:
        return self._blobs_dir / digest[:2] / digest

    def _ref_path(self, key: str) -> Path:
        return self._refs_dir / key

    def _read_ref(self, ref_path: Path) -> dict | None:
        if not ref_path.exists():
            return None
        try:
            import json

            return json.loads(ref_path.read_text())
        except ValueError as exc:
            raise StoreError(f"corrupt ref file at {ref_path}") from exc

    def _atomic_write(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
        tmp_path.write_bytes(data)
        tmp_path.replace(path)

    def put(self, key: str, data: bytes, *, content_type: str | None = None) -> ArtifactRef:
        import json

        validate_key(key)
        digest = hashlib.sha256(data).hexdigest()
        ref_path = self._ref_path(key)
        existing = self._read_ref(ref_path)
        if existing is not None and existing.get("digest") == digest:
            # Identical content already recorded under this key -- no write
            # of any kind, blob or ref. This is the idempotency the
            # contract requires: put() of unchanged content is a no-op.
            return ArtifactRef(
                key=key,
                uri=self.uri(key),
                digest=digest,
                bytes_written=len(data),
                content_type=existing.get("content_type"),
            )

        blob_path = self._blob_path(digest)
        if not blob_path.exists():
            self._atomic_write(blob_path, data)

        ref_payload = json.dumps(
            {"digest": digest, "content_type": content_type, "bytes": len(data)}, sort_keys=True
        ).encode("utf-8")
        self._atomic_write(ref_path, ref_payload)

        return ArtifactRef(
            key=key, uri=self.uri(key), digest=digest, bytes_written=len(data), content_type=content_type
        )

    def get(self, key: str) -> bytes:
        validate_key(key)
        ref = self._read_ref(self._ref_path(key))
        if ref is None:
            raise StoreError(f"no artifact stored for key {key!r}")
        blob_path = self._blob_path(ref["digest"])
        if not blob_path.exists():
            raise StoreError(
                f"ref for key {key!r} points at missing blob {ref['digest']} -- store corruption"
            )
        return blob_path.read_bytes()

    def exists(self, key: str) -> bool:
        validate_key(key)
        return self._ref_path(key).exists()

    def list(self, prefix: str = "") -> list[str]:
        if prefix.startswith("/"):
            raise ArtifactKeyError(f"prefix must not start with '/': {prefix!r}")
        if not self._refs_dir.exists():
            return []
        keys = [
            path.relative_to(self._refs_dir).as_posix()
            for path in self._refs_dir.rglob("*")
            if path.is_file()
        ]
        return sorted(key for key in keys if key.startswith(prefix))

    def uri(self, key: str) -> str:
        validate_key(key)
        base = (self.root / self.prefix) if self.prefix else self.root
        return (base / key).resolve().as_uri()


class S3ArtifactStore:
    """S3-backed artifact store. Requires ``boto3`` and a bucket.

    ``boto3`` is imported here, not at module scope, so a codebase that
    never selects the ``s3`` backend never needs it installed. A missing
    package, a missing bucket, or a client construction failure (typically
    absent credentials) all raise :class:`StoreUnavailableError` rather than
    deciding silently to write somewhere else -- that decision belongs to
    the caller via ``open_artifact_store(fallback=True)``.
    """

    def __init__(self, *, bucket: str | None, prefix: str = "") -> None:
        if not bucket:
            raise StoreUnavailableError(
                "S3 artifact store requires a bucket name (set AEGIS_S3_BUCKET or pass "
                "bucket= explicitly); none was given"
            )
        try:
            import boto3
        except ImportError as exc:
            raise StoreUnavailableError(
                "S3 artifact store requires the 'boto3' package, which is not installed "
                "in this environment (pip install boto3)"
            ) from exc

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.notes: list[str] = []
        try:
            self._client = boto3.client("s3")
        except Exception as exc:  # noqa: BLE001 - boto3 raises a wide, version-dependent family of errors
            raise StoreUnavailableError(f"could not create an S3 client: {exc}") from exc

    def _full_key(self, key: str) -> str:
        validate_key(key)
        return f"{self.prefix}/{key}" if self.prefix else key

    def put(self, key: str, data: bytes, *, content_type: str | None = None) -> ArtifactRef:
        full_key = self._full_key(key)
        digest = hashlib.sha256(data).hexdigest()
        if self.exists(key):
            head = self._client.head_object(Bucket=self.bucket, Key=full_key)
            if head.get("Metadata", {}).get("sha256") == digest:
                return ArtifactRef(
                    key=key, uri=self.uri(key), digest=digest, bytes_written=len(data),
                    content_type=content_type,
                )
        extra_args: dict = {"Metadata": {"sha256": digest}}
        if content_type:
            extra_args["ContentType"] = content_type
        self._client.put_object(Bucket=self.bucket, Key=full_key, Body=data, **extra_args)
        return ArtifactRef(
            key=key, uri=self.uri(key), digest=digest, bytes_written=len(data), content_type=content_type
        )

    def get(self, key: str) -> bytes:
        full_key = self._full_key(key)
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=full_key)
        except Exception as exc:  # noqa: BLE001 - botocore.ClientError's code lives in exc.response
            error_code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if error_code in ("NoSuchKey", "404"):
                raise StoreError(f"no artifact stored for key {key!r}") from exc
            raise StoreError(f"could not read artifact {key!r} from S3: {exc}") from exc
        return response["Body"].read()

    def exists(self, key: str) -> bool:
        full_key = self._full_key(key)
        try:
            self._client.head_object(Bucket=self.bucket, Key=full_key)
            return True
        except Exception as exc:  # noqa: BLE001 - botocore.ClientError's code lives in exc.response
            error_code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if error_code in ("404", "NoSuchKey"):
                return False
            raise StoreError(f"could not check existence of {key!r} in S3: {exc}") from exc

    def list(self, prefix: str = "") -> list[str]:
        full_prefix = f"{self.prefix}/{prefix}" if self.prefix else prefix
        keys: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                object_key = obj["Key"]
                keys.append(object_key[len(self.prefix) + 1 :] if self.prefix else object_key)
        return sorted(keys)

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{self._full_key(key)}"


class GCSArtifactStore:
    """GCS-backed artifact store. Requires ``google-cloud-storage`` and a bucket.

    Same lazy-import and fail-loud contract as :class:`S3ArtifactStore`.
    """

    def __init__(self, *, bucket: str | None, prefix: str = "") -> None:
        if not bucket:
            raise StoreUnavailableError(
                "GCS artifact store requires a bucket name (set AEGIS_GCS_BUCKET or pass "
                "bucket= explicitly); none was given"
            )
        try:
            from google.cloud import storage
        except ImportError as exc:
            raise StoreUnavailableError(
                "GCS artifact store requires the 'google-cloud-storage' package, which is "
                "not installed in this environment (pip install google-cloud-storage)"
            ) from exc

        self.bucket_name = bucket
        self.prefix = prefix.strip("/")
        self.notes: list[str] = []
        try:
            self._client = storage.Client()
            self._bucket = self._client.bucket(bucket)
        except Exception as exc:  # noqa: BLE001 - google-auth raises a wide credentials-error family
            raise StoreUnavailableError(f"could not create a GCS client: {exc}") from exc

    def _full_key(self, key: str) -> str:
        validate_key(key)
        return f"{self.prefix}/{key}" if self.prefix else key

    def put(self, key: str, data: bytes, *, content_type: str | None = None) -> ArtifactRef:
        full_key = self._full_key(key)
        digest = hashlib.sha256(data).hexdigest()
        blob = self._bucket.blob(full_key)
        if blob.exists(self._client):
            blob.reload(client=self._client)
            if (blob.metadata or {}).get("sha256") == digest:
                return ArtifactRef(
                    key=key, uri=self.uri(key), digest=digest, bytes_written=len(data),
                    content_type=content_type,
                )
        blob.metadata = {"sha256": digest}
        blob.upload_from_string(data, content_type=content_type or "application/octet-stream")
        return ArtifactRef(
            key=key, uri=self.uri(key), digest=digest, bytes_written=len(data), content_type=content_type
        )

    def get(self, key: str) -> bytes:
        full_key = self._full_key(key)
        blob = self._bucket.blob(full_key)
        if not blob.exists(self._client):
            raise StoreError(f"no artifact stored for key {key!r}")
        return blob.download_as_bytes()

    def exists(self, key: str) -> bool:
        return self._bucket.blob(self._full_key(key)).exists(self._client)

    def list(self, prefix: str = "") -> list[str]:
        full_prefix = f"{self.prefix}/{prefix}" if self.prefix else prefix
        keys = []
        for blob in self._client.list_blobs(self.bucket_name, prefix=full_prefix):
            keys.append(blob.name[len(self.prefix) + 1 :] if self.prefix else blob.name)
        return sorted(keys)

    def uri(self, key: str) -> str:
        return f"gs://{self.bucket_name}/{self._full_key(key)}"


def open_artifact_store(config: StoreConfig | None = None, *, fallback: bool = False) -> ArtifactStore:
    """Build the store named by ``config`` (or the environment).

    A cloud backend that cannot be constructed raises
    :class:`StoreUnavailableError` unless ``fallback=True``, in which case
    the failure is recorded on the returned local store's ``.notes`` and
    execution continues offline. The default is to raise: silently writing
    somewhere other than what was configured is exactly the failure mode
    ``docs/LIMITATIONS.md`` bans for cloud storage.
    """
    cfg = config if config is not None else StoreConfig.from_env()

    if cfg.backend == "local":
        return LocalArtifactStore(cfg.root, prefix=cfg.prefix)

    if cfg.backend == "s3":
        try:
            return S3ArtifactStore(bucket=cfg.s3_bucket, prefix=cfg.prefix)
        except StoreUnavailableError as exc:
            if not fallback:
                raise
            local_store = LocalArtifactStore(cfg.root, prefix=cfg.prefix)
            local_store.notes.append(f"s3 backend unavailable ({exc}); fell back to local at {local_store.root}")
            return local_store

    if cfg.backend == "gcs":
        try:
            return GCSArtifactStore(bucket=cfg.gcs_bucket, prefix=cfg.prefix)
        except StoreUnavailableError as exc:
            if not fallback:
                raise
            local_store = LocalArtifactStore(cfg.root, prefix=cfg.prefix)
            local_store.notes.append(f"gcs backend unavailable ({exc}); fell back to local at {local_store.root}")
            return local_store

    raise StoreError(f"unknown store backend {cfg.backend!r}")
