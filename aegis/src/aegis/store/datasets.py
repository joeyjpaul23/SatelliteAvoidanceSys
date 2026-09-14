"""Sharded Parquet datasets, manifest-tracked, round-tripped through an ArtifactStore.

Why shard
---------
``aegis.ml.train`` reads training tensors back scenario-by-scenario for its
held-out-by-scenario-digest split (see the step-14/15 contract); a single
monolithic Parquet file would force loading the whole dataset to read one
shard's worth of rows. Sharding also keeps any one artifact small enough to
upload to S3/GCS without multi-part complexity.

The manifest is the source of truth for what a dataset *is* -- shard keys,
row counts, schema version, and the scenario digests it was built from.
Normalization statistics belong in ``feature_stats`` here rather than being
recomputed at inference (the ml contract requires exactly this, to avoid
train/inference skew).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from ..core.timebase import format_epoch, utc_now
from .artifacts import ArtifactStore
from .errors import StoreError

__all__ = ["ShardInfo", "DatasetManifest", "DatasetWriter", "DatasetReader"]

_MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class ShardInfo:
    """One Parquet shard's coordinates within a dataset."""

    key: str
    rows: int
    bytes: int

    def to_json(self) -> dict[str, Any]:
        return {"key": self.key, "rows": self.rows, "bytes": self.bytes}

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ShardInfo:
        return cls(key=payload["key"], rows=payload["rows"], bytes=payload["bytes"])


@dataclass(frozen=True)
class DatasetManifest:
    """Everything needed to locate and reassemble a sharded dataset."""

    dataset_id: str
    schema_version: int
    shards: tuple[ShardInfo, ...]
    scenario_digests: tuple[str, ...]
    feature_stats: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    @property
    def total_rows(self) -> int:
        return sum(shard.rows for shard in self.shards)

    def to_json(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "schema_version": self.schema_version,
            "shards": [shard.to_json() for shard in self.shards],
            "scenario_digests": list(self.scenario_digests),
            "feature_stats": self.feature_stats,
            "created_at": self.created_at,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> DatasetManifest:
        return cls(
            dataset_id=payload["dataset_id"],
            schema_version=payload["schema_version"],
            shards=tuple(ShardInfo.from_json(shard) for shard in payload["shards"]),
            scenario_digests=tuple(payload["scenario_digests"]),
            feature_stats=payload.get("feature_stats", {}),
            created_at=payload.get("created_at", ""),
        )


def _manifest_key(prefix: str, dataset_id: str) -> str:
    return f"{prefix}/{dataset_id}/{_MANIFEST_NAME}" if prefix else f"{dataset_id}/{_MANIFEST_NAME}"


def _shard_key(prefix: str, dataset_id: str, shard_index: int) -> str:
    base = f"{prefix}/{dataset_id}" if prefix else dataset_id
    return f"{base}/shard-{shard_index:05d}.parquet"


class DatasetWriter:
    """Writes a :class:`pyarrow.Table` as sharded Parquet plus a manifest.

    Row order is preserved within and across shards (shard *i* holds rows
    ``[i * rows_per_shard, (i+1) * rows_per_shard)``) so a reader that only
    needs shard 0 never has to guess which rows it contains.
    """

    def __init__(
        self,
        store: ArtifactStore,
        *,
        dataset_id: str,
        prefix: str = "datasets",
        schema_version: int = 1,
        rows_per_shard: int = 2000,
    ) -> None:
        if rows_per_shard <= 0:
            raise ValueError(f"rows_per_shard must be positive, got {rows_per_shard}")
        self.store = store
        self.dataset_id = dataset_id
        self.prefix = prefix.strip("/")
        self.schema_version = schema_version
        self.rows_per_shard = rows_per_shard

    def write_table(
        self,
        table: pa.Table,
        *,
        scenario_digests: Sequence[str] = (),
        feature_stats: dict[str, Any] | None = None,
    ) -> DatasetManifest:
        shards: list[ShardInfo] = []
        n_rows = table.num_rows
        shard_index = 0
        for start in range(0, n_rows, self.rows_per_shard):
            end = min(start + self.rows_per_shard, n_rows)
            shard_table = table.slice(start, end - start)
            buffer = self._encode_shard(shard_table)
            key = _shard_key(self.prefix, self.dataset_id, shard_index)
            ref = self.store.put(key, buffer, content_type="application/vnd.apache.parquet")
            shards.append(ShardInfo(key=key, rows=end - start, bytes=ref.bytes_written))
            shard_index += 1

        manifest = DatasetManifest(
            dataset_id=self.dataset_id,
            schema_version=self.schema_version,
            shards=tuple(shards),
            scenario_digests=tuple(scenario_digests),
            feature_stats=feature_stats or {},
            created_at=format_epoch(utc_now()),
        )
        manifest_bytes = json.dumps(manifest.to_json(), sort_keys=True).encode("utf-8")
        self.store.put(
            _manifest_key(self.prefix, self.dataset_id), manifest_bytes, content_type="application/json"
        )
        return manifest

    @staticmethod
    def _encode_shard(table: pa.Table) -> bytes:
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink)
        return sink.getvalue().to_pybytes()


class DatasetReader:
    """Reads a manifest and its shards back through an :class:`ArtifactStore`."""

    def __init__(self, store: ArtifactStore, *, dataset_id: str, prefix: str = "datasets") -> None:
        self.store = store
        self.dataset_id = dataset_id
        self.prefix = prefix.strip("/")

    def read_manifest(self) -> DatasetManifest:
        key = _manifest_key(self.prefix, self.dataset_id)
        payload = json.loads(self.store.get(key).decode("utf-8"))
        return DatasetManifest.from_json(payload)

    def read_table(self, manifest: DatasetManifest | None = None) -> pa.Table:
        manifest = manifest if manifest is not None else self.read_manifest()
        if not manifest.shards:
            raise StoreError(f"dataset {manifest.dataset_id!r} has no shards to read")
        tables = [
            pq.read_table(pa.BufferReader(self.store.get(shard.key))) for shard in manifest.shards
        ]
        return pa.concat_tables(tables)
