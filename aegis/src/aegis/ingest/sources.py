"""Catalog container, data-source identifiers, and ingest structural errors.

A catalog is homogeneous: every object shares one ``data_source``. Mixed
CelesTrak and synthetic objects are a structural error, not a merge to be
papered over. The two acquisition paths never share objects, caches, or
parsers -- this module only names the wall.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ..core.objects import SpaceObject
from ..core.timebase import ensure_utc, utc_now

__all__ = [
    "DataSource",
    "Catalog",
    "CatalogError",
    "MixedDataSourceError",
    "SyntheticNotAuthorizedError",
]


class DataSource:
    """Acquisition path that produced a catalogued object.

    These are the only values ingest may write onto ``SpaceObject.data_source``.
    CelesTrak and Space-Track are both real tracking data, but they are
    separate acquisition paths and a catalog never mixes them.
    """

    CELESTRAK = "CELESTRAK"
    SPACETRACK = "SPACETRACK"
    SYNTHETIC = "SYNTHETIC"
    ALL = (CELESTRAK, SPACETRACK, SYNTHETIC)


class CatalogError(Exception):
    """Base for catalog and ingest structural errors."""


class MixedDataSourceError(CatalogError):
    """Raised when objects or catalogs from different sources are combined."""


class SyntheticNotAuthorizedError(CatalogError):
    """Raised when synthetic generation is refused by the dual gate."""


def _require_known_source(source: str) -> None:
    if source not in DataSource.ALL:
        raise CatalogError(f"unknown catalog source: {source!r}")


def _reject_mixed(source: str, objects: list[SpaceObject]) -> None:
    _require_known_source(source)
    for obj in objects:
        if obj.data_source != source:
            raise MixedDataSourceError(
                f"object {obj.object_id} has data_source={obj.data_source!r} "
                f"but catalog source is {source!r}"
            )


@dataclass
class Catalog:
    """A homogeneous collection of space objects from a single data source.

    Parameters
    ----------
    source
        One of ``DataSource.ALL``.
    objects
        Catalogued objects; every entry must carry ``data_source == source``.
        An empty list is allowed.
    fetched_at
        Timezone-aware UTC datetime of acquisition.
    query
        What was requested (CelesTrak group, or a synthetic spec summary).
    """

    source: str
    objects: list[SpaceObject] = field(default_factory=list)
    fetched_at: datetime = field(default_factory=utc_now)
    query: str = ""

    def __post_init__(self) -> None:
        self.fetched_at = ensure_utc(self.fetched_at)
        self.objects = list(self.objects)
        _reject_mixed(self.source, self.objects)

    def __len__(self) -> int:
        return len(self.objects)

    def __iter__(self):
        return iter(self.objects)

    def merge(self, other: Catalog) -> Catalog:
        """Return a new catalog combining two same-source catalogs."""
        if not isinstance(other, Catalog):
            raise TypeError(f"can only merge Catalog, not {type(other).__name__}")
        if self.source != other.source:
            raise MixedDataSourceError(
                f"cannot merge catalogs from {self.source!r} and {other.source!r}"
            )
        _reject_mixed(self.source, other.objects)
        query = self.query if self.query == other.query else f"{self.query} | {other.query}"
        fetched_at = max(self.fetched_at, other.fetched_at)
        return Catalog(
            source=self.source,
            objects=self.objects + other.objects,
            fetched_at=fetched_at,
            query=query.strip(" |"),
        )

    def extend(self, other: Catalog) -> Catalog:
        """Append another same-source catalog in place. Returns ``self``."""
        if not isinstance(other, Catalog):
            raise TypeError(f"can only extend with Catalog, not {type(other).__name__}")
        if self.source != other.source:
            raise MixedDataSourceError(
                f"cannot extend {self.source!r} catalog with {other.source!r}"
            )
        _reject_mixed(self.source, other.objects)
        self.objects.extend(other.objects)
        if other.query and other.query != self.query:
            if self.query:
                self.query = f"{self.query} | {other.query}"
            else:
                self.query = other.query
        if other.fetched_at > self.fetched_at:
            self.fetched_at = other.fetched_at
        return self

    def __add__(self, other: object) -> Catalog:
        if not isinstance(other, Catalog):
            return NotImplemented
        return self.merge(other)

    def __repr__(self) -> str:
        return f"Catalog(source={self.source!r}, n={len(self.objects)}, query={self.query!r})"
