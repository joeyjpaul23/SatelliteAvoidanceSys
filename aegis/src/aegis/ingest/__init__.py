"""Orbital catalog acquisition.

Two paths exist and they do not mix:

``celestrak``
    Real GP / SupGP downloads from CelesTrak, with an on-disk cache.
``synthetic``
    In-memory generated constellation, dual-gated behind an authorization
    object and ``AEGIS_ALLOW_SYNTHETIC=1``.
"""

from .sources import (
    Catalog,
    CatalogError,
    DataSource,
    MixedDataSourceError,
    SyntheticNotAuthorizedError,
)

__all__ = [
    "DataSource",
    "Catalog",
    "CatalogError",
    "MixedDataSourceError",
    "SyntheticNotAuthorizedError",
    "SyntheticAuthorization",
    "SyntheticSpec",
    "CelesTrakClient",
    "CelesTrakError",
    "catalog_from_tle_file",
    "fetch_celestrak",
    "generate_synthetic",
]

_CELESTRAK_EXPORTS = frozenset(
    {"CelesTrakClient", "CelesTrakError", "catalog_from_tle_file", "fetch_celestrak"}
)
_SYNTHETIC_EXPORTS = frozenset(
    {"SyntheticAuthorization", "SyntheticSpec", "generate_synthetic"}
)


def __getattr__(name: str):
    if name in _CELESTRAK_EXPORTS:
        from . import celestrak

        return getattr(celestrak, name)
    if name in _SYNTHETIC_EXPORTS:
        from . import synthetic

        return getattr(synthetic, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
