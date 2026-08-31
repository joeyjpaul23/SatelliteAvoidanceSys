"""Apogee / perigee pair prefilter.

Two objects whose radial bands cannot overlap, even after padding for
Brouwer-mean / osculating oscillation and short-period J2, cannot meet.
The pad lives in :data:`aegis.constants.PERIGEE_APOGEE_PAD_KM`.
"""

from __future__ import annotations

from ..constants import PERIGEE_APOGEE_PAD_KM
from ..core.objects import SpaceObject

__all__ = ["prefilter_pairs"]


def _radial_band(obj: SpaceObject) -> tuple[float, float] | None:
    """Return ``(perigee_radius_km, apogee_radius_km)`` when elements exist."""
    elements = obj.elements
    if elements is None:
        return None
    perigee = float(elements.perigee_radius_km)
    apogee = float(elements.apogee_radius_km)
    if apogee < perigee:
        perigee, apogee = apogee, perigee
    return perigee, apogee


def _bands_can_meet(
    band_a: tuple[float, float] | None,
    band_b: tuple[float, float] | None,
    pad_km: float,
) -> bool:
    """True unless both bands are known and cannot overlap after padding."""
    if band_a is None or band_b is None:
        return True
    return (band_a[0] - pad_km) <= (band_b[1] + pad_km) and (band_b[0] - pad_km) <= (
        band_a[1] + pad_km
    )


def prefilter_pairs(
    objects: list[SpaceObject],
    *,
    pad_km: float = PERIGEE_APOGEE_PAD_KM,
) -> list[tuple[int, int]]:
    """Return index pairs ``(i, j)`` with ``i < j`` that survive the radial filter.

    Two circular LEO satellites at the same altitude survive. A 550 km
    object and a 20000 km object do not. Pairs whose elements are missing
    cannot be excluded and are kept.
    """
    bands = [_radial_band(obj) for obj in objects]
    pairs: list[tuple[int, int]] = []
    n_objects = len(objects)
    for i in range(n_objects):
        for j in range(i + 1, n_objects):
            if _bands_can_meet(bands[i], bands[j], pad_km):
                pairs.append((i, j))
    return pairs
