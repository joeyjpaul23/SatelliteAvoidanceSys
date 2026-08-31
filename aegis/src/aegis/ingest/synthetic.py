"""In-memory synthetic constellation generator.

This path never opens HTTP connections and never reads or writes a CelesTrak
cache. Generation is refused unless the caller presents a
:class:`SyntheticAuthorization` *and* ``AEGIS_ALLOW_SYNTHETIC=1`` is set.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone

from ..constants import MU_EARTH_KM3_S2, R_EARTH_KM, REV_PER_DAY_TO_RAD_PER_S
from ..core.objects import ObjectType, Operator, OrbitalElements, SpaceObject
from ..core.timebase import ensure_utc, format_epoch, utc_now
from .sources import Catalog, DataSource, SyntheticNotAuthorizedError

__all__ = ["SyntheticAuthorization", "SyntheticSpec", "generate_synthetic"]

#: Documented default epoch: far enough in the past that SGP4 accepts it.
DEFAULT_SYNTHETIC_EPOCH = datetime(2010, 1, 1, tzinfo=timezone.utc)

#: Circular-ish LEO; SGP4 is happier with a tiny eccentricity than with zero.
_SYNTHETIC_ECCENTRICITY = 1.0e-4

#: Mean-anomaly offset between the first two plane-0 satellites.
#: At 550 km this is ~10 km along-track -- well inside the Starlink 44 km box.
_CONJUNCTION_MA_OFFSET_DEG = 0.08

_SYNTHETIC_REFUSED = (
    "synthetic data was refused: AEGIS_ALLOW_SYNTHETIC=1 "
    "plus an explicit authorization are required"
)


class SyntheticAuthorization:
    """Explicit acknowledgement that the caller wants generated, not real, data.

    Construction requires the keyword ``acknowledge_synthetic=True``. There is
    no factory or bypass; the environment gate is checked separately in
    :func:`generate_synthetic`.
    """

    def __init__(self, *, acknowledge_synthetic: bool) -> None:
        if acknowledge_synthetic is not True:
            raise SyntheticNotAuthorizedError(_SYNTHETIC_REFUSED)
        self.acknowledge_synthetic = True


@dataclass
class SyntheticSpec:
    """Walker-like LEO constellation used only on the synthetic path.

    All fields have defaults so a tester can construct ``SyntheticSpec()``.
    The default epoch is ``2010-01-01T00:00:00+00:00``.
    """

    n_planes: int = 1
    sats_per_plane: int = 3
    altitude_km: float = 550.0
    inclination_deg: float = 53.0
    epoch: datetime = DEFAULT_SYNTHETIC_EPOCH
    seed: int = 0
    operator_id: str = "SYNTHETIC-OP"
    operator_name: str = "Synthetic Operator"
    include_known_conjunction_triple: bool = True

    def __post_init__(self) -> None:
        self.epoch = ensure_utc(self.epoch)

    def summary(self) -> str:
        return (
            f"synthetic n_planes={self.n_planes} sats_per_plane={self.sats_per_plane} "
            f"altitude_km={self.altitude_km} inclination_deg={self.inclination_deg} "
            f"seed={self.seed} epoch={format_epoch(self.epoch)} "
            f"include_known_conjunction_triple={self.include_known_conjunction_triple}"
        )


def _require_authorization(authorization: object) -> None:
    if not isinstance(authorization, SyntheticAuthorization):
        raise SyntheticNotAuthorizedError(_SYNTHETIC_REFUSED)
    if authorization.acknowledge_synthetic is not True:
        raise SyntheticNotAuthorizedError(_SYNTHETIC_REFUSED)
    if os.environ.get("AEGIS_ALLOW_SYNTHETIC") != "1":
        raise SyntheticNotAuthorizedError(_SYNTHETIC_REFUSED)


def _mean_motion_rev_per_day(altitude_km: float) -> float:
    semi_major_axis_km = R_EARTH_KM + altitude_km
    mean_motion_rad_s = (MU_EARTH_KM3_S2 / (semi_major_axis_km**3)) ** 0.5
    return mean_motion_rad_s / REV_PER_DAY_TO_RAD_PER_S


def _wrap_deg(angle: float) -> float:
    return angle % 360.0


def _mean_anomaly(spec: SyntheticSpec, plane: int, sat: int, ma_offset: float) -> float:
    n_sats = spec.sats_per_plane
    n_planes = spec.n_planes
    walker_phase = 0.0
    if n_planes > 1 and n_sats > 0:
        walker_phase = plane * (360.0 / (n_planes * n_sats))

    use_triple = (
        spec.include_known_conjunction_triple
        and plane == 0
        and n_sats >= 3
    )
    if use_triple:
        if sat == 0:
            return _wrap_deg(ma_offset)
        if sat == 1:
            return _wrap_deg(ma_offset + _CONJUNCTION_MA_OFFSET_DEG)
        if sat == 2:
            return _wrap_deg(ma_offset + 180.0)
        return _wrap_deg(ma_offset + (sat + 0.5) * (360.0 / n_sats))

    if n_sats <= 0:
        return _wrap_deg(ma_offset)
    return _wrap_deg(ma_offset + sat * (360.0 / n_sats) + walker_phase)


def generate_synthetic(authorization: SyntheticAuthorization, spec: SyntheticSpec) -> Catalog:
    """Build an in-memory constellation. Dual-gated; no network, no cache."""
    _require_authorization(authorization)

    rng = random.Random(spec.seed)
    raan_offset = rng.uniform(0.0, 360.0)
    ma_offset = rng.uniform(0.0, 360.0)
    mean_motion = _mean_motion_rev_per_day(spec.altitude_km)
    operator = Operator(
        identifier=spec.operator_id,
        name=spec.operator_name,
        maneuverable=True,
    )

    objects: list[SpaceObject] = []
    serial = 90001
    n_planes = max(spec.n_planes, 0)
    n_sats = max(spec.sats_per_plane, 0)
    for plane in range(n_planes):
        raan = _wrap_deg(raan_offset + (plane * 360.0 / n_planes if n_planes else 0.0))
        for sat in range(n_sats):
            object_id = str(serial)
            serial += 1
            elements = OrbitalElements(
                epoch=spec.epoch,
                mean_motion_rev_per_day=mean_motion,
                eccentricity=_SYNTHETIC_ECCENTRICITY,
                inclination_deg=spec.inclination_deg,
                raan_deg=raan,
                arg_perigee_deg=0.0,
                mean_anomaly_deg=_mean_anomaly(spec, plane, sat, ma_offset),
                bstar=0.0,
            )
            objects.append(
                SpaceObject(
                    object_id=object_id,
                    name=f"SYNTHETIC-{object_id}",
                    object_type=ObjectType.PAYLOAD,
                    elements=elements,
                    operator=operator,
                    data_source=DataSource.SYNTHETIC,
                )
            )

    return Catalog(
        source=DataSource.SYNTHETIC,
        objects=objects,
        fetched_at=utc_now(),
        query=spec.summary(),
    )
