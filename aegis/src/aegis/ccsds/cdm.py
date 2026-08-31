"""Minimal CCSDS Conjunction Data Message (CDM) reader, KVN only."""

from __future__ import annotations

import numpy as np

from ..core.conjunction import Conjunction
from ..core.frames import rtn_to_eci_matrix
from ..core.objects import SpaceObject
from ..core.state import StateVector
from ..core.timebase import format_epoch, parse_epoch
from ..ingest.sources import DataSource
from .errors import CcsdsError

__all__ = ["read_cdm"]

# DataSource allows only CELESTRAK | SYNTHETIC. CDM objects are not a third
# catalog: they are labeled CELESTRAK so mixed-source checks stay homogeneous,
# with metadata["origin"] = "CDM" recording that they came from a CDM file.
# They are never marked SYNTHETIC.

_NAME_ALIASES = {
    1: ("OBJECT1_OBJECT", "OBJECT1_NAME"),
    2: ("OBJECT2_OBJECT", "OBJECT2_NAME"),
}
_ID_ALIASES = {
    1: ("OBJECT1_OBJECT_DESIGNATOR", "OBJECT1_ID"),
    2: ("OBJECT2_OBJECT_DESIGNATOR", "OBJECT2_ID"),
}
_STATE_KEYS = {
    1: ("OBJECT1_X", "OBJECT1_Y", "OBJECT1_Z", "OBJECT1_X_DOT", "OBJECT1_Y_DOT", "OBJECT1_Z_DOT"),
    2: ("OBJECT2_X", "OBJECT2_Y", "OBJECT2_Z", "OBJECT2_X_DOT", "OBJECT2_Y_DOT", "OBJECT2_Z_DOT"),
}
_COV_KEYS = {
    1: ("OBJECT1_CR_R", "OBJECT1_CT_T", "OBJECT1_CN_N"),
    2: ("OBJECT2_CR_R", "OBJECT2_CT_T", "OBJECT2_CN_N"),
}


def _strip_units(value: str) -> str:
    if value.endswith("]") and "[" in value:
        return value[: value.rfind("[")].strip()
    return value


def _parse_kvn(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.upper().startswith("COMMENT"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().upper()
        value = _strip_units(value.strip())
        if key:
            fields[key] = value
    return fields


def _require(fields: dict[str, str], *aliases: str) -> str:
    for alias in aliases:
        value = fields.get(alias)
        if value:
            return value
    raise CcsdsError(f"missing required CDM field: {aliases[0]}")


def _require_float(fields: dict[str, str], *aliases: str) -> float:
    text = _require(fields, *aliases)
    try:
        return float(text)
    except ValueError as error:
        raise CcsdsError(f"invalid numeric CDM field {aliases[0]}: {text!r}") from error


def _optional_float(fields: dict[str, str], key: str) -> float | None:
    text = fields.get(key)
    if text is None or text == "":
        return None
    try:
        return float(text)
    except ValueError as error:
        raise CcsdsError(f"invalid numeric CDM field {key}: {text!r}") from error


def _state_at_tca(fields: dict[str, str], index: int, tca) -> StateVector:
    keys = _STATE_KEYS[index]
    components = [_require_float(fields, key) for key in keys]
    return StateVector(
        epoch=tca,
        position_km=np.array(components[:3], dtype=float),
        velocity_km_s=np.array(components[3:], dtype=float),
        frame="TEME",
    )


def _covariance_diag(fields: dict[str, str], index: int) -> tuple[float, float, float] | None:
    values = [_optional_float(fields, key) for key in _COV_KEYS[index]]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise CcsdsError(
            f"incomplete OBJECT{index} RTN covariance; "
            f"need {_COV_KEYS[index][0]}, {_COV_KEYS[index][1]}, and {_COV_KEYS[index][2]}"
        )
    return (values[0], values[1], values[2])  # type: ignore[return-value]


def _cdm_object(object_id: str, name: str, covariance: tuple[float, float, float] | None) -> SpaceObject:
    metadata: dict = {"origin": "CDM"}
    if covariance is not None:
        metadata["covariance_rtn_diag_km2"] = covariance
        metadata["cdm"] = True
    return SpaceObject(
        object_id=object_id,
        name=name,
        data_source=DataSource.CELESTRAK,
        metadata=metadata,
    )


def _relative_rtn(primary: StateVector, secondary: StateVector) -> tuple[np.ndarray, np.ndarray]:
    """Other-minus-primary in the primary RTN frame (screening convention)."""
    rotation = rtn_to_eci_matrix(primary.position_km, primary.velocity_km_s)
    position_rtn = rotation.T @ (secondary.position_km - primary.position_km)
    velocity_rtn = rotation.T @ (secondary.velocity_km_s - primary.velocity_km_s)
    return position_rtn, velocity_rtn


def _conjunction_id(primary_id: str, secondary_id: str, tca) -> str:
    return f"{primary_id}:{secondary_id}:{format_epoch(tca)}"


def read_cdm(text: str) -> Conjunction:
    """Parse a minimal CCSDS CDM in KVN into a :class:`~aegis.core.conjunction.Conjunction`.

    Objects are labeled ``data_source=CELESTRAK`` with ``metadata["origin"]="CDM"``.
    That keeps the catalog wall intact (only CELESTRAK | SYNTHETIC) without
    inventing a third source or marking CDM objects SYNTHETIC.

    CDM states are ingested as published and labeled TEME; the conjunction
    records ``metadata["cdm_frame_note"]`` accordingly. Does not call
    ``generate_synthetic``.
    """
    fields = _parse_kvn(text)

    try:
        tca = parse_epoch(_require(fields, "TCA"))
    except ValueError as error:
        raise CcsdsError(f"invalid TCA: {fields.get('TCA')!r}") from error

    miss_distance_km = _require_float(fields, "MISS_DISTANCE")
    relative_speed_km_s = _require_float(fields, "RELATIVE_SPEED")

    primary_name = _require(fields, *_NAME_ALIASES[1])
    secondary_name = _require(fields, *_NAME_ALIASES[2])
    primary_id = _require(fields, *_ID_ALIASES[1])
    secondary_id = _require(fields, *_ID_ALIASES[2])

    primary_state = _state_at_tca(fields, 1, tca)
    secondary_state = _state_at_tca(fields, 2, tca)

    primary = _cdm_object(primary_id, primary_name, _covariance_diag(fields, 1))
    secondary = _cdm_object(secondary_id, secondary_name, _covariance_diag(fields, 2))

    position_rtn, velocity_rtn = _relative_rtn(primary_state, secondary_state)

    metadata: dict = {
        "cdm_frame_note": "CDM states ingested as published",
    }
    collision_probability = _optional_float(fields, "COLLISION_PROBABILITY")
    if collision_probability is not None:
        metadata["collision_probability"] = collision_probability

    return Conjunction(
        conjunction_id=_conjunction_id(primary_id, secondary_id, tca),
        primary=primary,
        secondary=secondary,
        tca=tca,
        miss_distance_km=miss_distance_km,
        relative_speed_km_s=relative_speed_km_s,
        relative_position_rtn_km=position_rtn,
        relative_velocity_rtn_km_s=velocity_rtn,
        primary_state=primary_state,
        secondary_state=secondary_state,
        metadata=metadata,
    )
