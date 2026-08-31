"""CCSDS Orbit Ephemeris Message (OEM) v3.0 writer, KVN only."""

from __future__ import annotations

from ..core.objects import SpaceObject
from ..core.state import StateVector
from ..core.timebase import format_epoch, utc_now
from .errors import CcsdsError

__all__ = ["write_oem"]


def write_oem(
    object: SpaceObject,
    states: list[StateVector],
    *,
    originator: str = "AEGIS",
) -> str:
    """Return a CCSDS OEM v3.0 KVN string for ``object`` and ``states``.

    Positions are kilometres and velocities kilometres per second, as stored
    on each :class:`~aegis.core.state.StateVector`. At least one state is
    required.
    """
    if not states:
        raise CcsdsError("write_oem requires at least one state")

    name = object.name or object.object_id
    start = format_epoch(states[0].epoch, decimals=3)
    stop = format_epoch(states[-1].epoch, decimals=3)

    lines = [
        "CCSDS_OEM_VERS = 3.0",
        f"CREATION_DATE = {format_epoch(utc_now(), decimals=3)}",
        f"ORIGINATOR = {originator}",
        "",
        "META_START",
        f"OBJECT_NAME = {name}",
        f"OBJECT_ID = {object.object_id}",
        "CENTER_NAME = EARTH",
        "REF_FRAME = TEME",
        "TIME_SYSTEM = UTC",
        f"START_TIME = {start}",
        f"STOP_TIME = {stop}",
        "META_STOP",
        "",
    ]
    for state in states:
        epoch = format_epoch(state.epoch, decimals=3)
        x, y, z = (float(v) for v in state.position_km)
        vx, vy, vz = (float(v) for v in state.velocity_km_s)
        lines.append(
            f"{epoch}  {x:.6f} {y:.6f} {z:.6f} {vx:.9f} {vy:.9f} {vz:.9f}"
        )
    lines.append("")
    return "\n".join(lines)
