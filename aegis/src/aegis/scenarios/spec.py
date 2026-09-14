"""Scenario request and result types, and the canonical scenario digest.

A ``Scenario`` is the unit every downstream stage of step14 consumes:
``aegis.screening.screen``, ``aegis.risk.assess_catalog``, the fleetopt
planners, and (eventually) ``aegis.store``'s ``ExperimentStore``. Keeping it
a plain, serializable bundle -- objects plus a window plus provenance --
means a scenario built by :mod:`aegis.scenarios.generator`, one replayed
from committed TLEs, and one loaded back out of a store round-trip are
interchangeable without a special case anywhere downstream.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime

from ..constants import SCREENING_BOX_STARLINK_KM
from ..core.objects import OrbitalElements, SpaceObject
from ..core.timebase import ensure_utc, format_epoch, shift

__all__ = ["ScenarioSpec", "Scenario", "scenario_digest"]


@dataclass
class ScenarioSpec:
    """A scenario request: a family name, a seed, and free-form parameters.

    ``params`` carries every family-specific knob (fanout, altitude,
    inclinations, fixture names, ...) as a plain dict rather than as a
    per-family dataclass subclass. The family table in
    ``aegis/specs/step14_fleet_optimization.md`` section 13.2 is expected to
    keep growing, and a plain dict means adding a family never touches this
    type -- only :mod:`aegis.scenarios.generator`.
    """

    family: str
    seed: int
    params: dict = field(default_factory=dict)

    def key(self) -> tuple[str, int]:
        """The ``(family, seed)`` pair a :mod:`aegis.scenarios.registry` suite indexes by."""
        return (self.family, self.seed)


@dataclass
class Scenario:
    """One concrete, screenable scenario.

    Attributes
    ----------
    scenario_id
        Deterministic, human-readable label. Not the cache key --
        :func:`scenario_digest` is -- this field only needs to be *stable*
        for a given ``(family, seed, params)``, not collision-proof.
    objects
        The catalog to screen. Homogeneous ``data_source`` per
        ``aegis.ingest.sources.Catalog``'s existing rule; every synthetic
        family sets ``DataSource.SYNTHETIC`` on every object it creates so
        the mixed-source guards in screening and risk keep applying.
    window_start, window_duration_s
        The screening window this scenario was designed against. Structural
        claims a family makes (e.g. "all TCAs within one orbital period")
        are claims about conjunctions found inside *this* window, not an
        arbitrary caller-chosen one.
    family, seed
        The generator inputs that produced ``objects``.
    provenance
        Free-form record of how this scenario was built: the family
        description, the generator parameters actually used, fixture paths
        for ``replay-tle``, and (once computed) this scenario's own digest.
        Never a place to paper over a missing input.
    screening_box_km
        RTN half-width hint for ``aegis.screening.screen``. A hint, not a
        guarantee -- callers screening ``replay-tle`` or ``dense-shell``
        scenarios may reasonably choose a different box.
    """

    scenario_id: str
    objects: list[SpaceObject]
    window_start: datetime
    window_duration_s: float
    family: str
    seed: int
    provenance: dict = field(default_factory=dict)
    screening_box_km: tuple[float, float, float] = SCREENING_BOX_STARLINK_KM

    def __post_init__(self) -> None:
        self.window_start = ensure_utc(self.window_start)
        if self.window_duration_s < 0.0:
            raise ValueError("window_duration_s must be non-negative")

    @property
    def window_end(self) -> datetime:
        return shift(self.window_start, self.window_duration_s)

    def __len__(self) -> int:
        return len(self.objects)


def _round_significant(value: float, digits: int = 12) -> float:
    """Round ``value`` to ``digits`` significant figures.

    The digest must be stable across processes and platforms. Two BLAS
    builds computing the same trig identity can disagree in the 15th
    decimal place; rounding to 12 significant figures absorbs that noise
    while still catching a genuine change in orbital elements, which move
    at far coarser precision than 1e-12 relative.
    """
    if value == 0.0 or not math.isfinite(value):
        return value
    exponent = math.floor(math.log10(abs(value)))
    return round(value, digits - int(exponent) - 1)


def _elements_payload(elements: OrbitalElements | None) -> dict | None:
    if elements is None:
        return None
    return {
        "epoch": format_epoch(elements.epoch, decimals=6),
        "mean_motion_rev_per_day": _round_significant(elements.mean_motion_rev_per_day),
        "eccentricity": _round_significant(elements.eccentricity),
        "inclination_deg": _round_significant(elements.inclination_deg),
        "raan_deg": _round_significant(elements.raan_deg),
        "arg_perigee_deg": _round_significant(elements.arg_perigee_deg),
        "mean_anomaly_deg": _round_significant(elements.mean_anomaly_deg),
        "bstar": _round_significant(elements.bstar),
        "mean_motion_dot": _round_significant(elements.mean_motion_dot),
        "mean_motion_ddot": _round_significant(elements.mean_motion_ddot),
        "element_set_number": elements.element_set_number,
        "revolution_number": elements.revolution_number,
    }


def _object_payload(obj: SpaceObject) -> dict:
    operator = obj.operator
    return {
        "object_id": obj.object_id,
        "object_type": obj.object_type,
        "elements": _elements_payload(obj.elements),
        "operator_id": operator.identifier if operator is not None else None,
        "operator_maneuverable": operator.maneuverable if operator is not None else None,
        "data_source": obj.data_source,
        "hard_body_radius_m": (
            _round_significant(obj.hard_body_radius_m) if obj.hard_body_radius_m is not None else None
        ),
    }


def scenario_digest(scenario: Scenario) -> str:
    """SHA-256 over an explicit, sorted-key JSON serialization.

    This is the cache and storage key ``aegis.store`` indexes scenarios by,
    so it must be reproducible for the same content on any machine, any
    Python version, in any process. Concretely that means:

    * ``json.dumps(..., sort_keys=True)`` rather than a raw dict walk, so no
      result depends on dict insertion order;
    * every float rounded through :func:`_round_significant` first, so no
      result depends on platform floating-point noise;
    * never Python's builtin ``hash()``, which is randomly salted per
      process by design (``PYTHONHASHSEED``) and therefore useless here;
    * ``scenario_id`` itself is excluded -- it is a human label, not
      content, and including it would make the digest depend on how it was
      spelled rather than what the scenario actually contains.
    """
    payload = {
        "family": scenario.family,
        "seed": scenario.seed,
        "window_start": format_epoch(scenario.window_start, decimals=6),
        "window_duration_s": _round_significant(float(scenario.window_duration_s)),
        "screening_box_km": [_round_significant(float(v)) for v in scenario.screening_box_km],
        "objects": [_object_payload(obj) for obj in scenario.objects],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
