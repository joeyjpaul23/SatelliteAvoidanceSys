"""Tensorising a planned scenario into a variable-topology conjunction graph.

Why this module exists
----------------------
The learned component of this project (novelty ledger claim C6) predicts the
active set and dual prices of the fleet program from the *conjunction graph*
rather than from a fixed-size parameter vector. Bertsimas & Stellato
(arXiv:1907.02206) and Cauligi et al., CoCo (arXiv:2004.03736) learn active
sets and integer strategies with flat networks over a fixed parameter vector,
so every training problem must have the same number of constraints; Gasse et
al. (arXiv:1906.01629) use a bipartite constraint/variable graph but predict
branching decisions inside an exact branch-and-bound. Here the topology is
whatever the scenario contains -- as many satellites, conjunctions and induced
pairs as screening found -- and it is the row labels carried on every edge
that let a prediction be handed back to :mod:`aegis.fleetopt.solver` by name.

The graph
---------
Nodes are the objects of the scenario. Edges are the rows of the fleet
program that carry physics: every resolve row (a screened conjunction) and
every latent row (a "do not create this conjunction" constraint from
:mod:`aegis.fleetopt.latent`). Both kinds join the same two node types and
both are linear constraints of the same shape, ``u . (d + B x) >= threshold``,
which is why they are one edge set with a kind flag rather than two graphs.
The edge label is *exactly* the solver's row label -- ``resolve:<id>`` or
:attr:`aegis.fleetopt.latent.LatentConstraint.label` -- so a predicted
probability maps to a row without string surgery on either side.

Every physical quantity the physics loss (contract section 15.3) needs is
carried on the sample verbatim, in the units :mod:`aegis.fleetopt` uses: the
projected miss vector ``P d`` (km), the projected sensitivity ``P B`` (km per
km/s), the nominal linearisation direction and the required miss or latent
floor. The network's proposed burns are pushed through these with
:mod:`aegis.ml.torchphysics`; the sample never stores a pre-computed residual
the model could learn to copy.

Feature design
--------------
The features are a *description* of the geometry, not a preview of the
answer. Nothing derived from the solved plan appears in a feature. Quantities
with heavy tails -- miss distances that range from a few hundred metres to a
hundred-kilometre reachability gate, sensitivity norms that grow with lead
time -- are stored as ``log1p`` so that z-scoring does not collapse the
regime the active set actually lives in. Flags and one-hots are left as they
are: z-scoring a binary column makes it depend on the class balance of the
training split, which is the kind of train/inference skew the stored
statistics exist to prevent.

Normalisation statistics are fitted once on the training split and stored;
inference must reuse them. A :data:`SCHEMA_VERSION` mismatch raises rather
than warns because a model fed features in a different order is not "a bit
off", it is wrong in a way no metric downstream would notice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..constants import R_EARTH_KM
from ..core.objects import ObjectType, SpaceObject
from ..core.timebase import ensure_utc, seconds_between
from ..fleetopt.bplane import MissSensitivity
from ..fleetopt.dynamics import BurnGrid
from ..fleetopt.latent import LatentConstraint
from ..fleetopt.planners import PlanContext
from ..propagation.propagator import Sgp4Propagator
from ..scenarios import Scenario, scenario_digest

__all__ = [
    "SCHEMA_VERSION",
    "NODE_FEATURE_NAMES",
    "EDGE_FEATURE_NAMES",
    "NODE_PASSTHROUGH",
    "EDGE_PASSTHROUGH",
    "EDGE_KIND_RESOLVE",
    "EDGE_KIND_LATENT_GRID",
    "EDGE_KIND_LATENT_BPLANE",
    "EDGE_KIND_NAMES",
    "PC_LOG_FLOOR",
    "FeatureSchemaError",
    "GraphSample",
    "FeatureStats",
    "FeatureSpec",
    "current_spec",
    "tensorize",
    "fit_normalisation",
    "apply_normalisation",
]


class FeatureSchemaError(ValueError):
    """Raised when features, statistics or a checkpoint disagree on the schema."""


#: Bump whenever a feature is added, removed, reordered or its transform
#: changes. Stored statistics and checkpoints carry it and are refused on a
#: mismatch.
SCHEMA_VERSION = "aegis-ml-features-1"

#: ``log10(Pc)`` is floored here. Below 1e-15 the Alfano integral is at the
#: edge of double precision and the distinction is meaningless to a planner.
PC_LOG_FLOOR = -15.0

EDGE_KIND_RESOLVE = 0
EDGE_KIND_LATENT_GRID = 1
EDGE_KIND_LATENT_BPLANE = 2
EDGE_KIND_NAMES = ("resolve", "latent-grid", "latent-bplane")

NODE_FEATURE_NAMES: tuple[str, ...] = (
    "mean_motion_rad_s",
    "altitude_km",
    "eccentricity",
    "sin_inclination",
    "cos_inclination",
    "sin_raan",
    "cos_raan",
    "maneuverable",
    "is_debris",
    "remaining_dv_budget_km_s",
    "slot_count",
    "lead_to_earliest_tca_periods",
    "degree_resolve",
    "degree_total",
    "log10_max_pc",
)

EDGE_FEATURE_NAMES: tuple[str, ...] = (
    "log10_pc",
    "log1p_miss_km",
    "log1p_required_km",
    "log1p_shortfall_km",
    "relative_speed_km_s",
    "lead_from_first_slot_periods",
    "sigma_major_km",
    "sigma_minor_km",
    "hbr_km",
    "cos_velocity_bplane_a",
    "cos_velocity_bplane_b",
    "log1p_sensitivity_norm_a",
    "log1p_sensitivity_norm_b",
    "kind_resolve",
    "kind_latent_grid",
    "kind_latent_bplane",
    "intra_fleet",
    "controllable",
    "margin_km",
)

#: Columns that are flags or one-hots and are never z-scored.
NODE_PASSTHROUGH: frozenset[str] = frozenset({"maneuverable", "is_debris"})
EDGE_PASSTHROUGH: frozenset[str] = frozenset(
    {"kind_resolve", "kind_latent_grid", "kind_latent_bplane", "intra_fleet", "controllable"}
)

_ZERO = 1e-12
#: A feature whose training-split standard deviation is below this is treated
#: as constant: centred, but not scaled, so it cannot blow up at inference.
_STD_FLOOR = 1e-8


@dataclass
class GraphSample:
    """One scenario as plain tensors.

    Feature tensors are raw until :func:`apply_normalisation` has been run,
    which ``meta["normalised"]`` records. Physics tensors are never
    normalised: they feed :mod:`aegis.ml.torchphysics` and must stay in km
    and km/s.

    ``node_columns`` maps ``(node, slot, axis)`` to the column of the decision
    vector ``x`` that :class:`aegis.fleetopt.dynamics.BurnGrid` assigns it, or
    ``-1`` when the node has no such column (not in the grid, slot beyond its
    count, or a non-transverse axis on an along-track-only grid). Scattering
    ``node_dv`` through it rebuilds ``x`` exactly, which is how the physics
    loss connects a per-node prediction to the per-edge sensitivities.
    """

    node_features: Tensor
    node_ids: list[str]
    node_index: dict[str, int]
    edge_index: Tensor
    edge_features: Tensor
    edge_labels: list[str]
    edge_kind: Tensor
    slot_mask: Tensor
    node_columns: Tensor
    dv_mask: Tensor
    miss_vectors: Tensor
    sensitivities: Tensor
    directions: Tensor
    floors: Tensor
    n_vars: int
    edge_active: Tensor | None = None
    edge_dual: Tensor | None = None
    node_dv: Tensor | None = None
    global_feasible: Tensor | None = None
    global_cost: Tensor | None = None
    global_premium: Tensor | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def k_max(self) -> int:
        return int(self.slot_mask.shape[1])

    @property
    def has_labels(self) -> bool:
        return self.edge_active is not None

    @property
    def digest(self) -> str:
        return str(self.meta.get("digest", ""))

    def validate(self) -> None:
        """Shape and finiteness invariants; raises rather than returning False."""
        n, e, k = self.n_nodes, self.n_edges, self.k_max
        expect = {
            "node_features": (n, len(NODE_FEATURE_NAMES)),
            "edge_index": (2, e),
            "edge_features": (e, len(EDGE_FEATURE_NAMES)),
            "edge_kind": (e,),
            "slot_mask": (n, k),
            "node_columns": (n, 3 * k),
            "dv_mask": (n, 3 * k),
            "miss_vectors": (e, 3),
            "sensitivities": (e, 3, self.n_vars),
            "directions": (e, 3),
            "floors": (e,),
        }
        if self.edge_active is not None:
            expect["edge_active"] = (e,)
        if self.edge_dual is not None:
            expect["edge_dual"] = (e,)
        if self.node_dv is not None:
            expect["node_dv"] = (n, 3 * k)
        for name, shape in expect.items():
            tensor = getattr(self, name)
            if tuple(tensor.shape) != shape:
                raise FeatureSchemaError(f"{name} has shape {tuple(tensor.shape)}, expected {shape}")
        if len(self.edge_labels) != e:
            raise FeatureSchemaError("one edge label per edge is required")
        if len(self.node_ids) != n or len(self.node_index) != n:
            raise FeatureSchemaError("node_ids and node_index must cover every node")
        if e and (int(self.edge_index.max()) >= n or int(self.edge_index.min()) < 0):
            raise FeatureSchemaError("edge_index refers to a node outside the sample")
        if int(self.node_columns.max().item() if n and k else -1) >= self.n_vars:
            raise FeatureSchemaError("node_columns refers to a column beyond n_vars")
        for name in ("node_features", "edge_features", "miss_vectors", "sensitivities", "directions", "floors"):
            if not bool(torch.isfinite(getattr(self, name)).all()):
                raise FeatureSchemaError(f"{name} contains a non-finite value")
        for name in ("edge_active", "edge_dual", "node_dv", "global_feasible", "global_cost", "global_premium"):
            tensor = getattr(self, name)
            if tensor is not None and not bool(torch.isfinite(tensor).all()):
                raise FeatureSchemaError(f"{name} contains a non-finite value")

    def to_dict(self) -> dict[str, Any]:
        """Plain tensors, lists and dicts only, so ``torch.save`` needs no pickled classes."""
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "node_ids": list(self.node_ids),
            "edge_labels": list(self.edge_labels),
            "n_vars": int(self.n_vars),
            "meta": dict(self.meta),
        }
        for name in _TENSOR_FIELDS:
            payload[name] = getattr(self, name)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GraphSample:
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise FeatureSchemaError(
                f"sample was written with feature schema {version!r}; this code is {SCHEMA_VERSION!r}"
            )
        node_ids = list(payload["node_ids"])
        sample = cls(
            node_ids=node_ids,
            node_index={node_id: index for index, node_id in enumerate(node_ids)},
            edge_labels=list(payload["edge_labels"]),
            n_vars=int(payload["n_vars"]),
            meta=dict(payload.get("meta", {})),
            **{name: payload.get(name) for name in _TENSOR_FIELDS},
        )
        sample.validate()
        return sample


_TENSOR_FIELDS = (
    "node_features",
    "edge_index",
    "edge_features",
    "edge_kind",
    "slot_mask",
    "node_columns",
    "dv_mask",
    "miss_vectors",
    "sensitivities",
    "directions",
    "floors",
    "edge_active",
    "edge_dual",
    "node_dv",
    "global_feasible",
    "global_cost",
    "global_premium",
)


@dataclass(frozen=True)
class FeatureStats:
    """Per-feature mean and standard deviation, fitted once and stored.

    Passthrough columns carry mean 0 and std 1 so applying the statistics is
    a single affine map with no per-column branching.
    """

    schema_version: str
    node_mean: Tensor
    node_std: Tensor
    edge_mean: Tensor
    edge_std: Tensor
    fitted_samples: int = 0
    fitted_nodes: int = 0
    fitted_edges: int = 0

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise FeatureSchemaError(
                f"statistics carry schema {self.schema_version!r}; this code is {SCHEMA_VERSION!r}"
            )
        if tuple(self.node_mean.shape) != (len(NODE_FEATURE_NAMES),) or tuple(self.node_std.shape) != (
            len(NODE_FEATURE_NAMES),
        ):
            raise FeatureSchemaError("node statistics do not match the node feature count")
        if tuple(self.edge_mean.shape) != (len(EDGE_FEATURE_NAMES),) or tuple(self.edge_std.shape) != (
            len(EDGE_FEATURE_NAMES),
        ):
            raise FeatureSchemaError("edge statistics do not match the edge feature count")
        if bool((self.node_std <= 0).any()) or bool((self.edge_std <= 0).any()):
            raise FeatureSchemaError("every standard deviation must be strictly positive")

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "node_feature_names": list(NODE_FEATURE_NAMES),
            "edge_feature_names": list(EDGE_FEATURE_NAMES),
            "node_mean": [float(v) for v in self.node_mean],
            "node_std": [float(v) for v in self.node_std],
            "edge_mean": [float(v) for v in self.edge_mean],
            "edge_std": [float(v) for v in self.edge_std],
            "fitted_samples": int(self.fitted_samples),
            "fitted_nodes": int(self.fitted_nodes),
            "fitted_edges": int(self.fitted_edges),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> FeatureStats:
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise FeatureSchemaError(
                f"statistics carry schema {version!r}; this code is {SCHEMA_VERSION!r}"
            )
        if list(payload.get("node_feature_names", [])) != list(NODE_FEATURE_NAMES) or list(
            payload.get("edge_feature_names", [])
        ) != list(EDGE_FEATURE_NAMES):
            raise FeatureSchemaError("stored feature names differ from this code's feature order")
        as_tensor = lambda key: torch.tensor(payload[key], dtype=torch.float64)  # noqa: E731
        return cls(
            schema_version=version,
            node_mean=as_tensor("node_mean"),
            node_std=as_tensor("node_std"),
            edge_mean=as_tensor("edge_mean"),
            edge_std=as_tensor("edge_std"),
            fitted_samples=int(payload.get("fitted_samples", 0)),
            fitted_nodes=int(payload.get("fitted_nodes", 0)),
            fitted_edges=int(payload.get("fitted_edges", 0)),
        )


@dataclass(frozen=True)
class FeatureSpec:
    """The versioned feature layout, optionally with fitted statistics."""

    schema_version: str = SCHEMA_VERSION
    node_feature_names: tuple[str, ...] = NODE_FEATURE_NAMES
    edge_feature_names: tuple[str, ...] = EDGE_FEATURE_NAMES
    node_passthrough: frozenset[str] = NODE_PASSTHROUGH
    edge_passthrough: frozenset[str] = EDGE_PASSTHROUGH
    stats: FeatureStats | None = None

    @property
    def n_node_features(self) -> int:
        return len(self.node_feature_names)

    @property
    def n_edge_features(self) -> int:
        return len(self.edge_feature_names)

    def check(self, other_version: str) -> None:
        if other_version != self.schema_version:
            raise FeatureSchemaError(
                f"feature schema {other_version!r} does not match {self.schema_version!r}"
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "node_feature_names": list(self.node_feature_names),
            "edge_feature_names": list(self.edge_feature_names),
            "node_passthrough": sorted(self.node_passthrough),
            "edge_passthrough": sorted(self.edge_passthrough),
            "stats": self.stats.to_json() if self.stats is not None else None,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> FeatureSpec:
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise FeatureSchemaError(
                f"spec carries schema {version!r}; this code is {SCHEMA_VERSION!r}"
            )
        if tuple(payload["node_feature_names"]) != NODE_FEATURE_NAMES or tuple(
            payload["edge_feature_names"]
        ) != EDGE_FEATURE_NAMES:
            raise FeatureSchemaError("stored feature names differ from this code's feature order")
        stats = payload.get("stats")
        return cls(stats=FeatureStats.from_json(stats) if stats is not None else None)


def current_spec(stats: FeatureStats | None = None) -> FeatureSpec:
    return FeatureSpec(stats=stats)


# ---------------------------------------------------------------------------
# Tensorisation
# ---------------------------------------------------------------------------


def _log10_pc(probability: float) -> float:
    if not math.isfinite(probability) or probability <= 0.0:
        return PC_LOG_FLOOR
    return max(PC_LOG_FLOOR, math.log10(probability))


def _cos_to_bplane(velocity: np.ndarray, relative_velocity: np.ndarray) -> float:
    """``sqrt(1 - (v_hat . w_hat)^2)``: the fraction of ``v`` lying in the B-plane.

    A vanishing relative velocity has no encounter plane, and
    :func:`aegis.fleetopt.bplane.bplane_projector` then keeps every direction,
    so 1 is the consistent answer.
    """
    speed = float(np.linalg.norm(velocity))
    relative_speed = float(np.linalg.norm(relative_velocity))
    if speed < _ZERO or relative_speed < _ZERO:
        return 1.0
    cosine = float(np.dot(velocity, relative_velocity)) / (speed * relative_speed)
    return float(math.sqrt(max(0.0, 1.0 - cosine * cosine)))


def _is_intra_fleet(a: SpaceObject, b: SpaceObject) -> bool:
    """Mirrors :attr:`aegis.core.conjunction.Conjunction.is_intra_fleet` for any pair."""
    if a.operator is None or b.operator is None:
        return False
    return a.operator.identifier == b.operator.identifier and a.operator.maneuverable


def _endpoint_sensitivity_norm(grid: BurnGrid, sensitivity: np.ndarray, object_id: str) -> float:
    if not grid.has(object_id):
        return 0.0
    columns = grid.columns(object_id)
    if not columns:
        return 0.0
    return float(np.linalg.norm(sensitivity[:, columns]))


def _period_s(grid: BurnGrid, obj: SpaceObject | None, object_id: str) -> float:
    if grid.has(object_id):
        return 2.0 * math.pi / grid.mean_motion[object_id]
    if obj is not None and obj.elements is not None and obj.elements.mean_motion_rev_per_day > 0.0:
        return float(obj.elements.period_s)
    return 0.0


def _lead_from_first_slot_periods(
    grid: BurnGrid, epoch: datetime, endpoints: tuple[str, str]
) -> float:
    """Lead time from the earliest burn slot of either endpoint, in that satellite's orbits.

    Lead time in orbits rather than seconds because the CW secular term grows
    as ``3 n sigma`` (see :mod:`aegis.fleetopt.dynamics`): one orbit of lead
    buys the same fractional displacement at any altitude.
    """
    best = 0.0
    for object_id in endpoints:
        if not grid.has(object_id):
            continue
        lead_s = seconds_between(grid.earliest_epoch(object_id), epoch)
        periods = lead_s / (2.0 * math.pi / grid.mean_motion[object_id])
        best = max(best, periods)
    return float(best)


@dataclass
class _EdgeRow:
    label: str
    a: str
    b: str
    kind: int
    epoch: datetime
    features: list[float]
    miss_vector: np.ndarray
    sensitivity: np.ndarray
    direction: np.ndarray
    floor: float


def _resolve_edge(
    sensitivity: MissSensitivity,
    context: PlanContext,
    by_id: dict[str, SpaceObject],
    conjunction_lookup: dict[str, Any],
) -> _EdgeRow:
    conjunction = conjunction_lookup[sensitivity.conjunction_id]
    grid = context.grid
    velocity_a = np.asarray(conjunction.primary_state.velocity_km_s, dtype=float)
    velocity_b = np.asarray(conjunction.secondary_state.velocity_km_s, dtype=float)
    relative = velocity_b - velocity_a
    a, b = sensitivity.primary_id, sensitivity.secondary_id
    features = [
        _log10_pc(sensitivity.probability_before),
        math.log1p(max(0.0, sensitivity.nominal_miss_km)),
        math.log1p(max(0.0, sensitivity.required_miss_km)),
        math.log1p(sensitivity.shortfall_km),
        float(sensitivity.relative_speed_km_s),
        _lead_from_first_slot_periods(grid, sensitivity.tca, (a, b)),
        float(sensitivity.sigma_major_km),
        float(sensitivity.sigma_minor_km),
        float(sensitivity.hard_body_radius_km),
        _cos_to_bplane(velocity_a, relative),
        _cos_to_bplane(velocity_b, relative),
        math.log1p(_endpoint_sensitivity_norm(grid, sensitivity.sensitivity, a)),
        math.log1p(_endpoint_sensitivity_norm(grid, sensitivity.sensitivity, b)),
        1.0,
        0.0,
        0.0,
        1.0 if _is_intra_fleet(by_id[a], by_id[b]) else 0.0,
        1.0 if sensitivity.is_controllable else 0.0,
        float(sensitivity.nominal_miss_km - sensitivity.required_miss_km),
    ]
    return _EdgeRow(
        label=f"resolve:{sensitivity.conjunction_id}",
        a=a,
        b=b,
        kind=EDGE_KIND_RESOLVE,
        epoch=sensitivity.tca,
        features=features,
        miss_vector=sensitivity.miss_vector_km,
        sensitivity=sensitivity.sensitivity,
        direction=sensitivity.nominal_direction,
        floor=float(sensitivity.required_miss_km),
    )


def _latent_edge(
    constraint: LatentConstraint,
    context: PlanContext,
    by_id: dict[str, SpaceObject],
    velocity_a: np.ndarray,
    velocity_b: np.ndarray,
) -> _EdgeRow:
    grid = context.grid
    a, b = constraint.object_a, constraint.object_b
    relative = velocity_b - velocity_a
    kind = EDGE_KIND_LATENT_GRID if constraint.kind == "grid" else EDGE_KIND_LATENT_BPLANE
    shortfall = max(0.0, constraint.floor_km - constraint.separation_km)
    features = [
        PC_LOG_FLOOR,
        math.log1p(max(0.0, constraint.separation_km)),
        math.log1p(max(0.0, constraint.floor_km)),
        math.log1p(shortfall),
        float(constraint.relative_speed_km_s),
        _lead_from_first_slot_periods(grid, constraint.epoch, (a, b)),
        0.0,
        0.0,
        0.0,
        _cos_to_bplane(velocity_a, relative),
        _cos_to_bplane(velocity_b, relative),
        math.log1p(_endpoint_sensitivity_norm(grid, constraint.sensitivity, a)),
        math.log1p(_endpoint_sensitivity_norm(grid, constraint.sensitivity, b)),
        0.0,
        1.0 if kind == EDGE_KIND_LATENT_GRID else 0.0,
        1.0 if kind == EDGE_KIND_LATENT_BPLANE else 0.0,
        1.0 if _is_intra_fleet(by_id[a], by_id[b]) else 0.0,
        1.0 if constraint.is_controllable else 0.0,
        float(constraint.slack_km),
    ]
    return _EdgeRow(
        label=constraint.label,
        a=a,
        b=b,
        kind=kind,
        epoch=constraint.epoch,
        features=features,
        miss_vector=constraint.offset_km,
        sensitivity=constraint.sensitivity,
        direction=constraint.direction,
        floor=float(constraint.floor_km),
    )


def _latent_velocities(
    scenario: Scenario, context: PlanContext
) -> dict[tuple[str, datetime], np.ndarray]:
    """Inertial velocity of each latent endpoint at its row epoch.

    :class:`aegis.fleetopt.latent.LatentConstraint` keeps only the relative
    speed, so the endpoint velocities are re-propagated here. This is the one
    place tensorisation runs SGP4; it is a few thousand single-epoch
    evaluations at most and the alternative -- widening ``LatentConstraint`` --
    would touch a certified module for a feature's convenience.
    """
    needed: set[tuple[str, datetime]] = set()
    for constraint in context.latent:
        needed.add((constraint.object_a, constraint.epoch))
        needed.add((constraint.object_b, constraint.epoch))
    if not needed:
        return {}
    propagator = Sgp4Propagator(scenario.objects)
    index_of = {object_id: index for index, object_id in enumerate(propagator.object_ids)}
    velocities: dict[tuple[str, datetime], np.ndarray] = {}
    for object_id, epoch in sorted(needed, key=lambda item: (item[0], item[1])):
        state = propagator.propagate_one(index_of[object_id], epoch)
        velocities[(object_id, epoch)] = np.asarray(state.velocity_km_s, dtype=float)
    return velocities


def _node_row(
    obj: SpaceObject,
    context: PlanContext,
    *,
    planning_epoch: datetime,
    earliest_epoch: datetime | None,
    degree_resolve: int,
    degree_total: int,
    max_pc: float,
    notes: list[str],
) -> list[float]:
    grid = context.grid
    elements = obj.elements
    if elements is None or elements.mean_motion_rev_per_day <= 0.0:
        notes.append(f"{obj.object_id}: no orbital elements; orbit features zeroed")
        mean_motion = altitude = eccentricity = 0.0
        inclination = raan = 0.0
    else:
        mean_motion = float(elements.mean_motion_rad_s)
        altitude = float(elements.semi_major_axis_km - R_EARTH_KM)
        eccentricity = float(elements.eccentricity)
        inclination = math.radians(float(elements.inclination_deg))
        raan = math.radians(float(elements.raan_deg))

    in_grid = grid.has(obj.object_id)
    budget = float(context.request.dv_budget_km_s) if in_grid else 0.0
    slots = float(len(grid.slot_epochs(obj.object_id))) if in_grid else 0.0

    period = _period_s(grid, obj, obj.object_id)
    if earliest_epoch is None:
        lead_s = float(context.request.window_duration_s)
    else:
        lead_s = seconds_between(planning_epoch, earliest_epoch)
    lead_periods = lead_s / period if period > 0.0 else 0.0

    return [
        mean_motion,
        altitude,
        eccentricity,
        math.sin(inclination),
        math.cos(inclination),
        math.sin(raan),
        math.cos(raan),
        1.0 if obj.is_maneuverable else 0.0,
        1.0 if obj.object_type == ObjectType.DEBRIS else 0.0,
        budget,
        slots,
        float(lead_periods),
        float(degree_resolve),
        float(degree_total),
        _log10_pc(max_pc),
    ]


def _node_columns(grid: BurnGrid, node_ids: list[str], k_max: int) -> tuple[Tensor, Tensor]:
    columns = torch.full((len(node_ids), 3 * k_max), -1, dtype=torch.long)
    slots = torch.zeros((len(node_ids), k_max), dtype=torch.bool)
    axes = range(3) if grid.axes == 3 else (1,)
    for row, object_id in enumerate(node_ids):
        if not grid.has(object_id):
            continue
        for slot in range(len(grid.slot_epochs(object_id))):
            slots[row, slot] = True
            for axis in axes:
                columns[row, 3 * slot + axis] = grid.index(object_id, slot, axis)
    return columns, slots


def tensorize(
    scenario: Scenario,
    context: PlanContext,
    labels: dict[str, Any] | None = None,
    *,
    dtype: torch.dtype = torch.float32,
) -> GraphSample:
    """Build the :class:`GraphSample` for one planned scenario.

    ``context`` must have been built from this scenario's objects; the node
    set is the scenario's objects in sorted-id order so that two samples of
    the same scenario are byte-identical. ``labels`` is the dict produced by
    :func:`aegis.ml.dataset.build_labels`; when given, every edge must appear
    in it -- a missing row is a mismatch between the labelled problem and the
    tensorised one, and is raised rather than filled with zero.
    """
    objects = sorted(scenario.objects, key=lambda obj: obj.object_id)
    if len({obj.object_id for obj in objects}) != len(objects):
        raise FeatureSchemaError("scenario contains duplicate object ids")
    by_id = {obj.object_id: obj for obj in objects}
    node_ids = [obj.object_id for obj in objects]
    node_index = {object_id: index for index, object_id in enumerate(node_ids)}
    grid = context.grid
    for sat_id in grid.satellite_ids:
        if sat_id not in node_index:
            raise FeatureSchemaError(f"burn grid satellite {sat_id!r} is not an object of the scenario")

    notes: list[str] = []
    conjunction_lookup = {
        entry.conjunction.conjunction_id: entry.conjunction for entry in context.assessed.entries
    }

    rows: list[_EdgeRow] = []
    for sensitivity in context.sensitivities:
        if sensitivity.conjunction_id not in conjunction_lookup:
            raise FeatureSchemaError(
                f"sensitivity {sensitivity.conjunction_id!r} has no conjunction in the assessed catalog"
            )
        rows.append(_resolve_edge(sensitivity, context, by_id, conjunction_lookup))
    velocities = _latent_velocities(scenario, context)
    for constraint in context.latent:
        rows.append(
            _latent_edge(
                constraint,
                context,
                by_id,
                velocities[(constraint.object_a, constraint.epoch)],
                velocities[(constraint.object_b, constraint.epoch)],
            )
        )
    for row in rows:
        if row.a not in node_index or row.b not in node_index:
            raise FeatureSchemaError(f"edge {row.label!r} joins an object outside the scenario")
    seen: set[str] = set()
    for row in rows:
        if row.label in seen:
            raise FeatureSchemaError(f"duplicate edge label {row.label!r}")
        seen.add(row.label)

    planning_epoch = ensure_utc(context.request.now) if context.request.now is not None else ensure_utc(
        context.request.window_start
    )
    earliest: dict[str, datetime] = {}
    degree_resolve = {object_id: 0 for object_id in node_ids}
    degree_total = {object_id: 0 for object_id in node_ids}
    max_pc = {object_id: 0.0 for object_id in node_ids}
    for row in rows:
        for object_id in (row.a, row.b):
            degree_total[object_id] += 1
            previous = earliest.get(object_id)
            if previous is None or row.epoch < previous:
                earliest[object_id] = row.epoch
    for sensitivity in context.sensitivities:
        for object_id in (sensitivity.primary_id, sensitivity.secondary_id):
            degree_resolve[object_id] += 1
            max_pc[object_id] = max(max_pc[object_id], float(sensitivity.probability_before))

    node_rows = [
        _node_row(
            obj,
            context,
            planning_epoch=planning_epoch,
            earliest_epoch=earliest.get(obj.object_id),
            degree_resolve=degree_resolve[obj.object_id],
            degree_total=degree_total[obj.object_id],
            max_pc=max_pc[obj.object_id],
            notes=notes,
        )
        for obj in objects
    ]

    n_vars = int(grid.n_vars)
    k_max = int(grid.max_slots)
    n_edges = len(rows)
    node_columns, slot_mask = _node_columns(grid, node_ids, k_max)

    node_features = torch.tensor(node_rows, dtype=dtype).reshape(len(node_ids), len(NODE_FEATURE_NAMES))
    edge_features = torch.tensor([row.features for row in rows], dtype=dtype).reshape(
        n_edges, len(EDGE_FEATURE_NAMES)
    )
    edge_index = torch.tensor(
        [[node_index[row.a] for row in rows], [node_index[row.b] for row in rows]], dtype=torch.long
    ).reshape(2, n_edges)
    edge_kind = torch.tensor([row.kind for row in rows], dtype=torch.long).reshape(n_edges)
    miss_vectors = torch.tensor(
        np.array([row.miss_vector for row in rows], dtype=float).reshape(n_edges, 3), dtype=dtype
    )
    sensitivities = torch.tensor(
        np.array([row.sensitivity for row in rows], dtype=float).reshape(n_edges, 3, n_vars), dtype=dtype
    )
    directions = torch.tensor(
        np.array([row.direction for row in rows], dtype=float).reshape(n_edges, 3), dtype=dtype
    )
    floors = torch.tensor([row.floor for row in rows], dtype=dtype).reshape(n_edges)

    meta: dict[str, Any] = {
        "scenario_id": scenario.scenario_id,
        "digest": scenario_digest(scenario),
        "family": scenario.family,
        "seed": int(scenario.seed),
        "n_vars": n_vars,
        "K_max": k_max,
        "axes": int(grid.axes),
        "n_resolve": len(context.sensitivities),
        "n_latent": len(context.latent),
        "planning_epoch": planning_epoch.isoformat(),
        "schema_version": SCHEMA_VERSION,
        "normalised": False,
        "notes": notes,
    }

    sample = GraphSample(
        node_features=node_features,
        node_ids=node_ids,
        node_index=node_index,
        edge_index=edge_index,
        edge_features=edge_features,
        edge_labels=[row.label for row in rows],
        edge_kind=edge_kind,
        slot_mask=slot_mask,
        node_columns=node_columns,
        dv_mask=node_columns >= 0,
        miss_vectors=miss_vectors,
        sensitivities=sensitivities,
        directions=directions,
        floors=floors,
        n_vars=n_vars,
        meta=meta,
    )
    if labels is not None:
        _attach_labels(sample, labels, dtype)
    sample.validate()
    return sample


def _attach_labels(sample: GraphSample, labels: dict[str, Any], dtype: torch.dtype) -> None:
    active = labels["edge_active"]
    duals = labels["edge_dual"]
    missing = [label for label in sample.edge_labels if label not in active or label not in duals]
    if missing:
        raise FeatureSchemaError(
            f"{len(missing)} edge(s) have no label, e.g. {missing[0]!r}; the labelled problem and "
            "the tensorised one differ"
        )
    n_edges = sample.n_edges
    sample.edge_active = torch.tensor(
        [float(active[label]) for label in sample.edge_labels], dtype=dtype
    ).reshape(n_edges)
    sample.edge_dual = torch.tensor(
        [float(duals[label]) for label in sample.edge_labels], dtype=dtype
    ).reshape(n_edges)

    width = 3 * sample.k_max
    node_dv = torch.zeros((sample.n_nodes, width), dtype=dtype)
    per_node = labels["node_dv"]
    for object_id, values in per_node.items():
        if object_id not in sample.node_index:
            raise FeatureSchemaError(f"node_dv label for unknown object {object_id!r}")
        vector = torch.as_tensor(np.asarray(values, dtype=float), dtype=dtype).reshape(-1)
        if vector.numel() != width:
            raise FeatureSchemaError(
                f"node_dv for {object_id!r} has {vector.numel()} entries, expected {width}"
            )
        node_dv[sample.node_index[object_id]] = vector
    sample.node_dv = node_dv

    sample.global_feasible = torch.tensor(1.0 if labels["feasible"] else 0.0, dtype=dtype)
    sample.global_cost = torch.tensor(float(labels["cost"]), dtype=dtype)
    premium = labels.get("premium")
    sample.global_premium = (
        torch.tensor(float(premium), dtype=dtype) if premium is not None and math.isfinite(premium) else None
    )
    sample.meta["premium_reason"] = str(labels.get("premium_reason", ""))
    sample.meta["label_status"] = str(labels.get("status", ""))
    sample.meta["slack_absorbed_km"] = float(labels.get("slack_absorbed_km", 0.0))
    sample.meta["n_active"] = int(labels.get("n_active", 0))
    sample.meta["labelled"] = True


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def _moments(rows: Tensor, passthrough: tuple[bool, ...]) -> tuple[Tensor, Tensor]:
    width = rows.shape[1]
    mean = torch.zeros(width, dtype=torch.float64)
    std = torch.ones(width, dtype=torch.float64)
    if rows.shape[0] == 0:
        return mean, std
    values = rows.to(torch.float64)
    fitted_mean = values.mean(dim=0)
    fitted_std = values.std(dim=0, unbiased=False) if rows.shape[0] > 1 else torch.zeros(width, dtype=torch.float64)
    for column in range(width):
        if passthrough[column]:
            continue
        mean[column] = fitted_mean[column]
        std[column] = fitted_std[column] if float(fitted_std[column]) > _STD_FLOOR else 1.0
    return mean, std


def fit_normalisation(samples: list[GraphSample]) -> FeatureStats:
    """Fit per-feature mean and std on raw samples -- the training split only.

    Refuses normalised samples: fitting on already-centred data would store
    statistics that undo nothing and silently shift every downstream number.
    """
    if not samples:
        raise FeatureSchemaError("cannot fit normalisation on zero samples")
    for sample in samples:
        if sample.meta.get("normalised"):
            raise FeatureSchemaError(f"sample {sample.digest!r} is already normalised")
        if sample.meta.get("schema_version") != SCHEMA_VERSION:
            raise FeatureSchemaError("sample schema does not match this code")
    node_rows = torch.cat([sample.node_features for sample in samples], dim=0)
    edge_rows = torch.cat([sample.edge_features for sample in samples], dim=0)
    node_pass = tuple(name in NODE_PASSTHROUGH for name in NODE_FEATURE_NAMES)
    edge_pass = tuple(name in EDGE_PASSTHROUGH for name in EDGE_FEATURE_NAMES)
    node_mean, node_std = _moments(node_rows, node_pass)
    edge_mean, edge_std = _moments(edge_rows, edge_pass)
    return FeatureStats(
        schema_version=SCHEMA_VERSION,
        node_mean=node_mean,
        node_std=node_std,
        edge_mean=edge_mean,
        edge_std=edge_std,
        fitted_samples=len(samples),
        fitted_nodes=int(node_rows.shape[0]),
        fitted_edges=int(edge_rows.shape[0]),
    )


def apply_normalisation(sample: GraphSample, stats: FeatureStats) -> GraphSample:
    """Return a copy of ``sample`` with z-scored features and the same physics tensors."""
    if stats.schema_version != SCHEMA_VERSION or sample.meta.get("schema_version") != SCHEMA_VERSION:
        raise FeatureSchemaError("sample and statistics must both carry the current schema version")
    if sample.meta.get("normalised"):
        raise FeatureSchemaError(f"sample {sample.digest!r} is already normalised")
    dtype = sample.node_features.dtype
    node_features = (sample.node_features - stats.node_mean.to(dtype)) / stats.node_std.to(dtype)
    edge_features = (sample.edge_features - stats.edge_mean.to(dtype)) / stats.edge_std.to(dtype)
    payload = sample.to_dict()
    payload["node_features"] = node_features
    payload["edge_features"] = edge_features
    payload["meta"] = {**sample.meta, "normalised": True}
    return GraphSample.from_dict(payload)
