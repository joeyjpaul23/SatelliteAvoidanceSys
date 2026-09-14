"""Labelled graph datasets: exact solves as supervision, split by scenario.

Why the labels are what they are
--------------------------------
The network is trained to *predict* what the optimizer would decide, and the
optimizer -- not the network -- remains the authority at planning time
(contract section 15; novelty ledger claim C6, "predict-then-certify"). So
every target here is read off an exact solve of the full fleet-safe program by
:func:`aegis.fleetopt.solver.sequential_solve`:

* ``edge_active`` is :func:`aegis.fleetopt.certify.binding_rows` -- rows with
  a strictly positive dual. It is the set :func:`aegis.fleetopt.solver.lazy_solve`
  guesses, and a wrong guess costs a lazy round, never feasibility.
* ``edge_dual`` is the dual itself: the marginal delta-v cost of tightening
  that row by one kilometre, the number Proposition 6 in
  :mod:`aegis.fleetopt.certify` turns into the exact-penalty threshold.
* ``node_dv`` is the optimal decision vector, laid out per satellite and slot
  so the physics loss can rebuild ``x`` through ``node_columns``.
* ``global_cost``, ``global_feasible`` and ``global_premium`` are the scalar
  summaries the experiments report; the premium needs the second,
  ``fuel-only`` solve and is taken from :func:`aegis.fleetopt.pareto.safety_premium`
  so that "undefined" stays undefined rather than becoming a zero.

Why the split is by scenario digest
-----------------------------------
Every edge of one scenario shares that scenario's solution: the dual of one
latent row is determined jointly with every other row of the same program.
Splitting edges at random would put a graph's own optimum into its training
set and the held-out metric would measure memorisation. The split key is
:func:`aegis.scenarios.scenario_digest`, hashed into a bucket, so it is a pure
function of scenario content -- stable across runs, machines and resumptions,
and independent of the order scenarios were generated in.

Storage
-------
Each sample is one ``torch.save`` of :meth:`GraphSample.to_dict` -- plain
tensors, lists and dicts, loadable with ``weights_only=True`` -- keyed by
scenario digest in an :class:`aegis.store.ArtifactStore`. A local directory
is wrapped in :class:`aegis.store.LocalArtifactStore`, whose content-addressed
layout makes a re-run of the same scenario a no-op write. The manifest is one
JSON document holding the schema version, every entry with its split, every
failure with its traceback, and the normalisation statistics fitted on the
training split.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor

from ..core.timebase import format_epoch, shift, utc_now
from ..fleetopt.certify import binding_rows
from ..fleetopt.pareto import PremiumRecord, safety_premium
from ..fleetopt.planners import PlanContext, PlanRequest, build_context
from ..fleetopt.solver import ScpResult, sequential_solve
from ..ingest.synthetic import SyntheticAuthorization
from ..risk.batch import assess_catalog
from ..scenarios import Scenario, generate, scenario_digest
from ..screening import screen
from ..store import ArtifactStore, LocalArtifactStore
from .features import (
    SCHEMA_VERSION,
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    FeatureSchemaError,
    FeatureStats,
    GraphSample,
    apply_normalisation,
    fit_normalisation,
    tensorize,
)

__all__ = [
    "DatasetError",
    "SPLITS",
    "DEFAULT_SPLIT_FRACTIONS",
    "DEFAULT_LEAD_PAD_ORBITS",
    "split_for_digest",
    "padded_window",
    "prepare_context",
    "build_labels",
    "DatasetEntry",
    "DatasetIndex",
    "generate_dataset",
    "load_manifest",
    "load_dataset",
    "GraphBatch",
    "collate",
]


class DatasetError(RuntimeError):
    """A dataset could not be written, read or batched as asked."""


SPLITS = ("train", "val", "test")
DEFAULT_SPLIT_FRACTIONS = (0.8, 0.1, 0.1)

#: How far before its designed window a scenario is planned from. The
#: generated families place their TCAs 0.01 to 1.01 orbits after their own
#: window start, and a burn needs at least ``MIN_LEAD_TIME_ORBITS`` of lead,
#: so planning from the window start leaves every satellite without a slot.
#: :mod:`aegis.experiments.runner` pads by three orbits for exactly this
#: reason; the same value is used here so the labels describe the problem the
#: benchmark actually solves.
DEFAULT_LEAD_PAD_ORBITS = 3.0

_MANIFEST_KEY = "manifest.json"
_SAMPLE_PREFIX = "samples"
_SLACK_TOL_KM = 1e-9
_RESIDUAL_TOL_KM = 1e-9


def split_for_digest(digest: str, fractions: Sequence[float] = DEFAULT_SPLIT_FRACTIONS) -> str:
    """Deterministic ``train``/``val``/``test`` bucket for one scenario digest.

    The digest is re-hashed so that consecutive seeds of one family -- whose
    digests share no structure anyway -- cannot cluster in a split, and the
    bucket edge is a pure function of the digest and the fractions.
    """
    if len(fractions) != 3 or any(f < 0.0 for f in fractions):
        raise DatasetError("split fractions must be three non-negative numbers")
    total = float(sum(fractions))
    if total <= 0.0:
        raise DatasetError("split fractions must not all be zero")
    bucket = int(hashlib.sha256(digest.encode("utf-8")).hexdigest()[:12], 16) / float(16**12)
    train, val = fractions[0] / total, fractions[1] / total
    if bucket < train:
        return "train"
    if bucket < train + val:
        return "val"
    return "test"


def padded_window(scenario: Scenario, lead_pad_orbits: float) -> tuple[datetime, float]:
    """Start the planning window ``lead_pad_orbits`` median orbits early.

    Mirrors the padding :mod:`aegis.experiments.runner` applies before
    screening, and for the same reason: padding backwards keeps the
    generated geometry -- and the structural claim each family makes about
    it -- exactly as designed, while giving the burn grid room to exist.
    """
    periods = [
        obj.elements.period_s
        for obj in scenario.objects
        if obj.elements is not None and obj.elements.mean_motion_rev_per_day > 0.0
    ]
    if not periods or lead_pad_orbits <= 0.0:
        return scenario.window_start, float(scenario.window_duration_s)
    pad_s = float(lead_pad_orbits) * float(np.median(periods))
    return shift(scenario.window_start, -pad_s), float(scenario.window_duration_s) + pad_s


def prepare_context(
    scenario: Scenario,
    *,
    lead_pad_orbits: float = DEFAULT_LEAD_PAD_ORBITS,
    screening_step_s: float | None = None,
    request_overrides: dict[str, Any] | None = None,
) -> PlanContext:
    """Screen, assess and build the planner-independent context for one scenario.

    ``request_overrides`` are applied as attributes of the
    :class:`aegis.fleetopt.planners.PlanRequest` after construction, so the
    dataset can be built for a non-default exclusion radius or latent mode
    and the manifest records exactly which.
    """
    window_start, window_duration_s = padded_window(scenario, lead_pad_orbits)
    screen_kwargs: dict[str, Any] = {"box_km": scenario.screening_box_km}
    if screening_step_s is not None:
        screen_kwargs["step_s"] = float(screening_step_s)
    conjunctions = screen(scenario.objects, window_start, window_duration_s, **screen_kwargs)
    assessed = assess_catalog(conjunctions, objects=scenario.objects)
    request = PlanRequest(
        assessed=assessed,
        objects=scenario.objects,
        window_start=window_start,
        window_duration_s=window_duration_s,
        now=window_start,
        screening_box_km=scenario.screening_box_km,
    )
    for name, value in (request_overrides or {}).items():
        if not hasattr(request, name):
            raise DatasetError(f"PlanRequest has no field {name!r} to override")
        setattr(request, name, value)
    return build_context(request)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def _charged_delta_v_km_s(result: ScpResult) -> float:
    """Total charged delta-v, read through the cost model like :mod:`aegis.fleetopt.pareto`."""
    data = result.data
    z = result.solution.z[: data.n_lifted]
    return float(
        sum(
            data.cost_model.charged_magnitude(z, sat_id, slot)
            for (sat_id, slot) in data.cost_model.magnitude_columns
        )
    )


def _node_dv_layout(context: PlanContext, x: np.ndarray) -> dict[str, list[float]]:
    """``x`` regrouped as ``(slot, axis)`` blocks per satellite, ``3 * K_max`` wide."""
    grid = context.grid
    width = 3 * grid.max_slots
    layout: dict[str, list[float]] = {}
    axes = range(3) if grid.axes == 3 else (1,)
    for sat_id in grid.satellite_ids:
        values = [0.0] * width
        for slot in range(len(grid.slot_epochs(sat_id))):
            for axis in axes:
                values[3 * slot + axis] = float(x[grid.index(sat_id, slot, axis)])
        layout[sat_id] = values
    return layout


def _slack_total(result: ScpResult) -> float:
    return float(
        sum(result.solution.slack(result.data, "resolve").values())
        + sum(result.solution.slack(result.data, "latent").values())
    )


def _count_resolved(context: PlanContext, x: np.ndarray) -> int:
    return sum(1 for s in context.sensitivities if s.residual_km(x) >= -_RESIDUAL_TOL_KM)


def _count_induced(context: PlanContext, x: np.ndarray) -> int:
    return sum(1 for c in context.latent if c.residual_km(x) < -_RESIDUAL_TOL_KM)


def build_labels(
    context: PlanContext, *, max_iterations: int | None = None, scenario_id: str = ""
) -> dict[str, Any]:
    """Solve exactly and extract every training target for one scenario.

    Two solves: the full ``fleet-safe`` program (resolve plus every latent
    row) supplies the active set, duals, decision vector, cost and
    feasibility; a ``fuel-only`` solve (resolve rows only) supplies the
    denominator of the safety premium. Both use the same
    :meth:`PlanContext.problem` the planners use, so a label is exactly what
    :class:`aegis.fleetopt.planners.FleetSafePlanner` would have returned.

    A scenario with nothing to plan -- no conjunctions or no usable burn slot
    -- is a legitimate sample: its labels are empty, its cost is zero and its
    premium is undefined with the reason recorded.
    """
    iterations = int(max_iterations if max_iterations is not None else context.request.scp_iterations)
    edge_labels = [f"resolve:{s.conjunction_id}" for s in context.sensitivities] + [
        c.label for c in context.latent
    ]
    labels: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "edge_active": {label: 0.0 for label in edge_labels},
        "edge_dual": {label: 0.0 for label in edge_labels},
        "node_dv": {},
        "feasible": True,
        "cost": 0.0,
        "premium": None,
        "premium_reason": "",
        "status": "trivial",
        "x": [],
        "delta_v_safe_km_s": 0.0,
        "delta_v_fuel_km_s": 0.0,
        "n_active": 0,
        "solve_time_s": 0.0,
        "scp": None,
        "notes": [],
    }

    if not context.sensitivities or context.grid.n_vars == 0:
        reason = "no conjunctions to plan for" if not context.sensitivities else (
            "no maneuverable satellite has a usable burn slot"
        )
        labels["premium_reason"] = reason
        labels["notes"].append(reason)
        # With nothing to move, every resolve row is unresolved unless its
        # nominal geometry already clears the requirement.
        if context.sensitivities:
            x = np.zeros(max(0, context.grid.n_vars))
            labels["feasible"] = _count_resolved(context, x) == len(context.sensitivities)
        labels["node_dv"] = _node_dv_layout(context, np.zeros(max(0, context.grid.n_vars)))
        return labels

    started = time.perf_counter()
    safe = sequential_solve(context.problem(latent=list(context.latent)), max_iterations=iterations)
    fuel = sequential_solve(context.problem(latent=[]), max_iterations=iterations)
    labels["solve_time_s"] = time.perf_counter() - started
    labels["status"] = safe.solution.status
    labels["scp"] = safe.summary()
    labels["notes"].extend(safe.notes)

    if not safe.solution.ok:
        raise DatasetError(
            f"fleet-safe solve returned {safe.solution.status!r}: {safe.solution.message}"
        )
    if not safe.solution.has_duals:
        raise DatasetError(
            f"backend {safe.solution.backend!r} returned no duals; the active-set label is undefined"
        )

    x = np.asarray(safe.x, dtype=float)
    active = binding_rows(safe.solution)
    for label in edge_labels:
        if label not in safe.solution.duals:
            raise DatasetError(f"row {label!r} is missing from the solved program's duals")
        labels["edge_dual"][label] = float(safe.solution.duals[label])
        labels["edge_active"][label] = 1.0 if label in active else 0.0
    labels["n_active"] = int(sum(labels["edge_active"].values()))
    labels["node_dv"] = _node_dv_layout(context, x)
    labels["x"] = [float(v) for v in x]
    labels["cost"] = float(safe.solution.objective)

    safe_slack = _slack_total(safe)
    labels["feasible"] = bool(safe_slack <= _SLACK_TOL_KM)
    labels["slack_absorbed_km"] = safe_slack
    labels["delta_v_safe_km_s"] = _charged_delta_v_km_s(safe)

    if fuel.solution.ok:
        x_fuel = np.asarray(fuel.x, dtype=float)
        labels["delta_v_fuel_km_s"] = _charged_delta_v_km_s(fuel)
        record: PremiumRecord = safety_premium(
            scenario_id,
            delta_v_fuel_km_s=labels["delta_v_fuel_km_s"],
            delta_v_safe_km_s=labels["delta_v_safe_km_s"],
            induced_fuel_only=_count_induced(context, x_fuel),
            induced_fleet_safe=_count_induced(context, x),
            resolved_fuel_only=_count_resolved(context, x_fuel),
            resolved_fleet_safe=_count_resolved(context, x),
            total_conjunctions=len(context.sensitivities),
            fuel_only_feasible=bool(_slack_total(fuel) <= _SLACK_TOL_KM),
            fleet_safe_feasible=labels["feasible"],
            coupling_number=float(context.graph.coupling_number),
            conflict_dimension=int(context.graph.conflict_dimension),
        )
        labels["premium"] = record.premium
        labels["premium_reason"] = record.reason
        labels["premium_record"] = record.as_dict()
    else:
        labels["premium_reason"] = f"fuel-only solve returned {fuel.solution.status!r}"
        labels["notes"].append(labels["premium_reason"])
    return labels


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@dataclass
class DatasetEntry:
    """One stored sample's manifest row."""

    digest: str
    scenario_id: str
    family: str
    seed: int
    split: str
    key: str
    n_nodes: int
    n_edges: int
    n_resolve: int
    n_latent: int
    n_vars: int
    k_max: int
    n_active: int
    feasible: bool
    cost: float
    premium: float | None
    premium_reason: str
    context_time_s: float
    solve_time_s: float
    bytes: int

    def to_json(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "scenario_id": self.scenario_id,
            "family": self.family,
            "seed": self.seed,
            "split": self.split,
            "key": self.key,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_resolve": self.n_resolve,
            "n_latent": self.n_latent,
            "n_vars": self.n_vars,
            "k_max": self.k_max,
            "n_active": self.n_active,
            "feasible": self.feasible,
            "cost": self.cost,
            "premium": self.premium,
            "premium_reason": self.premium_reason,
            "context_time_s": round(self.context_time_s, 6),
            "solve_time_s": round(self.solve_time_s, 6),
            "bytes": self.bytes,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> DatasetEntry:
        return cls(**{name: payload[name] for name in cls.__dataclass_fields__})


@dataclass
class DatasetIndex:
    """The manifest as an object: entries, failures and fitted statistics."""

    dataset_id: str
    schema_version: str = SCHEMA_VERSION
    split_fractions: tuple[float, float, float] = DEFAULT_SPLIT_FRACTIONS
    lead_pad_orbits: float = DEFAULT_LEAD_PAD_ORBITS
    request_overrides: dict[str, Any] = field(default_factory=dict)
    entries: list[DatasetEntry] = field(default_factory=list)
    failures: dict[str, dict[str, Any]] = field(default_factory=dict)
    feature_stats: FeatureStats | None = None
    created_at: str = ""
    updated_at: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def digests(self) -> set[str]:
        return {entry.digest for entry in self.entries}

    def by_split(self, split: str) -> list[DatasetEntry]:
        if split not in SPLITS:
            raise DatasetError(f"unknown split {split!r}; expected one of {SPLITS}")
        return [entry for entry in self.entries if entry.split == split]

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "schema_version": self.schema_version,
            "entries": len(self.entries),
            "failures": len(self.failures),
            "splits": {split: len(self.by_split(split)) for split in SPLITS},
            "edges": int(sum(entry.n_edges for entry in self.entries)),
            "active_edges": int(sum(entry.n_active for entry in self.entries)),
            "feature_stats_fitted": self.feature_stats is not None,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "schema_version": self.schema_version,
            "node_feature_names": list(NODE_FEATURE_NAMES),
            "edge_feature_names": list(EDGE_FEATURE_NAMES),
            "split_fractions": list(self.split_fractions),
            "lead_pad_orbits": self.lead_pad_orbits,
            "request_overrides": {k: _jsonable(v) for k, v in self.request_overrides.items()},
            "entries": [entry.to_json() for entry in self.entries],
            "failures": self.failures,
            "feature_stats": self.feature_stats.to_json() if self.feature_stats is not None else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "notes": list(self.notes),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> DatasetIndex:
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise FeatureSchemaError(
                f"dataset was written with feature schema {version!r}; this code is {SCHEMA_VERSION!r}"
            )
        if list(payload.get("node_feature_names", [])) != list(NODE_FEATURE_NAMES) or list(
            payload.get("edge_feature_names", [])
        ) != list(EDGE_FEATURE_NAMES):
            raise FeatureSchemaError("dataset feature names differ from this code's feature order")
        stats = payload.get("feature_stats")
        return cls(
            dataset_id=payload["dataset_id"],
            schema_version=version,
            split_fractions=tuple(payload.get("split_fractions", DEFAULT_SPLIT_FRACTIONS)),
            lead_pad_orbits=float(payload.get("lead_pad_orbits", DEFAULT_LEAD_PAD_ORBITS)),
            request_overrides=dict(payload.get("request_overrides", {})),
            entries=[DatasetEntry.from_json(item) for item in payload.get("entries", [])],
            failures=dict(payload.get("failures", {})),
            feature_stats=FeatureStats.from_json(stats) if stats is not None else None,
            created_at=payload.get("created_at", ""),
            updated_at=payload.get("updated_at", ""),
            notes=list(payload.get("notes", [])),
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _open(out_dir_or_store: str | Path | ArtifactStore) -> ArtifactStore:
    if isinstance(out_dir_or_store, (str, Path)):
        return LocalArtifactStore(Path(out_dir_or_store))
    return out_dir_or_store


def _sample_key(digest: str) -> str:
    return f"{_SAMPLE_PREFIX}/{digest}.pt"


def _encode_sample(sample: GraphSample) -> bytes:
    buffer = io.BytesIO()
    torch.save(sample.to_dict(), buffer)
    return buffer.getvalue()


def _decode_sample(data: bytes) -> GraphSample:
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    return GraphSample.from_dict(payload)


def _read_index(store: ArtifactStore, dataset_id: str) -> DatasetIndex | None:
    if not store.exists(_MANIFEST_KEY):
        return None
    index = DatasetIndex.from_json(json.loads(store.get(_MANIFEST_KEY).decode("utf-8")))
    if index.dataset_id != dataset_id:
        raise DatasetError(
            f"store already holds dataset {index.dataset_id!r}; refusing to mix in {dataset_id!r}"
        )
    return index


def _write_index(store: ArtifactStore, index: DatasetIndex) -> None:
    index.updated_at = format_epoch(utc_now())
    store.put(
        _MANIFEST_KEY,
        json.dumps(index.to_json(), sort_keys=True, indent=1).encode("utf-8"),
        content_type="application/json",
    )


def generate_dataset(
    families: Iterable[str],
    seeds: Iterable[int],
    *,
    authorization: SyntheticAuthorization | None,
    out_dir_or_store: str | Path | ArtifactStore,
    dataset_id: str = "conjunction-graphs",
    split_fractions: Sequence[float] = DEFAULT_SPLIT_FRACTIONS,
    lead_pad_orbits: float = DEFAULT_LEAD_PAD_ORBITS,
    screening_step_s: float | None = None,
    request_overrides: dict[str, Any] | None = None,
    scenario_overrides: dict[str, Any] | None = None,
    scp_iterations: int | None = None,
    dtype: torch.dtype = torch.float32,
    fit_stats: bool = True,
    progress: Callable[[str], None] | None = None,
) -> DatasetIndex:
    """Generate, solve, tensorise and store every ``(family, seed)`` pair.

    Resumable: a digest already in the manifest is skipped, so an interrupted
    run picks up where it stopped and a re-run with more seeds only adds. A
    scenario that fails at any stage is recorded under ``failures`` with its
    traceback and the sweep continues; a failed digest is retried on the
    next run because it is not an entry. Normalisation statistics are fitted
    on the training split at the end of every run and stored in the manifest
    -- samples themselves stay raw.
    """
    store = _open(out_dir_or_store)
    fractions = tuple(float(f) for f in split_fractions)
    if len(fractions) != 3:
        raise DatasetError("split_fractions must have exactly three entries")
    index = _read_index(store, dataset_id)
    if index is None:
        index = DatasetIndex(
            dataset_id=dataset_id,
            split_fractions=fractions,  # type: ignore[arg-type]
            lead_pad_orbits=float(lead_pad_orbits),
            request_overrides=dict(request_overrides or {}),
            created_at=format_epoch(utc_now()),
        )
    elif tuple(index.split_fractions) != fractions:
        raise DatasetError(
            f"existing dataset was split {tuple(index.split_fractions)}; refusing to resume with {fractions}"
        )
    known = index.digests
    say = progress or (lambda message: None)

    for family in families:
        for seed in seeds:
            started = time.perf_counter()
            digest = ""
            try:
                scenario = generate(family, int(seed), authorization=authorization, **(scenario_overrides or {}))
                digest = scenario_digest(scenario)
                if digest in known:
                    say(f"skip {family} seed {seed}: {digest[:12]} already stored")
                    continue
                context = prepare_context(
                    scenario,
                    lead_pad_orbits=lead_pad_orbits,
                    screening_step_s=screening_step_s,
                    request_overrides=request_overrides,
                )
                context_time = time.perf_counter() - started
                labels = build_labels(
                    context, max_iterations=scp_iterations, scenario_id=scenario.scenario_id
                )
                sample = tensorize(scenario, context, labels, dtype=dtype)
                data = _encode_sample(sample)
                key = _sample_key(digest)
                store.put(key, data, content_type="application/octet-stream")
                entry = DatasetEntry(
                    digest=digest,
                    scenario_id=scenario.scenario_id,
                    family=scenario.family,
                    seed=int(scenario.seed),
                    split=split_for_digest(digest, fractions),
                    key=key,
                    n_nodes=sample.n_nodes,
                    n_edges=sample.n_edges,
                    n_resolve=int(sample.meta["n_resolve"]),
                    n_latent=int(sample.meta["n_latent"]),
                    n_vars=sample.n_vars,
                    k_max=sample.k_max,
                    n_active=int(labels["n_active"]),
                    feasible=bool(labels["feasible"]),
                    cost=float(labels["cost"]),
                    premium=labels["premium"],
                    premium_reason=str(labels["premium_reason"]),
                    context_time_s=context_time,
                    solve_time_s=float(labels["solve_time_s"]),
                    bytes=len(data),
                )
                index.entries.append(entry)
                index.failures.pop(digest, None)
                known.add(digest)
                say(
                    f"{family} seed {seed} [{entry.split}]: {entry.n_nodes} nodes, {entry.n_edges} edges, "
                    f"{entry.n_active} active, {time.perf_counter() - started:.1f}s"
                )
            except Exception as error:  # noqa: BLE001 - recorded per scenario, never swallowed
                failure_key = digest or f"{family}:{seed}"
                index.failures[failure_key] = {
                    "family": family,
                    "seed": int(seed),
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(limit=8),
                    "at": format_epoch(utc_now()),
                }
                say(f"FAIL {family} seed {seed}: {type(error).__name__}: {error}")
            _write_index(store, index)

    if fit_stats:
        train = [
            _decode_sample(store.get(entry.key)) for entry in index.by_split("train")
        ]
        if train:
            index.feature_stats = fit_normalisation(train)
        else:
            index.notes.append("no training samples; normalisation statistics not fitted")
    _write_index(store, index)
    return index


def load_manifest(path_or_store: str | Path | ArtifactStore) -> DatasetIndex:
    store = _open(path_or_store)
    if not store.exists(_MANIFEST_KEY):
        raise DatasetError("no manifest.json in the dataset store")
    return DatasetIndex.from_json(json.loads(store.get(_MANIFEST_KEY).decode("utf-8")))


def load_dataset(
    path_or_store: str | Path | ArtifactStore,
    *,
    split: str | None = None,
    normalise: bool = False,
    stats: FeatureStats | None = None,
) -> list[GraphSample]:
    """Read samples back, optionally one split, optionally z-scored.

    ``normalise=True`` uses the statistics stored in the manifest -- the
    contract's "never recomputed at inference" -- unless ``stats`` is given
    explicitly, which is how a model trained on one dataset is evaluated on
    another with *its own* training statistics.
    """
    store = _open(path_or_store)
    index = load_manifest(store)
    entries = index.entries if split is None else index.by_split(split)
    chosen = stats if stats is not None else index.feature_stats
    if normalise and chosen is None:
        raise DatasetError("normalise=True but the dataset carries no feature statistics")
    samples: list[GraphSample] = []
    for entry in entries:
        sample = _decode_sample(store.get(entry.key))
        if sample.digest != entry.digest:
            raise DatasetError(f"sample at {entry.key} carries digest {sample.digest!r}, manifest says {entry.digest!r}")
        sample.meta["split"] = entry.split
        if normalise:
            sample = apply_normalisation(sample, chosen)  # type: ignore[arg-type]
        samples.append(sample)
    return samples


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


@dataclass
class GraphBatch:
    """Several graphs concatenated, with the offsets needed to tell them apart.

    Node and edge tensors are stacked along their first axis; ``edge_index``
    is shifted by each graph's node offset so it indexes the stacked node
    table directly. Per-graph quantities that must keep their own decision
    vector -- ``sensitivities`` and ``node_columns`` -- are padded to the
    widest ``n_vars`` in the batch rather than offset, so a ``(G, n_vars_max)``
    prediction feeds :func:`aegis.ml.torchphysics.constraint_residuals_torch`
    with ``edge_batch`` selecting each edge's row of it.
    """

    node_features: Tensor
    edge_index: Tensor
    edge_features: Tensor
    edge_kind: Tensor
    node_batch: Tensor
    edge_batch: Tensor
    node_offsets: Tensor
    edge_offsets: Tensor
    n_graphs: int
    slot_mask: Tensor
    node_columns: Tensor
    dv_mask: Tensor
    miss_vectors: Tensor
    sensitivities: Tensor
    directions: Tensor
    floors: Tensor
    n_vars: Tensor
    var_mask: Tensor
    node_ids: list[str]
    edge_labels: list[str]
    digests: list[str]
    metas: list[dict[str, Any]]
    edge_active: Tensor | None = None
    edge_dual: Tensor | None = None
    node_dv: Tensor | None = None
    global_feasible: Tensor | None = None
    global_cost: Tensor | None = None
    global_premium: Tensor | None = None
    global_premium_mask: Tensor | None = None

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
    def n_vars_max(self) -> int:
        return int(self.sensitivities.shape[2])

    def to(self, device: torch.device | str) -> GraphBatch:
        moved = {}
        for name, value in self.__dict__.items():
            moved[name] = value.to(device) if isinstance(value, Tensor) else value
        return GraphBatch(**moved)


def _pad_last(tensor: Tensor, width: int, fill: float | int | bool) -> Tensor:
    current = int(tensor.shape[-1])
    if current == width:
        return tensor
    if current > width:
        raise DatasetError("cannot pad a tensor down")
    padding = tensor.new_full(tensor.shape[:-1] + (width - current,), fill)
    return torch.cat([tensor, padding], dim=-1)


def _optional_cat(samples: list[GraphSample], name: str, pad_width: int | None) -> Tensor | None:
    present = [getattr(sample, name) is not None for sample in samples]
    if not any(present):
        return None
    if not all(present):
        raise DatasetError(f"{name} is present on some samples and missing on others; refusing to batch")
    pieces = [getattr(sample, name) for sample in samples]
    if pad_width is not None:
        pieces = [_pad_last(piece, pad_width, 0.0) for piece in pieces]
    return torch.cat(pieces, dim=0)


def collate(samples: Sequence[GraphSample]) -> GraphBatch:
    """Concatenate variable-size graphs with node offsets -- the standard trick.

    Refuses to mix normalised and raw samples, or labelled and unlabelled
    ones, because either mix would train on inconsistent inputs without any
    error surfacing later.
    """
    samples = list(samples)
    if not samples:
        raise DatasetError("cannot collate zero samples")
    normalised = {bool(sample.meta.get("normalised", False)) for sample in samples}
    if len(normalised) != 1:
        raise DatasetError("cannot batch normalised samples together with raw ones")
    for sample in samples:
        sample.validate()

    k_max = max(sample.k_max for sample in samples)
    n_vars_max = max(sample.n_vars for sample in samples)
    node_counts = torch.tensor([sample.n_nodes for sample in samples], dtype=torch.long)
    edge_counts = torch.tensor([sample.n_edges for sample in samples], dtype=torch.long)
    node_offsets = torch.cat([torch.zeros(1, dtype=torch.long), node_counts.cumsum(0)[:-1]])
    edge_offsets = torch.cat([torch.zeros(1, dtype=torch.long), edge_counts.cumsum(0)[:-1]])

    edge_index = torch.cat(
        [sample.edge_index + int(offset) for sample, offset in zip(samples, node_offsets)], dim=1
    )
    node_batch = torch.repeat_interleave(torch.arange(len(samples)), node_counts)
    edge_batch = torch.repeat_interleave(torch.arange(len(samples)), edge_counts)

    sensitivities = torch.cat(
        [_pad_last(sample.sensitivities, n_vars_max, 0.0) for sample in samples], dim=0
    )
    n_vars = torch.tensor([sample.n_vars for sample in samples], dtype=torch.long)
    var_mask = torch.arange(n_vars_max)[None, :] < n_vars[:, None]

    global_premium = None
    global_premium_mask = None
    if any(sample.global_premium is not None for sample in samples):
        dtype = samples[0].node_features.dtype
        global_premium = torch.zeros(len(samples), dtype=dtype)
        global_premium_mask = torch.zeros(len(samples), dtype=torch.bool)
        for row, sample in enumerate(samples):
            if sample.global_premium is not None:
                global_premium[row] = sample.global_premium.to(dtype)
                global_premium_mask[row] = True

    def stack_scalar(name: str) -> Tensor | None:
        present = [getattr(sample, name) is not None for sample in samples]
        if not any(present):
            return None
        if not all(present):
            raise DatasetError(f"{name} is present on some samples and missing on others")
        return torch.stack([getattr(sample, name).reshape(()) for sample in samples])

    return GraphBatch(
        node_features=torch.cat([sample.node_features for sample in samples], dim=0),
        edge_index=edge_index,
        edge_features=torch.cat([sample.edge_features for sample in samples], dim=0),
        edge_kind=torch.cat([sample.edge_kind for sample in samples], dim=0),
        node_batch=node_batch,
        edge_batch=edge_batch,
        node_offsets=node_offsets,
        edge_offsets=edge_offsets,
        n_graphs=len(samples),
        slot_mask=torch.cat([_pad_last(sample.slot_mask, k_max, False) for sample in samples], dim=0),
        node_columns=torch.cat([_pad_last(sample.node_columns, 3 * k_max, -1) for sample in samples], dim=0),
        dv_mask=torch.cat([_pad_last(sample.dv_mask, 3 * k_max, False) for sample in samples], dim=0),
        miss_vectors=torch.cat([sample.miss_vectors for sample in samples], dim=0),
        sensitivities=sensitivities,
        directions=torch.cat([sample.directions for sample in samples], dim=0),
        floors=torch.cat([sample.floors for sample in samples], dim=0),
        n_vars=n_vars,
        var_mask=var_mask,
        node_ids=[node_id for sample in samples for node_id in sample.node_ids],
        edge_labels=[label for sample in samples for label in sample.edge_labels],
        digests=[sample.digest for sample in samples],
        metas=[dict(sample.meta) for sample in samples],
        edge_active=_optional_cat(samples, "edge_active", None),
        edge_dual=_optional_cat(samples, "edge_dual", None),
        node_dv=_optional_cat(samples, "node_dv", 3 * k_max),
        global_feasible=stack_scalar("global_feasible"),
        global_cost=stack_scalar("global_cost"),
        global_premium=global_premium,
        global_premium_mask=global_premium_mask,
    )
