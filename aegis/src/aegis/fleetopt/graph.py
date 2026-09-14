"""The conjunction graph and the structural numbers that predict difficulty.

The research question this supports is not "how much delta-v does a plan
cost" but "what *kind* of conjunction network makes a safe plan expensive".
Armellin (2021) measured delta-v against a tightening probability threshold
for a single geometry and saw it rise from 6.4 to 431.1 mm/s. Klinkrad et al.
(ESA SP-587, 2005) measured fleet-level maneuver *frequency* against
orbit-determination accuracy. Neither stratified by the topology of the
simultaneous-conjunction network, which is the axis this module exposes.

Graph view
----------
Nodes are objects; edges are conjunctions. Degree is how many simultaneous
conjunctions one satellite is in; a connected component is a set of
conjunctions that must be planned together because they share spacecraft; and
TCA overlap says whether the events are genuinely simultaneous or merely
sequential.

Beyond the combinatorics
------------------------
Topology alone is not enough, because two star graphs can be easy or
impossible depending on whether the required displacements point in
compatible directions. Two spectral quantities capture that.

**Coupling number.** Stack the row-normalised projected resolve sensitivities
into :math:`B` and take the condition number of :math:`B B^{\\top}`. A
well-conditioned :math:`B B^{\\top}` means the conjunctions ask for nearly
independent displacements and can be satisfied cheaply in parallel; an
ill-conditioned one means some conjunctions are nearly collinear in control
space, so satisfying one nearly determines the others -- and if they ask for
*opposite* things, satisfying both is expensive or impossible. For the
Euclidean relaxation the minimum-norm solution of :math:`Bx = \\rho` costs
:math:`\\sqrt{\\rho^{\\top}(BB^{\\top})^{-1}\\rho}`, which makes the dependence
on the spectrum explicit: the cost blows up exactly along the
small-eigenvalue directions the condition number measures.

**Conflict dimension.** ``rank([B_resolve; B_latent]) - rank(B_resolve)``:
how many additional independent directions in control space the
induced-conjunction rows pin down. Zero means the latent rows live in the span
the resolve rows already constrain, so they are nearly free; a large value
means the no-induce requirement is asking the fleet to hold still in
directions the avoidance requirement never touched, which is where the safety
premium comes from.

These two numbers are the independent variables of the stratified premium
study in :mod:`aegis.experiments`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..core.conjunction import RiskLevel
from ..core.objects import SpaceObject
from ..core.timebase import seconds_between
from ..risk.batch import AssessedCatalog
from .bplane import MissSensitivity
from .latent import LatentConstraint

__all__ = [
    "ConjunctionGraph",
    "GraphMetrics",
    "build_conjunction_graph",
    "graph_metrics",
    "coupling_number",
    "conflict_dimension",
    "RANK_DEFICIENT_SENTINEL",
]

#: Reported in place of an infinite condition number, so every metric is a
#: finite float and downstream statistics never have to special-case inf.
RANK_DEFICIENT_SENTINEL = 1e18

_RANK_TOL = 1e-10


@dataclass
class ConjunctionGraph:
    """Undirected multigraph of objects joined by conjunctions.

    Implemented with plain dictionaries rather than a graph library: the
    operations needed here are degree, connected components, and a few sums,
    and the project already declines dependencies it does not need.
    """

    nodes: dict[str, SpaceObject] = field(default_factory=dict)
    edges: list[tuple[str, str, str]] = field(default_factory=list)
    edge_attributes: dict[str, dict] = field(default_factory=dict)
    adjacency: dict[str, set[str]] = field(default_factory=dict)

    def add_node(self, obj: SpaceObject) -> None:
        self.nodes.setdefault(obj.object_id, obj)
        self.adjacency.setdefault(obj.object_id, set())

    def add_edge(self, object_a: str, object_b: str, conjunction_id: str, **attributes) -> None:
        self.edges.append((object_a, object_b, conjunction_id))
        self.edge_attributes[conjunction_id] = dict(attributes)
        self.adjacency.setdefault(object_a, set()).add(object_b)
        self.adjacency.setdefault(object_b, set()).add(object_a)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def degree(self, object_id: str) -> int:
        """Number of conjunctions this object is in, counting repeats."""
        return sum(1 for a, b, _ in self.edges if object_id in (a, b))

    def degrees(self) -> dict[str, int]:
        counts = {object_id: 0 for object_id in self.nodes}
        for a, b, _ in self.edges:
            counts[a] = counts.get(a, 0) + 1
            counts[b] = counts.get(b, 0) + 1
        return counts

    def components(self) -> list[set[str]]:
        """Connected components over nodes that have at least one edge."""
        involved = {node for a, b, _ in self.edges for node in (a, b)}
        seen: set[str] = set()
        found: list[set[str]] = []
        for start in sorted(involved):
            if start in seen:
                continue
            stack = [start]
            component: set[str] = set()
            while stack:
                node = stack.pop()
                if node in component:
                    continue
                component.add(node)
                stack.extend(self.adjacency.get(node, set()) - component)
            seen |= component
            found.append(component)
        return found

    def maneuverable_nodes(self) -> list[str]:
        return sorted(
            object_id for object_id, obj in self.nodes.items() if obj.is_maneuverable
        )


def build_conjunction_graph(
    assessed: AssessedCatalog,
    objects: list[SpaceObject] | None = None,
) -> ConjunctionGraph:
    """Graph of every assessed conjunction, with its risk attributes on the edge."""
    graph = ConjunctionGraph()
    for obj in objects or []:
        graph.add_node(obj)
    for entry in assessed.entries:
        conjunction = entry.conjunction
        graph.add_node(conjunction.primary)
        graph.add_node(conjunction.secondary)
        graph.add_edge(
            conjunction.primary.object_id,
            conjunction.secondary.object_id,
            conjunction.conjunction_id,
            tca=conjunction.tca,
            miss_distance_km=float(conjunction.miss_distance_km),
            relative_speed_km_s=float(conjunction.relative_speed_km_s),
            probability=float(entry.assessment.probability),
            risk_level=entry.assessment.risk_level,
            intra_fleet=bool(conjunction.is_intra_fleet),
            maneuverable_ids=tuple(conjunction.maneuverable_object_ids),
            short_encounter_valid=bool(entry.assessment.short_encounter_valid),
        )
    return graph


def _normalized_stack(sensitivities: list[MissSensitivity]) -> np.ndarray:
    """Row-normalised scalar sensitivity rows, one per conjunction.

    Each conjunction contributes the row ``u_j^T P_j B_j`` evaluated at its
    nominal direction -- the same row the linear program sees. Normalising
    removes the arbitrary scale of the required-miss units so the condition
    number reflects geometry rather than bookkeeping.
    """
    rows = []
    for sensitivity in sensitivities:
        row = sensitivity.nominal_direction @ sensitivity.sensitivity
        norm = float(np.linalg.norm(row))
        if norm > _RANK_TOL:
            rows.append(row / norm)
    if not rows:
        return np.zeros((0, 0))
    return np.asarray(rows, dtype=float)


def coupling_number(sensitivities: list[MissSensitivity]) -> float:
    """``cond(B B^T)`` of the row-normalised resolve sensitivities.

    Returns 1.0 for a single controllable conjunction (trivially
    well-conditioned), 0.0 when nothing is controllable, and
    :data:`RANK_DEFICIENT_SENTINEL` when ``B B^T`` is singular -- which means
    two or more conjunctions demand displacements that are linearly dependent
    in control space, the hardest structural case.
    """
    stack = _normalized_stack(sensitivities)
    if stack.size == 0:
        return 0.0
    if stack.shape[0] == 1:
        return 1.0
    gram = stack @ stack.T
    eigenvalues = np.linalg.eigvalsh(gram)
    largest = float(np.max(eigenvalues))
    smallest = float(np.min(eigenvalues))
    if largest <= _RANK_TOL:
        return 0.0
    if smallest <= _RANK_TOL * largest:
        return RANK_DEFICIENT_SENTINEL
    return float(min(largest / smallest, RANK_DEFICIENT_SENTINEL))


def conflict_dimension(
    sensitivities: list[MissSensitivity],
    latent: list[LatentConstraint],
) -> int:
    """Extra independent control directions the latent rows pin down."""
    resolve_stack = _normalized_stack(sensitivities)
    latent_rows = []
    for constraint in latent:
        row = constraint.direction @ constraint.sensitivity
        norm = float(np.linalg.norm(row))
        if norm > _RANK_TOL:
            latent_rows.append(row / norm)
    if not latent_rows:
        return 0
    latent_stack = np.asarray(latent_rows, dtype=float)
    if resolve_stack.size == 0:
        return int(np.linalg.matrix_rank(latent_stack, tol=_RANK_TOL))
    combined = np.vstack([resolve_stack, latent_stack])
    return int(
        np.linalg.matrix_rank(combined, tol=_RANK_TOL)
        - np.linalg.matrix_rank(resolve_stack, tol=_RANK_TOL)
    )


@dataclass
class GraphMetrics:
    """Structural descriptors of one planning scenario."""

    nodes: int = 0
    edges: int = 0
    active_edges: int = 0
    maneuverable_nodes: int = 0
    intra_fleet_edges: int = 0
    component_count: int = 0
    largest_component: int = 0
    max_degree: int = 0
    mean_degree: float = 0.0
    tca_span_s: float = 0.0
    tca_overlap_fraction: float = 0.0
    intra_fleet_fraction: float = 0.0
    low_relative_velocity_edges: int = 0
    coupling_number: float = 0.0
    conflict_dimension: int = 0
    controllable_edges: int = 0
    median_miss_km: float = 0.0
    min_miss_km: float = 0.0
    max_probability: float = 0.0

    def as_dict(self) -> dict:
        values = {
            "nodes": self.nodes,
            "edges": self.edges,
            "active_edges": self.active_edges,
            "maneuverable_nodes": self.maneuverable_nodes,
            "intra_fleet_edges": self.intra_fleet_edges,
            "component_count": self.component_count,
            "largest_component": self.largest_component,
            "max_degree": self.max_degree,
            "mean_degree": round(self.mean_degree, 6),
            "tca_span_s": round(self.tca_span_s, 3),
            "tca_overlap_fraction": round(self.tca_overlap_fraction, 6),
            "intra_fleet_fraction": round(self.intra_fleet_fraction, 6),
            "low_relative_velocity_edges": self.low_relative_velocity_edges,
            "coupling_number": round(self.coupling_number, 6),
            "conflict_dimension": self.conflict_dimension,
            "controllable_edges": self.controllable_edges,
            "median_miss_km": round(self.median_miss_km, 6),
            "min_miss_km": round(self.min_miss_km, 6),
            "max_probability": self.max_probability,
        }
        for key, value in values.items():
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError(f"graph metric {key} is not finite: {value}")
        return values


def graph_metrics(
    graph: ConjunctionGraph,
    *,
    sensitivities: list[MissSensitivity] | None = None,
    latent: list[LatentConstraint] | None = None,
    active_level: str = RiskLevel.MONITOR,
    overlap_window_s: float | None = None,
) -> GraphMetrics:
    """Every structural number, computed once.

    ``overlap_window_s`` defaults to one orbital period at 550 km (5739 s),
    which is the natural scale for "simultaneous": two conjunctions within one
    orbit of each other involve the same satellite at nearly the same point in
    its control authority, so their burn slots overlap and they genuinely
    couple.
    """
    sensitivities = list(sensitivities or [])
    latent = list(latent or [])
    window_s = 5739.0 if overlap_window_s is None else float(overlap_window_s)

    degrees = graph.degrees()
    involved_degrees = [value for value in degrees.values() if value > 0]
    components = graph.components()
    attributes = [graph.edge_attributes[cid] for _, _, cid in graph.edges]

    tcas: list[datetime] = [attrs["tca"] for attrs in attributes if attrs.get("tca")]
    span_s = 0.0
    overlap_fraction = 0.0
    if len(tcas) >= 2:
        ordered = sorted(tcas)
        span_s = seconds_between(ordered[0], ordered[-1])
        pairs = 0
        overlapping = 0
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                pairs += 1
                if abs(seconds_between(ordered[i], ordered[j])) <= window_s:
                    overlapping += 1
        overlap_fraction = overlapping / pairs if pairs else 0.0
    elif len(tcas) == 1:
        overlap_fraction = 1.0

    misses = [attrs["miss_distance_km"] for attrs in attributes]
    threshold = RiskLevel.rank(active_level)

    return GraphMetrics(
        nodes=graph.node_count,
        edges=graph.edge_count,
        active_edges=sum(
            1 for attrs in attributes if RiskLevel.rank(attrs["risk_level"]) >= threshold
        ),
        maneuverable_nodes=len(graph.maneuverable_nodes()),
        intra_fleet_edges=sum(1 for attrs in attributes if attrs["intra_fleet"]),
        component_count=len(components),
        largest_component=max((len(c) for c in components), default=0),
        max_degree=max(involved_degrees, default=0),
        mean_degree=float(np.mean(involved_degrees)) if involved_degrees else 0.0,
        tca_span_s=span_s,
        tca_overlap_fraction=overlap_fraction,
        intra_fleet_fraction=(
            sum(1 for attrs in attributes if attrs["intra_fleet"]) / len(attributes)
            if attributes
            else 0.0
        ),
        low_relative_velocity_edges=sum(
            1 for attrs in attributes if not attrs.get("short_encounter_valid", True)
        ),
        coupling_number=coupling_number(sensitivities),
        conflict_dimension=conflict_dimension(sensitivities, latent),
        controllable_edges=sum(1 for s in sensitivities if s.is_controllable),
        median_miss_km=float(np.median(misses)) if misses else 0.0,
        min_miss_km=float(np.min(misses)) if misses else 0.0,
        max_probability=max((attrs["probability"] for attrs in attributes), default=0.0),
    )
