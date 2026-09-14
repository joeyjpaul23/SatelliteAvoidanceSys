"""Cost models: how a three-axis impulse becomes a linear objective.

The optimizer wants to minimise total fleet delta-v, which is a sum of
Euclidean norms :math:`\\sum_{i,k} \\lVert \\Delta v_{i,k}\\rVert_2`. That is a
second-order cone objective, and there is no SOCP solver in this project's
dependency set -- scipy gives us HiGHS, which is simplex and interior-point
for linear and mixed-integer linear programs only. Rather than add a
dependency, this module offers two linear cost models with stated
approximation quality, so the reported delta-v always comes with a known
relationship to the true Euclidean cost.

**L1 model** (default). Split each component into non-negative plus and minus
parts and charge ``w_axis * (plus + minus)``. Since
:math:`\\lVert v \\rVert_2 \\le \\lVert v \\rVert_1 \\le \\sqrt{3}\\lVert v\\rVert_2`,
the L1 objective never *under*-charges a burn: an optimizer minimising it
cannot cheat the fuel budget. The cost is that a burn needing all three axes
is charged up to 73 % too much, so the optimizer is biased towards axis-aligned
burns. For collision avoidance this bias is desirable rather than harmful --
along-track burns are the ones that work (see
:mod:`aegis.fleetopt.dynamics`) -- and it is made explicit through
``axis_weights``.

``axis_weights`` defaults to ``(3.0, 1.0, 5.0)``: radial three times and
cross-track five times the cost of along-track. This is an operational
preference, not a physical one. Radial impulses buy only bounded periodic
displacement, so spending them on separation is wasteful; cross-track impulses
change inclination and RAAN, which walks the satellite out of its
constellation slot. Both reasons are already written down in
:attr:`aegis.core.maneuver.Maneuver.is_along_track_only`. Set the weights to
``(1, 1, 1)`` to recover a pure L1 objective.

**Polyhedral cone model.** Introduce one magnitude variable :math:`t_{i,k}`
per burn slot and require :math:`g_\\ell^{\\top} \\Delta v_{i,k} \\le t_{i,k}`
for a set of unit directions :math:`\\{g_\\ell\\}` covering the sphere. Then
:math:`t^\\star_{i,k} = \\max_\\ell g_\\ell^{\\top}\\Delta v_{i,k}`, which
satisfies

.. math::

    \\cos(\\theta) \\lVert v \\rVert_2 \\le \\max_\\ell g_\\ell^{\\top} v
    \\le \\lVert v \\rVert_2 ,

where :math:`\\theta` is the covering radius of the direction set. This model
*under*-charges by at most :math:`1-\\cos\\theta`, so it is the one to use when
an honest delta-v number matters more than a conservative one. With a
once-subdivided icosahedron (42 directions) the covering radius is about
21 deg, giving at most 7 % under-charge; twice-subdivided (162 directions)
gives about 11 deg and 2 %.

Both models are reported side by side in the benchmark so the reader can see
the bracket rather than trusting one number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dynamics import BurnGrid
from .errors import AssemblyError

__all__ = [
    "CostModel",
    "l1_cost_model",
    "polyhedral_cone",
    "cone_cost_model",
    "covering_radius_deg",
    "DEFAULT_AXIS_WEIGHTS",
]

#: Radial, transverse, normal cost multipliers. See the module docstring.
DEFAULT_AXIS_WEIGHTS = (3.0, 1.0, 5.0)

_GOLDEN = (1.0 + 5.0**0.5) / 2.0


@dataclass
class CostModel:
    """A lifting of the signed decision vector into a non-negative LP.

    Attributes
    ----------
    lift
        ``(n_vars, n_lifted)`` matrix with ``x = lift @ z`` where ``z >= 0``.
        For the L1 model this is the plus/minus split; magnitude variables
        added by the cone model contribute zero columns.
    cost
        ``(n_lifted,)`` objective coefficients over ``z``.
    extra_rows
        ``(m, n_lifted)`` additional inequality rows ``extra_rows @ z <= extra_rhs``.
    labels
        One label per lifted column, for dual and solution reporting.
    row_labels
        One label per row of ``extra_rows``.
    magnitude_columns
        For each ``(satellite, slot)`` in grid order, the lifted columns whose
        sum is that burn's charged magnitude. Used to report per-burn delta-v
        consistently across both models.
    """

    name: str
    lift: np.ndarray
    cost: np.ndarray
    labels: list[str]
    extra_rows: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    extra_rhs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    row_labels: list[str] = field(default_factory=list)
    magnitude_columns: dict[tuple[str, int], list[int]] = field(default_factory=dict)
    approximation_note: str = ""

    def __post_init__(self) -> None:
        self.lift = np.asarray(self.lift, dtype=float)
        self.cost = np.asarray(self.cost, dtype=float)
        if self.lift.ndim != 2:
            raise AssemblyError("lift must be a 2-D matrix")
        if self.cost.shape != (self.lift.shape[1],):
            raise AssemblyError("cost must have one entry per lifted column")
        if len(self.labels) != self.lift.shape[1]:
            raise AssemblyError("labels must have one entry per lifted column")
        if self.extra_rows.size:
            self.extra_rows = np.asarray(self.extra_rows, dtype=float)
            if self.extra_rows.shape[1] != self.lift.shape[1]:
                raise AssemblyError("extra_rows width must match the lifted dimension")
            if self.extra_rhs.shape != (self.extra_rows.shape[0],):
                raise AssemblyError("extra_rhs must have one entry per extra row")

    @property
    def n_vars(self) -> int:
        return int(self.lift.shape[0])

    @property
    def n_lifted(self) -> int:
        return int(self.lift.shape[1])

    def recover(self, z: np.ndarray) -> np.ndarray:
        """Signed decision vector from the lifted solution."""
        return self.lift @ np.asarray(z, dtype=float).reshape(self.n_lifted)

    def charged_magnitude(self, z: np.ndarray, satellite_id: str, slot: int) -> float:
        """The magnitude this model charged for one burn."""
        columns = self.magnitude_columns.get((satellite_id, slot))
        if not columns:
            return 0.0
        z = np.asarray(z, dtype=float)
        return float(np.sum(z[columns]))


def l1_cost_model(
    grid: BurnGrid,
    *,
    axis_weights: tuple[float, float, float] = DEFAULT_AXIS_WEIGHTS,
) -> CostModel:
    """Plus/minus split with per-axis weights."""
    weights = tuple(float(w) for w in axis_weights)
    if len(weights) != 3 or any(w <= 0.0 for w in weights):
        raise AssemblyError("axis_weights must be three positive numbers")

    n_vars = grid.n_vars
    lift = np.zeros((n_vars, 2 * n_vars), dtype=float)
    cost = np.zeros(2 * n_vars, dtype=float)
    labels: list[str] = []
    magnitude_columns: dict[tuple[str, int], list[int]] = {}

    column_labels = grid.column_labels()
    axis_index = {"R": 0, "T": 1, "N": 2}
    for var in range(n_vars):
        lift[var, 2 * var] = 1.0
        lift[var, 2 * var + 1] = -1.0
        _, sat_id, slot_text, axis = column_labels[var].split(":")
        weight = weights[axis_index[axis]]
        cost[2 * var] = weight
        cost[2 * var + 1] = weight
        labels.append(f"{column_labels[var]}:+")
        labels.append(f"{column_labels[var]}:-")
        key = (sat_id, int(slot_text))
        magnitude_columns.setdefault(key, []).extend([2 * var, 2 * var + 1])

    return CostModel(
        name="l1",
        lift=lift,
        cost=cost,
        labels=labels,
        magnitude_columns=magnitude_columns,
        approximation_note=(
            "L1 never under-charges: ||v||_2 <= ||v||_1 <= sqrt(3)||v||_2. "
            f"Axis weights (R,T,N) = {weights}."
        ),
    )


def polyhedral_cone(order: int = 1) -> np.ndarray:
    """Near-uniform unit directions from a subdivided icosahedron.

    ``order=0`` returns the 12 vertices, ``order=1`` 42, ``order=2`` 162,
    ``order=3`` 642. Geodesic subdivision is used rather than a random or
    Fibonacci point set because the covering radius of a subdivided
    icosahedron is both small and *predictable*, and the approximation bound
    in the module docstring depends on knowing it.
    """
    if order < 0:
        raise AssemblyError("order must be non-negative")

    vertices = []
    for sign_a in (1.0, -1.0):
        for sign_b in (1.0, -1.0):
            vertices.extend(
                [
                    [0.0, sign_a * 1.0, sign_b * _GOLDEN],
                    [sign_a * 1.0, sign_b * _GOLDEN, 0.0],
                    [sign_b * _GOLDEN, 0.0, sign_a * 1.0],
                ]
            )
    points = np.unique(np.asarray(vertices, dtype=float), axis=0)
    points /= np.linalg.norm(points, axis=1, keepdims=True)

    faces = _icosahedron_faces(points)
    for _ in range(order):
        points, faces = _subdivide(points, faces)

    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    return _dedupe_directions(points)


def _icosahedron_faces(points: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangles of the icosahedron, found from mutual nearest neighbours.

    Deriving the faces from geometry rather than hard-coding an index table
    keeps this correct if the vertex ordering from ``np.unique`` changes.
    """
    count = points.shape[0]
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    edge_length = np.min(distances[distances > 1e-9])
    adjacency = [
        {j for j in range(count) if abs(distances[i, j] - edge_length) < 1e-6}
        for i in range(count)
    ]
    faces: set[tuple[int, int, int]] = set()
    for i in range(count):
        for j in adjacency[i]:
            for k in adjacency[i] & adjacency[j]:
                faces.add(tuple(sorted((i, j, k))))
    return sorted(faces)


def _subdivide(
    points: np.ndarray, faces: list[tuple[int, int, int]]
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    coordinates = [row for row in points]
    midpoints: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        existing = midpoints.get(key)
        if existing is not None:
            return existing
        vector = coordinates[a] + coordinates[b]
        vector = vector / np.linalg.norm(vector)
        coordinates.append(vector)
        index = len(coordinates) - 1
        midpoints[key] = index
        return index

    new_faces: list[tuple[int, int, int]] = []
    for a, b, c in faces:
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        new_faces.extend(
            [
                (a, ab, ca),
                (b, bc, ab),
                (c, ca, bc),
                (ab, bc, ca),
            ]
        )
    return np.asarray(coordinates, dtype=float), new_faces


def _dedupe_directions(points: np.ndarray) -> np.ndarray:
    rounded = np.round(points, 9)
    _, index = np.unique(rounded, axis=0, return_index=True)
    return points[np.sort(index)]


def covering_radius_deg(directions: np.ndarray, *, samples: int = 20000, seed: int = 0) -> float:
    """Monte-Carlo estimate of ``max_v min_l angle(v, g_l)`` in degrees.

    Reported with the cost model so the under-charge bound
    ``1 - cos(covering_radius)`` is always available next to the delta-v
    numbers it qualifies.
    """
    rng = np.random.default_rng(seed)
    probes = rng.normal(size=(int(samples), 3))
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)
    best = np.max(probes @ np.asarray(directions, dtype=float).T, axis=1)
    best = np.clip(best, -1.0, 1.0)
    return float(np.degrees(np.arccos(np.min(best))))


def cone_cost_model(
    grid: BurnGrid,
    *,
    order: int = 1,
    axis_weights: tuple[float, float, float] = DEFAULT_AXIS_WEIGHTS,
    directions: np.ndarray | None = None,
) -> CostModel:
    """Polyhedral approximation of the Euclidean burn magnitude.

    Lifted variables are the plus/minus split of ``x`` followed by one
    magnitude variable per burn slot. Axis weights are applied by scaling each
    facet normal componentwise, which charges a weighted norm
    ``||W v||`` rather than ``||v||`` and keeps the model a single cone per
    slot.
    """
    weights = np.asarray([float(w) for w in axis_weights], dtype=float)
    if weights.shape != (3,) or np.any(weights <= 0.0):
        raise AssemblyError("axis_weights must be three positive numbers")

    facets = polyhedral_cone(order) if directions is None else np.asarray(directions, dtype=float)
    if facets.ndim != 2 or facets.shape[1] != 3:
        raise AssemblyError("directions must have shape (L, 3)")
    norms = np.linalg.norm(facets, axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-9):
        raise AssemblyError("cone directions must be unit vectors")

    n_vars = grid.n_vars
    slots = [
        (sat_id, slot)
        for sat_id in grid.satellite_ids
        for slot in range(len(grid.slot_epochs(sat_id)))
    ]
    n_split = 2 * n_vars
    n_lifted = n_split + len(slots)

    lift = np.zeros((n_vars, n_lifted), dtype=float)
    for var in range(n_vars):
        lift[var, 2 * var] = 1.0
        lift[var, 2 * var + 1] = -1.0

    cost = np.zeros(n_lifted, dtype=float)
    cost[n_split:] = 1.0

    column_labels = grid.column_labels()
    labels = [
        label
        for var in range(n_vars)
        for label in (f"{column_labels[var]}:+", f"{column_labels[var]}:-")
    ]
    labels.extend(f"mag:{sat_id}:{slot}" for sat_id, slot in slots)

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    row_labels: list[str] = []
    magnitude_columns: dict[tuple[str, int], list[int]] = {}

    axis_range = range(3) if grid.axes == 3 else (1,)
    for slot_position, (sat_id, slot) in enumerate(slots):
        magnitude_column = n_split + slot_position
        magnitude_columns[(sat_id, slot)] = [magnitude_column]
        for facet_index, facet in enumerate(facets):
            row = np.zeros(n_lifted, dtype=float)
            for axis in axis_range:
                var = grid.index(sat_id, slot, axis)
                coefficient = float(facet[axis] * weights[axis])
                row[2 * var] += coefficient
                row[2 * var + 1] -= coefficient
            row[magnitude_column] = -1.0
            rows.append(row)
            rhs.append(0.0)
            row_labels.append(f"cone:{sat_id}:{slot}:{facet_index}")

    radius = covering_radius_deg(facets)
    return CostModel(
        name=f"cone{facets.shape[0]}",
        lift=lift,
        cost=cost,
        labels=labels,
        extra_rows=np.asarray(rows, dtype=float) if rows else np.zeros((0, n_lifted)),
        extra_rhs=np.asarray(rhs, dtype=float) if rhs else np.zeros(0),
        row_labels=row_labels,
        magnitude_columns=magnitude_columns,
        approximation_note=(
            f"{facets.shape[0]} facets, covering radius {radius:.2f} deg, "
            f"under-charges the weighted Euclidean norm by at most "
            f"{100.0 * (1.0 - np.cos(np.radians(radius))):.2f}%. "
            f"Axis weights (R,T,N) = {tuple(weights.tolist())}."
        ),
    )
