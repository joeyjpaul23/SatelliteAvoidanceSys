"""Assembled linear (or mixed-integer linear) program for a fleet plan.

This module is deliberately pure data and pure assembly: no solver is called
here, nothing is cached, and calling :func:`assemble` twice on the same inputs
returns numerically identical arrays. That makes the program itself testable
-- a row can be checked against the proposition it encodes without running an
optimizer -- and it makes the learned active-set predictor auditable, because
the rows it proposes to drop are addressable by name.

The program
-----------

.. code-block:: text

    minimise    c' z                                  (delta-v, plus operations cost)
                  + lambda * sum_j s_j                (resolve slack)
                  + mu     * sum_p sigma_p            (induced slack)

    subject to  u_j' P_j (d_j + B_j x) + s_j >= rho_j            for j in resolve
                u_p' (d_p + B_p x)   + sigma_p >= floor_p        for p in latent
                sum_{k,axis} |dv[i,k,axis]|        <= D_i        per satellite
                |dv[i,k,axis]|                     <= dv_cap     per component
                |e_m' delta_r_i(t_eval)|           <= box_m      station keeping
                sum_p sigma_p                      <= epsilon    induced budget
                x = lift z,  z >= 0,  s >= 0,  sigma >= 0

Slack variables on the safety rows are **mandatory**, not optional. A purely
radial or cross-track geometry has a bounded achievable response, so no finite
delta-v satisfies it; without slack the program would be infeasible and the
operator would get an exception instead of a ranked list of what could not be
resolved. :mod:`aegis.maneuver.planner` made the same choice for the same
reason and this module preserves it.

Every row and every column carries a label. Labels are the interface the
solver's duals, the certificate, the Farkas explanation, and the learned
predictor all speak, so they are stable and machine-parseable:

.. code-block:: text

    resolve:<conjunction-id>
    latent:<pair>@<iso-epoch>:<kind>
    budget:<satellite>
    cap:<satellite>:<slot>:<axis>:<sign>
    sk:<satellite>:<axis>:<sign>
    cone:<satellite>:<slot>:<facet>
    induced-budget
    link:<satellite>            (MILP: ties a satellite's burns to its binary)
    burnlink:<satellite>:<slot> (MILP: ties one slot's burns to its binary)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from scipy.sparse import coo_matrix

from ..constants import (
    DEFAULT_DV_BUDGET_KM_S,
    LP_SLACK_PENALTY,
    MAX_DV_PER_BURN_KM_S,
    STATION_KEEPING_BOX_KM,
)
from ..core.timebase import ensure_utc, shift
from .bplane import MissSensitivity, linearize
from .dynamics import BurnGrid, displacement_operator
from .errors import AssemblyError
from .latent import LatentConstraint
from .norms import CostModel, l1_cost_model

__all__ = [
    "FleetProblem",
    "LinearProgramData",
    "assemble",
    "station_keeping_rows",
]

_BIG_M_SAFETY_FACTOR = 1.0


@dataclass
class LinearProgramData:
    """Solver-ready arrays plus the labels that make them interpretable."""

    c: np.ndarray
    a_ub: np.ndarray
    """Inequality matrix, dense or CSR.

    :func:`assemble` returns a dense array for small programs and a
    ``scipy.sparse`` CSR matrix once the dense form would be large -- see
    ``_SPARSE_ELEMENT_THRESHOLD``. Both are accepted everywhere this field is
    consumed (``scipy.optimize.linprog`` and ``LinearConstraint`` both take
    either), and the two agree entrywise; only the storage differs.
    """
    b_ub: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    integrality: np.ndarray
    row_labels: list[str]
    col_labels: list[str]
    n_lifted: int
    n_resolve: int
    n_latent: int
    cost_model: CostModel

    def __post_init__(self) -> None:
        if self.a_ub.shape[0] != self.b_ub.shape[0]:
            raise AssemblyError("A_ub and b_ub row counts disagree")
        if self.a_ub.shape[1] != self.c.shape[0]:
            raise AssemblyError("A_ub width and objective length disagree")
        if len(self.row_labels) != self.a_ub.shape[0]:
            raise AssemblyError("one row label per row is required")
        if len(self.col_labels) != self.c.shape[0]:
            raise AssemblyError("one column label per column is required")
        self._row_lookup = {label: index for index, label in enumerate(self.row_labels)}
        self._col_lookup = {label: index for index, label in enumerate(self.col_labels)}

    @property
    def n_rows(self) -> int:
        return int(self.a_ub.shape[0])

    @property
    def n_cols(self) -> int:
        return int(self.c.shape[0])

    @property
    def bounds(self) -> list[tuple[float, float | None]]:
        return [
            (float(lo), None if not np.isfinite(hi) else float(hi))
            for lo, hi in zip(self.lower, self.upper)
        ]

    @property
    def is_integer(self) -> bool:
        return bool(np.any(self.integrality > 0))

    def row_index(self, label: str) -> int:
        index = self._row_lookup.get(label)
        if index is None:
            raise AssemblyError(f"no row labelled {label!r}")
        return index

    def col_index(self, label: str) -> int:
        index = self._col_lookup.get(label)
        if index is None:
            raise AssemblyError(f"no column labelled {label!r}")
        return index

    def resolve_slack_columns(self) -> list[int]:
        return [i for i, label in enumerate(self.col_labels) if label.startswith("slack:resolve:")]

    def latent_slack_columns(self) -> list[int]:
        return [i for i, label in enumerate(self.col_labels) if label.startswith("slack:latent:")]

    def decision_vector(self, z_full: np.ndarray) -> np.ndarray:
        """Recover the signed impulse vector ``x`` from a full solution."""
        z_full = np.asarray(z_full, dtype=float)
        return self.cost_model.recover(z_full[: self.n_lifted])

    def summary(self) -> dict:
        return {
            "rows": self.n_rows,
            "cols": self.n_cols,
            "resolve_rows": self.n_resolve,
            "latent_rows": self.n_latent,
            "integer": self.is_integer,
            "cost_model": self.cost_model.name,
            "cost_note": self.cost_model.approximation_note,
            "nonzeros": int(getattr(self.a_ub, "nnz", 0) or np.count_nonzero(self.a_ub)),
        }


@dataclass
class FleetProblem:
    """Everything needed to assemble the program, and nothing more."""

    grid: BurnGrid
    resolve: list[MissSensitivity] = field(default_factory=list)
    latent: list[LatentConstraint] = field(default_factory=list)
    cost_model: CostModel | None = None
    budgets: dict[str, float] = field(default_factory=dict)
    per_burn_cap_km_s: float = MAX_DV_PER_BURN_KM_S
    station_keeping_box_km: tuple[float, float] = STATION_KEEPING_BOX_KM
    station_keeping_epoch: dict[str, datetime] = field(default_factory=dict)
    resolve_slack_penalty: float = LP_SLACK_PENALTY
    latent_slack_penalty: float = LP_SLACK_PENALTY
    induced_budget: float | None = None
    total_slack_budget: float | None = None
    slack_caps: dict[str, float] = field(default_factory=dict)
    ops_cost_per_satellite: float = 0.0
    ops_cost_per_burn: float = 0.0
    integer: bool = False
    enforce_station_keeping: bool = True

    def __post_init__(self) -> None:
        if self.cost_model is None:
            self.cost_model = l1_cost_model(self.grid)
        if self.cost_model.n_vars != self.grid.n_vars:
            raise AssemblyError("cost model was built for a different burn grid")
        for sensitivity in self.resolve:
            if sensitivity.n_vars != self.grid.n_vars:
                raise AssemblyError(
                    f"resolve row {sensitivity.conjunction_id} has "
                    f"{sensitivity.n_vars} columns, grid has {self.grid.n_vars}"
                )
        for constraint in self.latent:
            if constraint.n_vars != self.grid.n_vars:
                raise AssemblyError(
                    f"latent row {constraint.label} has {constraint.n_vars} columns, "
                    f"grid has {self.grid.n_vars}"
                )
        if self.per_burn_cap_km_s <= 0.0:
            raise AssemblyError("per_burn_cap_km_s must be positive")
        if self.induced_budget is not None and self.induced_budget < 0.0:
            raise AssemblyError("induced_budget must be non-negative when set")
        if self.total_slack_budget is not None and self.total_slack_budget < 0.0:
            raise AssemblyError("total_slack_budget must be non-negative when set")
        for sat_id in self.grid.satellite_ids:
            self.budgets.setdefault(sat_id, DEFAULT_DV_BUDGET_KM_S)
            if self.budgets[sat_id] < 0.0:
                raise AssemblyError(f"budget for {sat_id} must be non-negative")

    @property
    def n_vars(self) -> int:
        return self.grid.n_vars

    def budget(self, satellite_id: str) -> float:
        return float(self.budgets.get(satellite_id, DEFAULT_DV_BUDGET_KM_S))

    def nominal_directions(self) -> dict[str, np.ndarray]:
        """Starting linearization directions: the nominal miss directions."""
        return {s.conjunction_id: s.nominal_direction for s in self.resolve}

    def resolve_pair_ids(self) -> frozenset[str]:
        return frozenset(
            f"{min(s.primary_id, s.secondary_id)}:{max(s.primary_id, s.secondary_id)}"
            for s in self.resolve
        )

    def with_latent(self, latent: list[LatentConstraint]) -> "FleetProblem":
        """A copy carrying a different latent row set, for lazy solving."""
        return FleetProblem(
            grid=self.grid,
            resolve=list(self.resolve),
            latent=list(latent),
            cost_model=self.cost_model,
            budgets=dict(self.budgets),
            per_burn_cap_km_s=self.per_burn_cap_km_s,
            station_keeping_box_km=self.station_keeping_box_km,
            station_keeping_epoch=dict(self.station_keeping_epoch),
            resolve_slack_penalty=self.resolve_slack_penalty,
            latent_slack_penalty=self.latent_slack_penalty,
            induced_budget=self.induced_budget,
            total_slack_budget=self.total_slack_budget,
            slack_caps=dict(self.slack_caps),
            ops_cost_per_satellite=self.ops_cost_per_satellite,
            ops_cost_per_burn=self.ops_cost_per_burn,
            integer=self.integer,
            enforce_station_keeping=self.enforce_station_keeping,
        )


def station_keeping_rows(
    grid: BurnGrid,
    satellite_id: str,
    *,
    evaluation_epoch: datetime | None = None,
) -> tuple[np.ndarray, list[str]]:
    """In-track and cross-track displacement rows for the slot box.

    Evaluated one orbital period after the satellite's last burn unless an
    explicit epoch is given. A collision avoidance maneuver perturbs the
    semi-major axis, which drifts the satellite out of its constellation slot;
    constraining the displacement is what stops the optimizer trading a
    conjunction for a slot violation. The same reasoning and the same box are
    already in :data:`aegis.constants.STATION_KEEPING_BOX_KM`.

    Returns signed rows for the transverse and normal axes; the caller adds
    both the positive and negative sense.
    """
    epochs = grid.slot_epochs(satellite_id)
    n_rad_s = grid.mean_motion[satellite_id]
    if evaluation_epoch is None:
        period_s = 2.0 * np.pi / n_rad_s
        evaluation_epoch = shift(max(epochs), period_s)
    operator = displacement_operator(grid, satellite_id, ensure_utc(evaluation_epoch))
    return operator[np.array([1, 2])], ["T", "N"]


#: Element count above which :func:`assemble` returns a CSR matrix instead of a
#: dense one. 4e6 elements is 32 MB dense, which every program small enough to
#: inspect by hand stays well under; the switch matters only where the dense
#: form would not fit comfortably in memory anyway.
_SPARSE_ELEMENT_THRESHOLD = 4_000_000


def assemble(
    problem: FleetProblem,
    directions: dict[str, np.ndarray] | None = None,
) -> LinearProgramData:
    """Build the solver arrays for one linearization.

    ``directions`` maps conjunction id to the unit vector used in the
    Cauchy-Schwarz restriction (Proposition 2). Missing entries fall back to
    the nominal miss direction, which is the starting point of the sequential
    refinement in :mod:`aegis.fleetopt.solver`.
    """
    grid = problem.grid
    cost_model = problem.cost_model
    assert cost_model is not None  # established in __post_init__
    directions = dict(directions or {})

    n_lifted = cost_model.n_lifted
    n_resolve = len(problem.resolve)
    n_latent = len(problem.latent)
    satellites = list(grid.satellite_ids)
    n_binary = 2 * len(satellites) if problem.integer else 0

    slack_offset = n_lifted
    latent_offset = slack_offset + n_resolve
    binary_offset = latent_offset + n_latent
    n_cols = binary_offset + n_binary

    c = np.zeros(n_cols, dtype=float)
    c[:n_lifted] = cost_model.cost
    c[slack_offset:latent_offset] = float(problem.resolve_slack_penalty)
    c[latent_offset:binary_offset] = float(problem.latent_slack_penalty)

    col_labels = list(cost_model.labels)
    col_labels.extend(f"slack:resolve:{s.conjunction_id}" for s in problem.resolve)
    col_labels.extend(f"slack:{constraint.label}" for constraint in problem.latent)
    if problem.integer:
        col_labels.extend(f"bin:sat:{sat_id}" for sat_id in satellites)
        col_labels.extend(f"bin:burn:{sat_id}" for sat_id in satellites)
        for position, sat_id in enumerate(satellites):
            c[binary_offset + position] = float(problem.ops_cost_per_satellite)
            c[binary_offset + len(satellites) + position] = float(problem.ops_cost_per_burn)

    lower = np.zeros(n_cols, dtype=float)
    upper = np.full(n_cols, np.inf, dtype=float)

    # Per-row slack ceilings. Stage two of a lexicographic solve uses these to
    # pin the *distribution* of shortfall, not merely its total: bounding only
    # the sum lets the optimizer trade a resolved conjunction for a
    # slightly-worse-elsewhere one at no cost to the objective, so the plan
    # that minimises delta-v can quietly resolve fewer conjunctions than the
    # plan that established the budget.
    if problem.slack_caps:
        for position, sensitivity in enumerate(problem.resolve):
            cap = problem.slack_caps.get(f"resolve:{sensitivity.conjunction_id}")
            if cap is not None:
                upper[slack_offset + position] = max(0.0, float(cap))
        for position, constraint in enumerate(problem.latent):
            cap = problem.slack_caps.get(constraint.label)
            if cap is not None:
                upper[latent_offset + position] = max(0.0, float(cap))
    integrality = np.zeros(n_cols, dtype=float)
    if problem.integer:
        integrality[binary_offset:] = 1.0
        upper[binary_offset:] = 1.0

    # Per-component caps become simple bounds on the plus/minus columns.
    cap = float(problem.per_burn_cap_km_s)
    for column, label in enumerate(cost_model.labels):
        if label.startswith("dv:"):
            upper[column] = cap
        elif label.startswith("mag:"):
            upper[column] = cap * float(np.sqrt(3.0))

    # Rows are accumulated as coordinate triplets rather than as dense vectors.
    # The slack block is diagonal -- each resolve and latent row touches
    # exactly one slack column -- so a latent row holds `n_lifted + 1`
    # nonzeros out of `n_cols`, and `n_cols` grows with the number of latent
    # rows. Storing them densely is therefore quadratic in the one dimension
    # that actually grows: at 11,936 rows the dense matrix and the list of
    # dense rows it is built from together reached 4.7 GB, against roughly
    # 20 MB of triplets for the same program.
    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []
    rhs: list[float] = []
    row_labels: list[str] = []

    def add_row(row: np.ndarray, bound: float, label: str) -> None:
        nonzero = np.flatnonzero(row)
        position = len(rhs)
        row_indices.extend([position] * nonzero.size)
        col_indices.extend(nonzero.tolist())
        values.extend(row[nonzero].tolist())
        rhs.append(float(bound))
        row_labels.append(label)

    def add_lifted_row(
        lifted: np.ndarray, slack_column: int, bound: float, label: str
    ) -> None:
        """A row that is ``lifted`` over the burn block plus one slack entry.

        This is the shape every resolve and latent row has, and they are the
        rows there are thousands of. Taking it directly avoids materialising
        an ``n_cols``-wide temporary per row, which is the difference between
        a transient allocation proportional to the latent count squared and
        one proportional to the burn count.
        """
        nonzero = np.flatnonzero(lifted)
        position = len(rhs)
        row_indices.extend([position] * nonzero.size)
        col_indices.extend(nonzero.tolist())
        values.extend(lifted[nonzero].tolist())
        row_indices.append(position)
        col_indices.append(slack_column)
        values.append(-1.0)
        rhs.append(float(bound))
        row_labels.append(label)

    if cost_model.extra_rows.size:
        for offset in range(cost_model.extra_rows.shape[0]):
            row = np.zeros(n_cols, dtype=float)
            row[:n_lifted] = cost_model.extra_rows[offset]
            add_row(row, float(cost_model.extra_rhs[offset]), cost_model.row_labels[offset])

    # Resolve rows, written as <= by negation so every row is an upper bound.
    for position, sensitivity in enumerate(problem.resolve):
        direction = directions.get(sensitivity.conjunction_id, sensitivity.nominal_direction)
        coefficients, bound = linearize(sensitivity, direction)
        add_lifted_row(
            -(coefficients @ cost_model.lift),
            slack_offset + position,
            -bound,
            f"resolve:{sensitivity.conjunction_id}",
        )

    # Latent rows, same sign convention.
    for position, constraint in enumerate(problem.latent):
        coefficients, bound = constraint.row()
        add_lifted_row(
            -(coefficients @ cost_model.lift),
            latent_offset + position,
            -bound,
            constraint.label,
        )

    # Per-satellite delta-v budget. The charged magnitude of each burn is
    # whatever the cost model charges for it -- the plus/minus split for the
    # L1 model, the auxiliary cone variable for the polyhedral model -- so
    # reading it off `magnitude_columns` keeps this row consistent with the
    # objective instead of re-deriving a layout.
    def magnitude_row(sat_id: str) -> np.ndarray:
        row = np.zeros(n_cols, dtype=float)
        for slot in range(len(grid.slot_epochs(sat_id))):
            for column in cost_model.magnitude_columns.get((sat_id, slot), ()):
                row[column] += 1.0
        if not row.any():
            raise AssemblyError(
                f"cost model {cost_model.name!r} exposes no magnitude columns for {sat_id!r}"
            )
        return row

    for position, sat_id in enumerate(satellites):
        row = magnitude_row(sat_id)
        budget = problem.budget(sat_id)
        if problem.integer:
            row[binary_offset + position] = -budget
            add_row(row, 0.0, f"link:{sat_id}")
        else:
            add_row(row, budget, f"budget:{sat_id}")

    if problem.integer:
        # A second binary counts whether any burn happens at all, so the two
        # operations costs can be priced independently: one charge for
        # disturbing a satellite, one for each commanded burn.
        for position, sat_id in enumerate(satellites):
            row = magnitude_row(sat_id)
            row[binary_offset + len(satellites) + position] = -problem.budget(sat_id)
            add_row(row, 0.0, f"burnlink:{sat_id}")
        for position, sat_id in enumerate(satellites):
            row = np.zeros(n_cols, dtype=float)
            row[binary_offset + len(satellites) + position] = 1.0
            row[binary_offset + position] = -1.0
            add_row(row, 0.0, f"binorder:{sat_id}")

    # Station keeping.
    if problem.enforce_station_keeping:
        box = (
            float(problem.station_keeping_box_km[0]),
            float(problem.station_keeping_box_km[1]),
        )
        for sat_id in satellites:
            operator, axis_labels = station_keeping_rows(
                grid, sat_id, evaluation_epoch=problem.station_keeping_epoch.get(sat_id)
            )
            for axis_position, axis_label in enumerate(axis_labels):
                lifted = operator[axis_position] @ cost_model.lift
                limit = box[axis_position]
                positive = np.zeros(n_cols, dtype=float)
                positive[:n_lifted] = lifted
                add_row(positive, limit, f"sk:{sat_id}:{axis_label}:+")
                negative = np.zeros(n_cols, dtype=float)
                negative[:n_lifted] = -lifted
                add_row(negative, limit, f"sk:{sat_id}:{axis_label}:-")

    # Total-shortfall budget. Stage two of a lexicographic solve pins the
    # safety achieved in stage one and then minimises delta-v underneath it.
    # Without this row a penalty formulation never really minimises delta-v
    # whenever any conjunction is short: at 1e4 per km of shortfall against a
    # delta-v cost of order 1e-3, the optimizer will spend the entire budget
    # to shave a micrometre, and the reported delta-v is whatever the solver
    # happened to land on rather than a minimum of anything.
    if problem.total_slack_budget is not None and (n_resolve or n_latent):
        row = np.zeros(n_cols, dtype=float)
        row[slack_offset:binary_offset] = 1.0
        add_row(row, float(problem.total_slack_budget), "slack-budget")

    # Induced-risk budget: the epsilon whose parametric sweep traces the
    # safety-premium frontier (see aegis.fleetopt.pareto).
    if problem.induced_budget is not None and n_latent:
        row = np.zeros(n_cols, dtype=float)
        row[latent_offset:binary_offset] = 1.0
        add_row(row, float(problem.induced_budget), "induced-budget")

    n_rows = len(rhs)
    if n_rows and n_rows * n_cols > _SPARSE_ELEMENT_THRESHOLD:
        a_ub = coo_matrix(
            (values, (row_indices, col_indices)), shape=(n_rows, n_cols), dtype=float
        ).tocsr()
    else:
        a_ub = np.zeros((n_rows, n_cols), dtype=float)
        if values:
            a_ub[row_indices, col_indices] = values
    b_ub = np.asarray(rhs, dtype=float) if rhs else np.zeros(0, dtype=float)

    return LinearProgramData(
        c=c,
        a_ub=a_ub,
        b_ub=b_ub,
        lower=lower,
        upper=upper,
        integrality=integrality,
        row_labels=row_labels,
        col_labels=col_labels,
        n_lifted=n_lifted,
        n_resolve=n_resolve,
        n_latent=n_latent,
        cost_model=cost_model,
    )
