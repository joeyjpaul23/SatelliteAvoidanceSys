"""A physics-informed graph network over the conjunction graph.

What it predicts, and why that is the useful thing
--------------------------------------------------
Not a maneuver. The network predicts the **structure of the optimum**: which
safety rows are active at the LP solution, what their dual prices are, and
roughly how the delta-v is distributed across the fleet. The optimizer then
uses that to solve a much smaller program and *proves* the answer is the same
one the full program would have given
(:func:`aegis.fleetopt.solver.lazy_solve`). A wrong prediction costs an extra
cutting-plane round; it cannot cost safety.

That split -- learn the structure, certify the answer -- is what distinguishes
this from the nearest prior art. Bertsimas & Stellato (arXiv:1907.02206) and
Cauligi et al. (arXiv:2004.03736) predict a strategy and solve the reduced
problem, but with a flat network over a fixed-size parameter vector and with
purely empirical certification. Gasse et al. (arXiv:1906.01629) use a GNN, but
to pick branching variables inside a branch-and-bound that was already exact.
A conjunction graph has no fixed size or topology -- the number of satellites,
conjunctions and latent pairs changes every planning cycle -- so a flat
network cannot represent it, and the deterministic closure is what makes the
prediction safe to use.

Architecture
------------
Standard message passing, deliberately: node and edge encoders, ``n_layers``
rounds of attention-weighted aggregation with residual connections and layer
norm, then task heads. Edges are undirected, so every message is sent both
ways. The attention weights are normalised per destination node with a segment
softmax built from ``index_add_``, with the usual max subtraction for
stability; there is no ``torch_geometric`` in this project's dependency set and
none is needed for that.

The physics loss
----------------
The one non-standard piece. The predicted per-node delta-v is scattered back
through ``node_columns`` into the decision vector ``x`` the LP actually uses,
and pushed through the **stored sensitivity tensors** to evaluate the real
constraint residuals:

.. math::

    L_{\\text{phys}} = \\sum_e \\operatorname{relu}\\!\\big(f_e -
        u_e^{\\top}(d_e + B_e \\hat{x})\\big)^2
        + w \\lVert \\hat{x} \\rVert_1 .

So the network is not only imitating a solver's output, it is being told --
in the units of the problem -- how far its proposal is from feasible and how
much fuel it is spending. Those are the same rows
:mod:`aegis.fleetopt.problem` assembles, so the gradient points at the actual
constraint geometry rather than at a proxy for it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from .features import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES, SCHEMA_VERSION

__all__ = [
    "ModelConfig",
    "ConjunctionPIGNN",
    "PignnLoss",
    "LossTerms",
    "ModelConfigMismatch",
    "segment_softmax",
    "decode_decision_vector",
]


class ModelConfigMismatch(ValueError):
    """A checkpoint's architecture does not match the one being loaded into."""


@dataclass(frozen=True)
class ModelConfig:
    """Everything that determines the architecture.

    Stored with every checkpoint and compared on load. Silently loading
    weights into a differently-shaped model is the kind of failure that
    produces plausible numbers, so it raises.
    """

    node_features: int = len(NODE_FEATURE_NAMES)
    edge_features: int = len(EDGE_FEATURE_NAMES)
    hidden: int = 96
    n_layers: int = 4
    heads: int = 4
    dropout: float = 0.05
    k_max: int = 8
    feature_schema: str = SCHEMA_VERSION
    dual_scale: float = 1e-4
    cost_scale: float = 1e-3

    def check(self, other: "ModelConfig") -> None:
        if asdict(self) != asdict(other):
            differences = [
                f"{key}: {value!r} != {getattr(other, key)!r}"
                for key, value in asdict(self).items()
                if value != getattr(other, key)
            ]
            raise ModelConfigMismatch(
                "checkpoint architecture does not match: " + "; ".join(differences)
            )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "ModelConfig":
        known = {field_name for field_name in asdict(cls()).keys()}
        return cls(**{key: value for key, value in payload.items() if key in known})


def _mlp(sizes: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[index], sizes[index + 1]))
        if index < len(sizes) - 2:
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def segment_softmax(scores: Tensor, index: Tensor, n_segments: int) -> Tensor:
    """Softmax over the entries sharing a destination, with max subtraction.

    ``scores`` is ``(M, H)`` and ``index`` is ``(M,)`` giving each row's
    segment. Implemented with ``index_add_`` and ``scatter_reduce_`` rather
    than a library call so the package keeps no graph-library dependency.
    """
    if scores.numel() == 0:
        return scores
    maxima = torch.full(
        (n_segments, scores.shape[1]), float("-inf"), dtype=scores.dtype, device=scores.device
    )
    maxima = maxima.scatter_reduce(
        0, index.unsqueeze(1).expand_as(scores), scores, reduce="amax", include_self=True
    )
    maxima = torch.nan_to_num(maxima, neginf=0.0)
    shifted = torch.exp(scores - maxima[index])
    totals = torch.zeros_like(maxima)
    totals.index_add_(0, index, shifted)
    return shifted / (totals[index] + 1e-12)


class MessagePassingLayer(nn.Module):
    """One round of attention-weighted aggregation, both directions."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.hidden
        self.heads = config.heads
        self.head_dim = hidden // config.heads
        if self.head_dim * config.heads != hidden:
            raise ValueError("hidden must be divisible by heads")
        self.message = _mlp([3 * hidden, 2 * hidden, hidden], config.dropout)
        self.attention = nn.Linear(3 * hidden, config.heads)
        self.update = _mlp([2 * hidden, 2 * hidden, hidden], config.dropout)
        self.edge_update = _mlp([3 * hidden, 2 * hidden, hidden], config.dropout)
        self.node_norm = nn.LayerNorm(hidden)
        self.edge_norm = nn.LayerNorm(hidden)

    def forward(self, h: Tensor, e: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        if e.numel() == 0:
            return self.node_norm(h + self.update(torch.cat([h, torch.zeros_like(h)], dim=1))), e

        source, destination = edge_index[0], edge_index[1]
        # Undirected: every edge contributes in both directions.
        src = torch.cat([source, destination])
        dst = torch.cat([destination, source])
        edge_repeat = torch.cat([e, e], dim=0)

        joint = torch.cat([h[src], h[dst], edge_repeat], dim=1)
        raw = self.message(joint)
        weights = segment_softmax(self.attention(joint), dst, h.shape[0])
        weighted = (
            raw.view(-1, self.heads, self.head_dim) * weights.unsqueeze(-1)
        ).reshape(-1, self.heads * self.head_dim)

        aggregated = torch.zeros_like(h)
        aggregated.index_add_(0, dst, weighted)
        h = self.node_norm(h + self.update(torch.cat([h, aggregated], dim=1)))
        e = self.edge_norm(e + self.edge_update(torch.cat([h[source], h[destination], e], dim=1)))
        return h, e


def decode_decision_vector(
    node_dv: Tensor,
    node_columns: Tensor,
    node_batch: Tensor,
    n_graphs: int,
    n_vars_max: int,
) -> Tensor:
    """Scatter a per-node prediction into per-graph decision vectors.

    ``node_columns`` carries the column of ``x`` each ``(node, slot, axis)``
    slot owns, or ``-1`` when it owns none. Returns ``(n_graphs, n_vars_max)``.
    This is the bridge between a graph prediction and the linear program: the
    physics loss is only meaningful because the same ``x`` the LP would see can
    be rebuilt from what the network says.
    """
    x = torch.zeros(n_graphs, n_vars_max, dtype=node_dv.dtype, device=node_dv.device)
    if node_dv.numel() == 0 or n_vars_max == 0:
        return x
    valid = node_columns >= 0
    if not bool(valid.any()):
        return x
    rows = node_batch.unsqueeze(1).expand_as(node_columns)[valid]
    columns = node_columns[valid]
    flat = rows * n_vars_max + columns
    x.view(-1).index_add_(0, flat, node_dv[valid].to(x.dtype))
    return x


class ConjunctionPIGNN(nn.Module):
    """Predicts the active set, the dual prices, and a delta-v allocation."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        hidden = self.config.hidden
        self.node_encoder = _mlp([self.config.node_features, hidden, hidden], self.config.dropout)
        self.edge_encoder = _mlp([self.config.edge_features, hidden, hidden], self.config.dropout)
        self.layers = nn.ModuleList(
            MessagePassingLayer(self.config) for _ in range(self.config.n_layers)
        )
        self.edge_active_head = _mlp([hidden, hidden, 1], self.config.dropout)
        self.edge_dual_head = _mlp([hidden, hidden, 1], self.config.dropout)
        self.node_dv_head = _mlp([hidden, hidden, 3 * self.config.k_max], self.config.dropout)
        self.global_head = _mlp([2 * hidden, hidden, 3], self.config.dropout)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, batch) -> dict[str, Tensor]:
        h = self.node_encoder(batch.node_features)
        e = self.edge_encoder(batch.edge_features)
        for layer in self.layers:
            h, e = layer(h, e, batch.edge_index)

        n_graphs = int(batch.n_graphs)
        pooled_mean = torch.zeros(n_graphs, h.shape[1], dtype=h.dtype, device=h.device)
        counts = torch.zeros(n_graphs, 1, dtype=h.dtype, device=h.device)
        pooled_mean.index_add_(0, batch.node_batch, h)
        counts.index_add_(0, batch.node_batch, torch.ones_like(counts[batch.node_batch]))
        pooled_mean = pooled_mean / counts.clamp_min(1.0)
        pooled_max = torch.full_like(pooled_mean, float("-inf"))
        pooled_max = pooled_max.scatter_reduce(
            0,
            batch.node_batch.unsqueeze(1).expand_as(h),
            h,
            reduce="amax",
            include_self=True,
        )
        pooled_max = torch.nan_to_num(pooled_max, neginf=0.0)
        globals_ = self.global_head(torch.cat([pooled_mean, pooled_max], dim=1))

        # The head is sized for the configured K_max so one checkpoint serves
        # every graph; a batch carries only as many slot columns as its widest
        # member needs, so the prediction is sliced to that width. A batch
        # wider than the checkpoint is a real mismatch and says so.
        width = int(batch.dv_mask.shape[1])
        if width > 3 * self.config.k_max:
            raise ModelConfigMismatch(
                f"batch needs {width} delta-v columns but the model was built for "
                f"{3 * self.config.k_max} (k_max={self.config.k_max})"
            )
        node_dv_raw = self.node_dv_head(h)[:, :width] * batch.dv_mask.to(h.dtype)
        return {
            "edge_active_logit": self.edge_active_head(e).squeeze(-1),
            "edge_dual": torch.nn.functional.softplus(self.edge_dual_head(e).squeeze(-1)),
            "node_dv": node_dv_raw,
            "global_feasible_logit": globals_[:, 0],
            "global_cost": torch.nn.functional.softplus(globals_[:, 1]),
            "global_premium": globals_[:, 2],
        }

    def state(self) -> dict[str, Any]:
        return {
            "config": self.config.to_json(),
            "state_dict": self.state_dict(),
            "parameter_count": self.parameter_count,
        }

    @classmethod
    def load(cls, payload: dict[str, Any], *, config: ModelConfig | None = None) -> "ConjunctionPIGNN":
        stored = ModelConfig.from_json(payload["config"])
        if config is not None:
            config.check(stored)
        model = cls(stored)
        model.load_state_dict(payload["state_dict"])
        return model


@dataclass
class LossTerms:
    """Every loss component, kept separate so a regression is attributable."""

    total: Tensor
    edge_active: Tensor
    edge_dual: Tensor
    node_dv: Tensor
    global_feasible: Tensor
    global_cost: Tensor
    global_premium: Tensor
    physics_violation: Tensor
    physics_fuel: Tensor
    weights: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach()),
            "edge_active": float(self.edge_active.detach()),
            "edge_dual": float(self.edge_dual.detach()),
            "node_dv": float(self.node_dv.detach()),
            "global_feasible": float(self.global_feasible.detach()),
            "global_cost": float(self.global_cost.detach()),
            "global_premium": float(self.global_premium.detach()),
            "physics_violation": float(self.physics_violation.detach()),
            "physics_fuel": float(self.physics_fuel.detach()),
        }


class PignnLoss(nn.Module):
    """Imitation of the solver plus a physics term in the problem's own units."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        weight_active: float = 1.0,
        weight_dual: float = 0.5,
        weight_dv: float = 1.0,
        weight_global: float = 0.2,
        weight_physics: float = 0.0,
        weight_fuel: float = 1e-3,
    ) -> None:
        super().__init__()
        self.config = config
        self.weights = {
            "active": weight_active,
            "dual": weight_dual,
            "dv": weight_dv,
            "global": weight_global,
            "physics": weight_physics,
            "fuel": weight_fuel,
        }
        self.huber = nn.SmoothL1Loss(reduction="mean")

    def set_physics_weight(self, value: float) -> None:
        """Annealed by the trainer: imitation first, physics once it is close."""
        self.weights["physics"] = float(value)

    def forward(self, prediction: dict[str, Tensor], batch) -> LossTerms:
        device = prediction["edge_active_logit"].device
        zero = torch.zeros((), device=device)

        active = zero
        if batch.edge_active is not None and batch.edge_active.numel():
            target = batch.edge_active.to(device=device, dtype=torch.float32)
            positives = float(target.sum())
            negatives = float(target.numel() - positives)
            # Active rows are the minority by construction -- most latent pairs
            # never bind -- so an unweighted BCE learns to answer "inactive".
            pos_weight = torch.tensor(
                max(negatives, 1.0) / max(positives, 1.0), device=device, dtype=torch.float32
            ).clamp(max=100.0)
            active = torch.nn.functional.binary_cross_entropy_with_logits(
                prediction["edge_active_logit"].float(), target, pos_weight=pos_weight
            )

        dual = zero
        if batch.edge_dual is not None and batch.edge_dual.numel():
            target = batch.edge_dual.to(device=device, dtype=torch.float32)
            mask = target > 0.0
            if bool(mask.any()):
                scale = self.config.dual_scale
                dual = self.huber(
                    torch.log1p(prediction["edge_dual"].float()[mask] / scale),
                    torch.log1p(target[mask] / scale),
                )

        node_dv = zero
        if batch.node_dv is not None and batch.node_dv.numel():
            target = batch.node_dv.to(device=device, dtype=torch.float32)
            mask = batch.dv_mask.to(device)
            if bool(mask.any()):
                node_dv = self.huber(
                    prediction["node_dv"].float()[mask] * 1e6, target[mask] * 1e6
                )

        feasible = zero
        if batch.global_feasible is not None and batch.global_feasible.numel():
            feasible = torch.nn.functional.binary_cross_entropy_with_logits(
                prediction["global_feasible_logit"].float(),
                batch.global_feasible.to(device=device, dtype=torch.float32),
            )

        cost = zero
        if batch.global_cost is not None and batch.global_cost.numel():
            scale = self.config.cost_scale
            cost = self.huber(
                torch.log1p(prediction["global_cost"].float() / scale),
                torch.log1p(batch.global_cost.to(device=device, dtype=torch.float32) / scale),
            )

        premium = zero
        if (
            batch.global_premium is not None
            and batch.global_premium_mask is not None
            and bool(batch.global_premium_mask.any())
        ):
            mask = batch.global_premium_mask.to(device)
            premium = self.huber(
                prediction["global_premium"].float()[mask],
                batch.global_premium.to(device=device, dtype=torch.float32)[mask],
            )

        violation, fuel = self._physics(prediction, batch)

        total = (
            self.weights["active"] * active
            + self.weights["dual"] * dual
            + self.weights["dv"] * node_dv
            + self.weights["global"] * (feasible + cost + premium)
            + self.weights["physics"] * violation
            + self.weights["fuel"] * fuel
        )
        return LossTerms(
            total=total,
            edge_active=active,
            edge_dual=dual,
            node_dv=node_dv,
            global_feasible=feasible,
            global_cost=cost,
            global_premium=premium,
            physics_violation=violation,
            physics_fuel=fuel,
            weights=dict(self.weights),
        )

    def _physics(self, prediction: dict[str, Tensor], batch) -> tuple[Tensor, Tensor]:
        """Constraint violation and fuel of the predicted plan, in km and km/s."""
        device = prediction["node_dv"].device
        zero = torch.zeros((), device=device)
        if batch.sensitivities.numel() == 0 or int(batch.sensitivities.shape[-1]) == 0:
            return zero, zero

        n_vars_max = int(batch.sensitivities.shape[-1])
        x = decode_decision_vector(
            prediction["node_dv"].float(),
            batch.node_columns.to(device),
            batch.node_batch.to(device),
            int(batch.n_graphs),
            n_vars_max,
        )
        per_edge_x = x[batch.edge_batch.to(device)]
        sensitivities = batch.sensitivities.to(device=device, dtype=torch.float32)
        displaced = torch.einsum("eij,ej->ei", sensitivities, per_edge_x)
        perturbed = batch.miss_vectors.to(device=device, dtype=torch.float32) + displaced
        projected = (batch.directions.to(device=device, dtype=torch.float32) * perturbed).sum(-1)
        shortfall = torch.relu(batch.floors.to(device=device, dtype=torch.float32) - projected)
        violation = (shortfall**2).mean()
        fuel = x.abs().sum(dim=1).mean() * 1e6
        return violation, fuel
