"""Turning a trained network into hints the optimizer may use but never trusts.

Every hint here is *safe by construction*, and each for a different reason:

``active_edges``
    A guess at which latent rows bind. :func:`aegis.fleetopt.solver.lazy_solve`
    solves with only those, then checks every omitted row and adds back any
    that is violated, terminating only when none is. The returned solution is
    optimal for the full problem for *any* guess, including an empty or
    adversarial one, so a bad prediction costs rounds and nothing else.

``initial_directions``
    Starting points for the sequential linearization. By Proposition 2 the
    affine restriction is conservative for *any* unit direction, so a
    network-supplied direction can make the plan more expensive or the
    refinement slower, never unsafe.

``predicted_dv``
    The raw decision vector. This one is **not** safe, and nothing uses it
    except the deliberately uncertified ``pignn-direct`` planner, which exists
    so its violation rate can be reported next to the certified planners
    rather than assumed away.

A missing or unreadable checkpoint yields ``None``. The learned planners then
fall back to their exact equivalents and say so. Fabricating hints -- or
silently substituting an optimizer result and calling it a prediction -- would
make the comparison meaningless.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..fleetopt.bplane import refine_direction
from ..fleetopt.planners import LearnedHints, PlanContext
from .dataset import collate
from .features import FeatureSpec, FeatureStats, apply_normalisation, tensorize
from .model import ConjunctionPIGNN, ModelConfig

__all__ = ["HintsProvider", "load_hints_provider", "predict_hints"]


@dataclass
class HintsProvider:
    """A loaded checkpoint, ready to be asked about a planning context."""

    model: ConjunctionPIGNN
    stats: FeatureStats
    threshold: float
    model_id: str
    metrics: dict

    def __call__(self, context: PlanContext, scenario=None) -> LearnedHints | None:
        return predict_hints(self, context, scenario=scenario)


def load_hints_provider(
    checkpoint: str | Path,
    *,
    threshold: float | None = None,
) -> HintsProvider | None:
    """Load a checkpoint, or return ``None`` with the reason left to the caller.

    Raises only on a *schema* mismatch, which is a genuine incompatibility
    between a model and the features it would be fed. A missing file is not an
    error -- running without a model is a supported configuration and the
    planners handle it -- so it returns ``None``.
    """
    path = Path(checkpoint)
    if not path.exists():
        return None
    try:
        payload = torch.load(io.BytesIO(path.read_bytes()), weights_only=False)
    except (OSError, RuntimeError, EOFError):
        return None

    config = ModelConfig.from_json(payload["config"])
    stats = FeatureStats.from_json(payload["feature_stats"])
    spec = FeatureSpec(schema_version=stats.schema_version)
    spec.check(config.feature_schema)

    model = ConjunctionPIGNN(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return HintsProvider(
        model=model,
        stats=stats,
        threshold=float(threshold if threshold is not None else payload.get("threshold", 0.5)),
        model_id=str(payload.get("model_id", path.stem)),
        metrics=dict(payload.get("test_metrics", {})),
    )


@torch.no_grad()
def predict_hints(
    provider: HintsProvider,
    context: PlanContext,
    *,
    scenario=None,
) -> LearnedHints | None:
    """Run the network on one planning context and package the hints."""
    if not context.sensitivities or context.grid.n_vars == 0:
        return None

    try:
        sample = tensorize(scenario, context)
        sample = apply_normalisation(sample, provider.stats)
        batch = collate([sample])
        prediction = provider.model(batch)
    except Exception:  # noqa: BLE001 - an unusable hint must not break planning
        return None

    probabilities = torch.sigmoid(prediction["edge_active_logit"]).cpu().numpy()
    labels = sample.edge_labels
    edge_probabilities = {
        label: float(value) for label, value in zip(labels, probabilities)
    }
    active = {
        label
        for label, value in edge_probabilities.items()
        if label.startswith("latent:") and value >= provider.threshold
    }

    x = _decode(prediction["node_dv"].cpu().numpy(), sample)
    directions = {}
    if x is not None:
        for sensitivity in context.sensitivities:
            directions[sensitivity.conjunction_id] = refine_direction(sensitivity, x)

    return LearnedHints(
        active_edges=active,
        edge_probabilities=edge_probabilities,
        initial_directions=directions,
        predicted_dv=x,
        model_id=provider.model_id,
        threshold=provider.threshold,
    )


def _decode(node_dv: np.ndarray, sample) -> np.ndarray | None:
    """Scatter the per-node prediction back into the decision vector."""
    if sample.n_vars == 0:
        return None
    x = np.zeros(sample.n_vars, dtype=float)
    columns = sample.node_columns.cpu().numpy()
    for node in range(min(node_dv.shape[0], columns.shape[0])):
        for slot in range(min(node_dv.shape[1], columns.shape[1])):
            column = int(columns[node, slot])
            if 0 <= column < sample.n_vars:
                x[column] += float(node_dv[node, slot])
    return x
