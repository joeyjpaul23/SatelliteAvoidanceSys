"""Choosing the active-set threshold, and why it is not 0.5.

The two errors are not symmetric, and the asymmetry is the whole design.

* A **missed** active row means the reduced program omits a constraint the
  optimum needs. :func:`aegis.fleetopt.solver.lazy_solve` finds it at the next
  round and adds it back, so the cost is one extra LP solve.
* A **falsely included** row means the reduced program carries a constraint it
  did not need. The cost is a slightly larger LP.

Neither costs safety -- the loop terminates only when every omitted row is
satisfied, and the returned solution is optimal for the full problem whatever
the guess was. So the threshold is chosen to hit a recall target, not to
maximise accuracy, and the number worth reporting alongside it is the fraction
of rows actually retained, because that is what the speedup comes from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .dataset import collate
from .features import GraphSample
from .model import ConjunctionPIGNN

__all__ = ["CalibrationResult", "calibrate_threshold"]


@dataclass
class CalibrationResult:
    """The chosen threshold and what it actually achieves."""

    threshold: float
    target_recall: float
    achieved_recall: float
    precision: float
    rows_retained: float
    active_rows: int
    total_rows: int
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "threshold": round(self.threshold, 8),
            "target_recall": self.target_recall,
            "achieved_recall": round(self.achieved_recall, 6),
            "precision": round(self.precision, 6),
            "rows_retained": round(self.rows_retained, 6),
            "active_rows": self.active_rows,
            "total_rows": self.total_rows,
            "notes": list(self.notes),
        }


@torch.no_grad()
def calibrate_threshold(
    model: ConjunctionPIGNN,
    samples: list[GraphSample],
    *,
    target_recall: float = 0.999,
    batch_size: int = 8,
) -> CalibrationResult:
    """Lowest threshold on the validation split reaching ``target_recall``.

    "Lowest" rather than "any": every reduction in the threshold retains more
    rows, so the smallest one that clears the target is the one that keeps the
    program smallest. With no active rows to learn from the threshold falls
    back to zero -- keep everything -- which is slow but correct, and the note
    says so rather than inventing a number.
    """
    model.eval()
    probabilities: list[np.ndarray] = []
    actual: list[np.ndarray] = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        if not chunk:
            continue
        batch = collate(chunk)
        if batch.edge_active is None or batch.edge_active.numel() == 0:
            continue
        prediction = model(batch)
        probabilities.append(
            torch.sigmoid(prediction["edge_active_logit"]).detach().cpu().numpy()
        )
        actual.append((batch.edge_active > 0.5).detach().cpu().numpy())

    if not probabilities:
        return CalibrationResult(
            threshold=0.0,
            target_recall=target_recall,
            achieved_recall=1.0,
            precision=0.0,
            rows_retained=1.0,
            active_rows=0,
            total_rows=0,
            notes=["no labelled edges available; keeping every row"],
        )

    scores = np.concatenate(probabilities)
    labels = np.concatenate(actual)
    positives = int(labels.sum())
    if positives == 0:
        return CalibrationResult(
            threshold=0.0,
            target_recall=target_recall,
            achieved_recall=1.0,
            precision=0.0,
            rows_retained=1.0,
            active_rows=0,
            total_rows=int(labels.size),
            notes=[
                "the validation split contains no active rows, so no threshold can be "
                "calibrated; keeping every row, which is correct but gives no speedup"
            ],
        )

    candidates = np.unique(np.concatenate([scores, np.array([0.0, 1.0])]))[::-1]
    chosen = 0.0
    for candidate in candidates:
        predicted = scores >= candidate
        recall = float((predicted & labels).sum()) / positives
        if recall >= target_recall:
            chosen = float(candidate)
            break

    predicted = scores >= chosen
    true_positive = float((predicted & labels).sum())
    return CalibrationResult(
        threshold=chosen,
        target_recall=target_recall,
        achieved_recall=true_positive / positives,
        precision=true_positive / max(float(predicted.sum()), 1.0),
        rows_retained=float(predicted.sum()) / float(labels.size),
        active_rows=positives,
        total_rows=int(labels.size),
    )
