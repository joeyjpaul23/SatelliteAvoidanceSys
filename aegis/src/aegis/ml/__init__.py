"""Physics-informed graph learning for fleet maneuver optimization.

The package's single claim, stated once: **the network accelerates, the
optimizer certifies.** Nothing here is trusted. A prediction enters the
optimizer only as a guess at which constraints matter and where to linearize,
and :func:`aegis.fleetopt.solver.lazy_solve` returns the exact optimum of the
full program whatever the guess was. See ``docs/prior-art-and-novelty-ledger.md``
claim C6 for the positioning against Bertsimas & Stellato, Cauligi et al. and
Gasse et al.
"""

from __future__ import annotations

from .calibrate import CalibrationResult, calibrate_threshold
from .dataset import (
    DatasetEntry,
    DatasetIndex,
    GraphBatch,
    build_labels,
    collate,
    generate_dataset,
    load_dataset,
    load_manifest,
    split_for_digest,
)
from .features import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    SCHEMA_VERSION,
    FeatureSpec,
    FeatureStats,
    GraphSample,
    apply_normalisation,
    fit_normalisation,
    tensorize,
)
from .infer import HintsProvider, load_hints_provider, predict_hints
from .model import ConjunctionPIGNN, LossTerms, ModelConfig, ModelConfigMismatch, PignnLoss
from .train import TrainConfig, TrainResult, evaluate, train

__all__ = [
    "CalibrationResult", "calibrate_threshold",
    "DatasetEntry", "DatasetIndex", "GraphBatch", "build_labels", "collate",
    "generate_dataset", "load_dataset", "load_manifest", "split_for_digest",
    "EDGE_FEATURE_NAMES", "NODE_FEATURE_NAMES", "SCHEMA_VERSION",
    "FeatureSpec", "FeatureStats", "GraphSample",
    "apply_normalisation", "fit_normalisation", "tensorize",
    "HintsProvider", "load_hints_provider", "predict_hints",
    "ConjunctionPIGNN", "LossTerms", "ModelConfig", "ModelConfigMismatch", "PignnLoss",
    "TrainConfig", "TrainResult", "evaluate", "train",
]
