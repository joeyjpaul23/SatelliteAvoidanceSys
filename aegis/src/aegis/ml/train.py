"""Training the conjunction network, and proving the run is reproducible.

The dataset is split by scenario digest, never by edge. That distinction is
the whole experiment: every edge of a graph shares one LP solution, so an
edge-level split puts the answer to a validation graph into the training set
and the reported accuracy measures memorisation. :mod:`aegis.ml.dataset`
enforces the split at generation time and this module never re-splits.

The physics-loss weight is annealed. Early on the network cannot produce a
delta-v allocation worth evaluating, so a constraint-violation term just adds
noise; once the imitation terms have converged it supplies the gradient that
imitation cannot -- the direction in which a proposal becomes *feasible*,
expressed in kilometres of separation rather than in distance from a
particular solver's output.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

from ..store import ExperimentStore, StoreConfig, open_artifact_store
from .dataset import collate, load_dataset, load_manifest
from .features import FeatureStats, GraphSample
from .model import ConjunctionPIGNN, ModelConfig, PignnLoss

__all__ = ["TrainConfig", "TrainResult", "train", "evaluate"]


@dataclass
class TrainConfig:
    """Everything that determines a training run, and therefore its result."""

    seed: int = 20260912
    epochs: int = 120
    batch_size: int = 8
    learning_rate: float = 3e-3
    weight_decay: float = 1e-5
    patience: int = 25
    physics_weight_max: float = 0.5
    physics_warmup_fraction: float = 0.3
    grad_clip: float = 5.0
    device: str = "cpu"
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model"] = self.model.to_json()
        return payload


@dataclass
class TrainResult:
    """What the run produced, including everything needed to repeat it."""

    config: TrainConfig
    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    train_samples: int = 0
    val_samples: int = 0
    test_metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_key: str = ""
    model_id: str = ""
    parameter_count: int = 0
    wall_clock_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
            "parameter_count": self.parameter_count,
            "wall_clock_s": round(self.wall_clock_s, 3),
            "checkpoint_key": self.checkpoint_key,
            "model_id": self.model_id,
            "test_metrics": dict(self.test_metrics),
            "final_train_loss": self.history[-1]["train"] if self.history else None,
            "notes": list(self.notes),
        }


def _batches(samples: list[GraphSample], size: int, generator: torch.Generator):
    order = torch.randperm(len(samples), generator=generator).tolist()
    for start in range(0, len(order), size):
        chunk = [samples[index] for index in order[start : start + size]]
        if chunk:
            yield collate(chunk)


def _physics_weight(epoch: int, config: TrainConfig) -> float:
    warmup = max(1, int(config.epochs * config.physics_warmup_fraction))
    if epoch < warmup:
        return 0.0
    span = max(1, config.epochs - warmup)
    return config.physics_weight_max * min(1.0, (epoch - warmup + 1) / span)


@torch.no_grad()
def evaluate(
    model: ConjunctionPIGNN,
    loss_fn: PignnLoss,
    samples: list[GraphSample],
    *,
    batch_size: int = 8,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Loss plus the metrics that matter for the downstream use.

    Recall on the active set is the one to watch. A missed active row costs an
    extra cutting-plane round in :func:`aegis.fleetopt.solver.lazy_solve`;
    a falsely-included one costs a slightly larger LP. The asymmetry is why
    :mod:`aegis.ml.calibrate` targets recall rather than accuracy.
    """
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    true_positive = false_negative = false_positive = true_negative = 0
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        if not chunk:
            continue
        batch = collate(chunk)
        prediction = model(batch)
        terms = loss_fn(prediction, batch)
        for key, value in terms.as_dict().items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
        if batch.edge_active is not None and batch.edge_active.numel():
            predicted = torch.sigmoid(prediction["edge_active_logit"]) >= threshold
            actual = batch.edge_active > 0.5
            true_positive += int((predicted & actual).sum())
            false_negative += int((~predicted & actual).sum())
            false_positive += int((predicted & ~actual).sum())
            true_negative += int((~predicted & ~actual).sum())

    metrics = {key: value / max(count, 1) for key, value in totals.items()}
    metrics["active_recall"] = true_positive / max(true_positive + false_negative, 1)
    metrics["active_precision"] = true_positive / max(true_positive + false_positive, 1)
    metrics["rows_retained"] = (true_positive + false_positive) / max(
        true_positive + false_positive + true_negative + false_negative, 1
    )
    metrics["active_rows"] = float(true_positive + false_negative)
    return metrics


def train(
    config: TrainConfig,
    dataset_path: str | Path,
    *,
    store_path: str | None = None,
    artifact_store=None,
    progress: Callable[[str], None] | None = None,
) -> TrainResult:
    """Train, validate, checkpoint, and register the model version."""
    started = time.perf_counter()
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    manifest = load_manifest(open_artifact_store_for(dataset_path))
    stats: FeatureStats | None = manifest.feature_stats
    if stats is None:
        raise ValueError(
            "dataset carries no feature statistics; normalisation must come from the "
            "training split and be stored, never recomputed at inference"
        )

    train_samples = load_dataset(dataset_path, split="train", normalise=True, stats=stats)
    val_samples = load_dataset(dataset_path, split="val", normalise=True, stats=stats)
    test_samples = load_dataset(dataset_path, split="test", normalise=True, stats=stats)
    if not train_samples:
        raise ValueError("training split is empty")

    k_max = max((sample.k_max for sample in train_samples + val_samples), default=1)
    model_config = ModelConfig(
        node_features=int(train_samples[0].node_features.shape[1]),
        edge_features=int(train_samples[0].edge_features.shape[1]),
        hidden=config.model.hidden,
        n_layers=config.model.n_layers,
        heads=config.model.heads,
        dropout=config.model.dropout,
        k_max=max(k_max, config.model.k_max),
        feature_schema=stats.schema_version,
    )
    model = ConjunctionPIGNN(model_config)
    loss_fn = PignnLoss(model_config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    result = TrainResult(
        config=config,
        train_samples=len(train_samples),
        val_samples=len(val_samples),
        parameter_count=model.parameter_count,
    )
    best_state: dict[str, Any] | None = None
    since_improvement = 0

    for epoch in range(config.epochs):
        loss_fn.set_physics_weight(_physics_weight(epoch, config))
        model.train()
        running = 0.0
        batches = 0
        for batch in _batches(train_samples, config.batch_size, generator):
            optimizer.zero_grad(set_to_none=True)
            terms = loss_fn(model(batch), batch)
            terms.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            running += float(terms.total.detach())
            batches += 1

        train_loss = running / max(batches, 1)
        validation = (
            evaluate(model, loss_fn, val_samples, batch_size=config.batch_size)
            if val_samples
            else {"total": train_loss}
        )
        record = {
            "epoch": epoch,
            "train": train_loss,
            "val": validation["total"],
            "physics_weight": loss_fn.weights["physics"],
            "active_recall": validation.get("active_recall", 0.0),
            "active_precision": validation.get("active_precision", 0.0),
        }
        result.history.append(record)
        if progress is not None:
            progress(
                f"epoch {epoch:3d} train {train_loss:.5f} val {validation['total']:.5f} "
                f"recall {validation.get('active_recall', 0.0):.3f} "
                f"w_phys {loss_fn.weights['physics']:.3f}"
            )

        if validation["total"] < result.best_val_loss - 1e-9:
            result.best_val_loss = validation["total"]
            result.best_epoch = epoch
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            since_improvement = 0
        else:
            since_improvement += 1
            if since_improvement >= config.patience:
                result.notes.append(f"early stopped at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if test_samples:
        result.test_metrics = evaluate(
            model, loss_fn, test_samples, batch_size=config.batch_size
        )

    payload = {
        "config": model_config.to_json(),
        "state_dict": model.state_dict(),
        "feature_stats": stats.to_json(),
        "train_config": config.to_json(),
        "history": result.history,
        "test_metrics": result.test_metrics,
        "dataset_id": manifest.dataset_id,
    }
    result.model_id = f"pignn-{manifest.dataset_id}-{config.seed}"
    result.checkpoint_key = f"models/{result.model_id}.pt"

    buffer = _serialise(payload)
    settings = StoreConfig.from_env()
    store = artifact_store if artifact_store is not None else open_artifact_store(settings)
    ref = store.put(result.checkpoint_key, buffer, content_type="application/octet-stream")
    store.put(
        f"models/{result.model_id}-metrics.json",
        json.dumps(result.summary(), indent=2, default=str).encode("utf-8"),
        content_type="application/json",
    )

    try:
        with ExperimentStore.open(store_path or settings.experiment_db) as experiments:
            experiments.record_model_version(
                architecture=model_config.to_json(),
                metrics=result.test_metrics or {"note": "no test split"},
                model_id=result.model_id,
                artifact_key=ref.key,
            )
    except Exception as error:  # noqa: BLE001 - the model is worth more than the log
        result.notes.append(f"model version not registered ({error})")

    result.wall_clock_s = time.perf_counter() - started
    return result


def _serialise(payload: dict[str, Any]) -> bytes:
    import io

    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def open_artifact_store_for(path: str | Path):
    """Accept either a directory or an already-open store."""
    from ..store import LocalArtifactStore
    from ..store.artifacts import ArtifactStore

    if isinstance(path, (str, Path)):
        return LocalArtifactStore(str(path))
    if isinstance(path, ArtifactStore):
        return path
    raise TypeError(f"cannot open a dataset from {type(path).__name__}")
