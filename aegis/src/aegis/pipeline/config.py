"""Pipeline configuration and result types."""

from __future__ import annotations

from dataclasses import dataclass

from ..constants import (
    DEFAULT_DV_BUDGET_KM_S,
    PC_TARGET_POST_MANEUVER,
    SCREENING_BOX_STARLINK_KM,
    SCREENING_STEP_S,
)
from ..core.maneuver import ManeuverPlan
from ..ingest.sources import Catalog
from ..risk.batch import AssessedCatalog

__all__ = ["PipelineConfig", "PipelineResult"]

#: Default screening window: 90 minutes. This is a fixed duration, not
#: 1.5 orbital periods of any particular object.
_DEFAULT_DURATION_S = 5400.0


@dataclass
class PipelineConfig:
    """Tunable parameters for one pipeline run.

    ``duration_s`` defaults to 5400 seconds (90 minutes). It is not 1.5
    orbital periods globally; callers who want a period-relative window
    must compute and pass that duration themselves.

    Starlink-specific numbers (inclination, constellation size) are not
    fields here and must not be hardcoded into the optimizer call -- only
    these config values are forwarded.
    """

    duration_s: float = _DEFAULT_DURATION_S
    step_s: float = SCREENING_STEP_S
    box_km: tuple[float, float, float] = SCREENING_BOX_STARLINK_KM
    target_pc: float = PC_TARGET_POST_MANEUVER
    dv_budget_km_s: float = DEFAULT_DV_BUDGET_KM_S
    max_iterations: int = 5
    max_objects: int | None = None


@dataclass
class PipelineResult:
    """Outcome of one ``run_pipeline`` call.

    Attributes
    ----------
    source
        ``CELESTRAK`` or ``SYNTHETIC``.
    catalog
        The catalog that was screened (after ``max_objects`` slicing).
    assessed
        Ranked conjunctions from the initial screen (before re-planning).
    plan
        Fleet maneuver plan. ``plan.summary()`` remains valid.
    covariance_source
        Provenance of the covariances used. The default TLE model is
        :data:`~aegis.core.state.CovarianceSource.SYNTHETIC_TLE`.
    config
        The config that produced this result.
    """

    source: str
    catalog: Catalog
    assessed: AssessedCatalog
    plan: ManeuverPlan
    covariance_source: str
    config: PipelineConfig