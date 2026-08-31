"""Collision probability and risk assessment.

Entry point is :func:`aegis.risk.assess`. The layer beneath it splits into:

``projection``
    Reduces a conjunction to the canonical 2D problem: an axis-aligned
    Gaussian and a disc. Shared by every probability method.
``alfano``
    The primary method -- Alfano's 2005 formulation evaluated with
    Gauss-Chebyshev quadrature. This is what Starlink's platform documents
    using, and what NASA CARA ships.
``chan``
    An independent analytic method used as a cross-check where its validity
    envelope permits.
"""

from .alfano import METHOD_NAME as ALFANO_METHOD
from .alfano import collision_probability, max_collision_probability
from .assessor import RiskAssessmentError, assess, assess_projection
from .batch import AssessedCatalog, RankedConjunction, assess_catalog
from .projection import ProjectedEncounter, project_encounter
from .validation import (
    REFERENCE_CASES,
    evaluate_reference_cases,
    write_validation_report,
)

__all__ = [
    "ALFANO_METHOD",
    "AssessedCatalog",
    "ProjectedEncounter",
    "REFERENCE_CASES",
    "RankedConjunction",
    "RiskAssessmentError",
    "assess",
    "assess_catalog",
    "assess_projection",
    "collision_probability",
    "evaluate_reference_cases",
    "max_collision_probability",
    "project_encounter",
    "write_validation_report",
]
