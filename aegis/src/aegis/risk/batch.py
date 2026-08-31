"""Catalog-level risk ranking.

Turns a list of geometric conjunctions into a ranked catalog of collision
probabilities. Each event is assessed with the existing two-body assessor
and a TLE covariance model -- the same synthetic uncertainty that
propagation already tags as :data:`~aegis.core.state.CovarianceSource.SYNTHETIC_TLE`.
That tag is preserved here. Ranking is for triage, not for an operational
maneuver decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.conjunction import Conjunction, RiskAssessment, RiskLevel
from ..core.objects import SpaceObject
from ..core.state import CovarianceSource
from ..core.timebase import seconds_between
from ..ingest import MixedDataSourceError
from ..propagation.covariance import TleCovarianceModel, default_covariance_model
from .assessor import assess

__all__ = ["RankedConjunction", "AssessedCatalog", "assess_catalog"]


@dataclass
class RankedConjunction:
    """One assessed conjunction and its position in the catalog ranking."""

    conjunction: Conjunction
    assessment: RiskAssessment
    rank: int


@dataclass
class AssessedCatalog:
    """Homogeneous, ranked set of assessed conjunctions.

    Attributes
    ----------
    source
        The shared ``data_source`` of every object in the input
        (``CELESTRAK`` or ``SYNTHETIC``). Empty string when there were no
        conjunctions.
    entries
        Ranked results, highest collision probability first. Ties break on
        ``conjunction_id``.
    covariance_source
        Provenance of the covariances used. The default TLE model always
        produces :data:`~aegis.core.state.CovarianceSource.SYNTHETIC_TLE`;
        that value is recorded here and is never relabelled as calculated.
    """

    source: str
    entries: list[RankedConjunction] = field(default_factory=list)
    covariance_source: str = CovarianceSource.SYNTHETIC_TLE

    def above(self, level: str) -> list[RankedConjunction]:
        """Entries at or above a named :class:`~aegis.core.conjunction.RiskLevel`."""
        threshold = RiskLevel.rank(level)
        return [
            entry
            for entry in self.entries
            if RiskLevel.rank(entry.assessment.risk_level) >= threshold
        ]

    def by_id(self, conjunction_id: str) -> RankedConjunction:
        """Return the ranked entry for ``conjunction_id``."""
        for entry in self.entries:
            if entry.conjunction.conjunction_id == conjunction_id:
                return entry
        raise KeyError(conjunction_id)


def _reject_mixed_sources(objects: list[SpaceObject]) -> None:
    if not objects:
        return
    sources = {obj.data_source for obj in objects}
    if len(sources) > 1:
        raise MixedDataSourceError(
            "cannot assess objects from mixed data sources: "
            + ", ".join(sorted(repr(s) for s in sources))
        )


def _propagation_days(obj: SpaceObject, conjunction: Conjunction) -> float:
    if obj.elements is None:
        return 0.0
    return seconds_between(obj.elements.epoch, conjunction.tca) / 86400.0


def assess_catalog(
    conjunctions: list[Conjunction],
    *,
    covariance_model: TleCovarianceModel | None = None,
    cross_check: bool = True,
    objects: list[SpaceObject] | None = None,
) -> AssessedCatalog:
    """Assess and rank a list of conjunctions.

    Parameters
    ----------
    conjunctions
        Geometric close approaches. Mixed primary/secondary ``data_source``
        values across the list raise
        :class:`~aegis.ingest.MixedDataSourceError`.
    covariance_model
        TLE uncertainty model. Defaults to
        :func:`~aegis.propagation.covariance.default_covariance_model`.
        Covariances it produces stay tagged
        :data:`~aegis.core.state.CovarianceSource.SYNTHETIC_TLE`.
    cross_check
        Forwarded to :func:`~aegis.risk.assessor.assess`.
    objects
        Optional catalog objects. If supplied, mixed ``data_source`` values
        raise :class:`~aegis.ingest.MixedDataSourceError`.
    """
    if objects is not None:
        _reject_mixed_sources(objects)

    if not conjunctions:
        return AssessedCatalog(source="", entries=[])

    pair_objects = [
        obj
        for conjunction in conjunctions
        for obj in (conjunction.primary, conjunction.secondary)
    ]
    _reject_mixed_sources(pair_objects)

    model = covariance_model if covariance_model is not None else default_covariance_model()

    assessed: list[tuple[Conjunction, RiskAssessment]] = []
    for conjunction in conjunctions:
        primary_covariance = model.covariance(
            conjunction.primary,
            _propagation_days(conjunction.primary, conjunction),
        )
        secondary_covariance = model.covariance(
            conjunction.secondary,
            _propagation_days(conjunction.secondary, conjunction),
        )
        assessment = assess(
            conjunction,
            primary_covariance,
            secondary_covariance,
            cross_check=cross_check,
        )
        assessed.append((conjunction, assessment))

    assessed.sort(
        key=lambda item: (-item[1].probability, item[0].conjunction_id)
    )
    entries = [
        RankedConjunction(conjunction=conjunction, assessment=assessment, rank=rank)
        for rank, (conjunction, assessment) in enumerate(assessed, start=1)
    ]
    return AssessedCatalog(
        source=pair_objects[0].data_source,
        entries=entries,
        covariance_source=CovarianceSource.SYNTHETIC_TLE,
    )
