"""Covariance models for objects whose uncertainty is not published.

The central honesty problem of any TLE-based conjunction system
--------------------------------------------------------------
Two-line elements carry no uncertainty information whatsoever. A rigorous
probability of collision *requires* a covariance, so any Pc computed from
TLEs is a probability for a covariance that was assumed, not measured. NASA's
own conjunction assessment handbook is blunt about this: using TLEs for
conjunction assessment "is not recommended because the TLE accuracy is not
sufficient to perform the necessary conjunction assessment calculations."

This system does not pretend otherwise. Covariances produced here are tagged
:data:`~aegis.core.state.CovarianceSource.SYNTHETIC_TLE`, that tag propagates
into every downstream assessment and CDM, and the operations console displays
it. A synthetic covariance supports *ranking and triage* -- deciding which
conjunctions deserve attention -- and does not support an operational
maneuver decision on its own. For that you need CDMs from a conjunction
assessment provider or operator ephemeris.

The error model
---------------
Orbit uncertainty is profoundly anisotropic and the anisotropy grows with
propagation time. A semi-major-axis error produces a *bounded* radial error
but an along-track error growing secularly as ``(3/2) n da t``. Concretely, a
10 m semi-major-axis error at LEO produces roughly 1.4 km/day of along-track
drift while radial error stays at 10 m.

Published measurements of TLE fit consistency for LEO give roughly
0.10 km radial, 0.47 km along-track, 0.13 km cross-track *at epoch*
(Flohrer et al., AMOS 2008) -- note along-track is already 4x radial with
zero propagation. Growth is quadratic in propagation time; independent work
finds 1 km position error reached somewhere between 1.2 and 9.2 days
depending on element set quality, and along-track errors near 50 km by 15
days.

The model below is deliberately simple and deliberately conservative, with
every coefficient named and sourced so it can be argued with.
"""

from __future__ import annotations

import numpy as np

from ..core.objects import ObjectType, SpaceObject
from ..core.state import Covariance, CovarianceSource

__all__ = ["TleCovarianceModel", "default_covariance_model"]


class TleCovarianceModel:
    """Empirical covariance model for TLE-derived states.

    Parameters
    ----------
    sigma_radial_epoch_km, sigma_transverse_epoch_km, sigma_normal_epoch_km
        One-sigma uncertainty at element-set epoch.
    transverse_growth_km_per_day
        Along-track error growth rate. The dominant term by far.
    radial_growth_km_per_day, normal_growth_km_per_day
        Growth in the bounded directions -- small, but not zero, because
        drag mismodelling eventually couples into all three.
    debris_multiplier
        Debris and rocket bodies have poorly determined ballistic
        coefficients and are tracked less often, so their errors are
        substantially larger than a well-tracked active payload's.
    """

    def __init__(
        self,
        *,
        sigma_radial_epoch_km: float = 0.10,
        sigma_transverse_epoch_km: float = 0.47,
        sigma_normal_epoch_km: float = 0.13,
        transverse_growth_km_per_day: float = 2.0,
        radial_growth_km_per_day: float = 0.05,
        normal_growth_km_per_day: float = 0.05,
        debris_multiplier: float = 3.0,
    ) -> None:
        self.sigma_radial_epoch_km = sigma_radial_epoch_km
        self.sigma_transverse_epoch_km = sigma_transverse_epoch_km
        self.sigma_normal_epoch_km = sigma_normal_epoch_km
        self.transverse_growth_km_per_day = transverse_growth_km_per_day
        self.radial_growth_km_per_day = radial_growth_km_per_day
        self.normal_growth_km_per_day = normal_growth_km_per_day
        self.debris_multiplier = debris_multiplier

    def object_multiplier(self, obj: SpaceObject) -> float:
        """Scale factor reflecting how well this class of object is tracked."""
        if obj.object_type in (ObjectType.DEBRIS, ObjectType.UNKNOWN):
            return self.debris_multiplier
        if obj.object_type == ObjectType.ROCKET_BODY:
            return self.debris_multiplier * 0.7
        return 1.0

    def sigmas_rtn_km(
        self, obj: SpaceObject, propagation_days: float
    ) -> tuple[float, float, float]:
        """One-sigma RTN uncertainties after propagating for a given span.

        ``propagation_days`` is measured from the element set epoch, not from
        "now" -- an object whose TLE is already three days old and is being
        propagated two days forward carries five days of accumulated error.
        """
        span = max(propagation_days, 0.0)
        multiplier = self.object_multiplier(obj)

        radial = self.sigma_radial_epoch_km + self.radial_growth_km_per_day * span
        transverse = self.sigma_transverse_epoch_km + self.transverse_growth_km_per_day * span
        normal = self.sigma_normal_epoch_km + self.normal_growth_km_per_day * span

        return radial * multiplier, transverse * multiplier, normal * multiplier

    def covariance(self, obj: SpaceObject, propagation_days: float) -> Covariance:
        """Build a diagonal RTN covariance for one object.

        Diagonal rather than correlated: the true covariance has real
        radial/along-track correlation from the energy-phase coupling, but
        inventing a correlation coefficient would be false precision on top
        of an already-synthetic quantity. A diagonal model with honest
        magnitudes is more defensible than a correlated one with invented
        structure.
        """
        radial, transverse, normal = self.sigmas_rtn_km(obj, propagation_days)
        return Covariance.from_sigmas_rtn(
            radial, transverse, normal, source=CovarianceSource.SYNTHETIC_TLE
        )

    def covariance_at(
        self, obj: SpaceObject, epoch_age_days: float, lead_time_days: float
    ) -> Covariance:
        """Covariance at a future epoch, accounting for element set age."""
        return self.covariance(obj, epoch_age_days + lead_time_days)


def default_covariance_model() -> TleCovarianceModel:
    """The model used unless a caller supplies its own."""
    return TleCovarianceModel()


def scale_for_confidence(covariance: Covariance, factor: float) -> Covariance:
    """Inflate a covariance by an empirical realism factor.

    Operationally derived covariances are known to be optimistic and
    inflation is standard practice. Be aware of the direction of travel
    though: if an event already sits in the dilution regime, inflating the
    covariance *lowers* the computed probability. Inflation is therefore not
    automatically the conservative choice, which is why the assessor reports
    a dilution flag alongside every result.
    """
    return covariance.scaled(factor)
