"""Orbit propagation and uncertainty modelling.

``propagator``
    Vectorised SGP4 across a shared time grid. Objects are initialised once
    and reused across time blocks.
``covariance``
    Synthetic covariance for TLE-derived states. TLEs carry no uncertainty
    information, so any covariance here is assumed rather than measured --
    every product of this module is tagged accordingly so the limitation
    stays visible downstream.
"""

from .covariance import TleCovarianceModel, default_covariance_model
from .propagator import PropagationError, PropagationGrid, Sgp4Propagator

__all__ = [
    "PropagationError",
    "PropagationGrid",
    "Sgp4Propagator",
    "TleCovarianceModel",
    "default_covariance_model",
]
