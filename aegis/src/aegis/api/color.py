"""Display-band helpers for the operations console.

Bands match :meth:`aegis.core.conjunction.RiskLevel.from_probability`.
Inflation is a display rule for TLE-grade covariance, not a new risk model.
"""

from __future__ import annotations

from ..core.conjunction import RiskLevel
from ..core.state import CovarianceSource

__all__ = [
    "BANDS",
    "BAND_COLORS",
    "band_from_pc",
    "inflate_band",
    "display_band",
]

BANDS = ("CLEAR", "MONITOR", "WATCH", "ACT")

BAND_COLORS = {
    "CLEAR": "#3DFF8A",
    "MONITOR": "#E8C547",
    "WATCH": "#E07A3D",
    "ACT": "#E23B2F",
}


def band_from_pc(probability: float) -> str:
    """Same thresholds as RiskLevel.from_probability / aegis.constants:
    ACT >= 1e-4, WATCH >= 1e-5, MONITOR >= 1e-7, else CLEAR.
    """
    return RiskLevel.from_probability(probability)


def inflate_band(band: str) -> str:
    """Raise one step. ACT stays ACT."""
    try:
        index = BANDS.index(band)
    except ValueError:
        return band
    return BANDS[min(index + 1, len(BANDS) - 1)]


def display_band(
    probability: float,
    *,
    covariance_source: str,
    dilution: bool,
    miss_km: float,
    sigma_major_km: float,
) -> str:
    """Band from Pc; +1 if covariance_source == SYNTHETIC_TLE
    and (dilution or miss_km < 3 * sigma_major_km).
    """
    band = band_from_pc(probability)
    if covariance_source != CovarianceSource.SYNTHETIC_TLE:
        return band
    raise_one = bool(dilution)
    if sigma_major_km > 0.0 and miss_km < 3.0 * sigma_major_km:
        raise_one = True
    return inflate_band(band) if raise_one else band
