"""Map a post-maneuver Pc target to a required miss distance."""

from __future__ import annotations

from ..constants import PC_TARGET_POST_MANEUVER
from ..risk.alfano import collision_probability

__all__ = ["required_miss_distance_km"]

#: If Alfano is still above the target at this miss, return the sentinel.
_SENTINEL_MISS_KM = 100.0
_SEARCH_ITERS = 80


def required_miss_distance_km(
    hard_body_radius_km: float,
    sigma_major_km: float,
    sigma_minor_km: float,
    target_pc: float = PC_TARGET_POST_MANEUVER,
) -> float:
    """Smallest major-axis miss with Alfano Pc at or below ``target_pc``.

    The miss is the Alfano ``miss_x`` component with ``miss_z = 0``. The
    result is monotonically non-decreasing as ``target_pc`` decreases. If
    even a 100 km miss cannot reach the target, that sentinel is returned
    and no exception is raised.
    """
    hbr = float(hard_body_radius_km)
    sigma_major = float(sigma_major_km)
    sigma_minor = float(sigma_minor_km)
    target = float(target_pc)

    if hbr <= 0.0 or target >= 1.0:
        return 0.0
    if target <= 0.0:
        return _SENTINEL_MISS_KM

    def _pc(miss_x: float) -> float:
        return collision_probability(
            sigma_major, sigma_minor, miss_x, 0.0, hbr
        )

    if _pc(0.0) <= target:
        return 0.0
    if _pc(_SENTINEL_MISS_KM) > target:
        return _SENTINEL_MISS_KM
    if hbr >= 3.0 * sigma_major and target < 1e-16:
        return _SENTINEL_MISS_KM

    lo = 0.0
    hi = _SENTINEL_MISS_KM
    for _ in range(_SEARCH_ITERS):
        mid = 0.5 * (lo + hi)
        if _pc(mid) <= target:
            hi = mid
        else:
            lo = mid
    return hi
