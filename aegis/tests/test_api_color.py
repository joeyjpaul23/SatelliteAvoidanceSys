"""Pure color-band helper contract (ui_console).

Band from Pc; raise one step when covariance is SYNTHETIC_TLE and the
object has dilution or miss < 3σ. Does not go past ACT.
"""

from __future__ import annotations

import aegis.api.color as color_mod
from aegis.api.color import display_band

_SYNTHETIC_TLE = "SYNTHETIC_TLE"
_OTHER_SOURCE = "CALCULATED"
_FAR_MISS_KM = 30.0
_CLOSE_MISS_KM = 10.0
_SIGMA_MAJOR_KM = 5.0  # 3σ = 15 km


def test_display_band_pc_zero_is_clear() -> None:
    assert (
        display_band(
            0.0,
            covariance_source=_OTHER_SOURCE,
            dilution=False,
            miss_km=_FAR_MISS_KM,
            sigma_major_km=_SIGMA_MAJOR_KM,
        )
        == "CLEAR"
    )
    assert (
        display_band(
            0.0,
            covariance_source=_SYNTHETIC_TLE,
            dilution=False,
            miss_km=_FAR_MISS_KM,
            sigma_major_km=_SIGMA_MAJOR_KM,
        )
        == "CLEAR"
    )


def test_display_band_pc_1e_7_is_monitor() -> None:
    assert (
        display_band(
            1e-7,
            covariance_source=_OTHER_SOURCE,
            dilution=False,
            miss_km=_FAR_MISS_KM,
            sigma_major_km=_SIGMA_MAJOR_KM,
        )
        == "MONITOR"
    )


def test_display_band_pc_1e_5_is_watch() -> None:
    assert (
        display_band(
            1e-5,
            covariance_source=_OTHER_SOURCE,
            dilution=False,
            miss_km=_FAR_MISS_KM,
            sigma_major_km=_SIGMA_MAJOR_KM,
        )
        == "WATCH"
    )


def test_display_band_pc_1e_4_is_act() -> None:
    assert (
        display_band(
            1e-4,
            covariance_source=_OTHER_SOURCE,
            dilution=False,
            miss_km=_FAR_MISS_KM,
            sigma_major_km=_SIGMA_MAJOR_KM,
        )
        == "ACT"
    )


def test_display_band_clear_synthetic_tle_dilution_inflates_to_monitor() -> None:
    assert (
        display_band(
            0.0,
            covariance_source=_SYNTHETIC_TLE,
            dilution=True,
            miss_km=_FAR_MISS_KM,
            sigma_major_km=_SIGMA_MAJOR_KM,
        )
        == "MONITOR"
    )


def test_display_band_monitor_synthetic_tle_close_miss_inflates_to_watch() -> None:
    assert (
        display_band(
            1e-7,
            covariance_source=_SYNTHETIC_TLE,
            dilution=False,
            miss_km=_CLOSE_MISS_KM,
            sigma_major_km=_SIGMA_MAJOR_KM,
        )
        == "WATCH"
    )


def test_display_band_watch_synthetic_tle_dilution_inflates_to_act() -> None:
    assert (
        display_band(
            1e-5,
            covariance_source=_SYNTHETIC_TLE,
            dilution=True,
            miss_km=_FAR_MISS_KM,
            sigma_major_km=_SIGMA_MAJOR_KM,
        )
        == "ACT"
    )


def test_display_band_act_synthetic_tle_dilution_stays_act() -> None:
    assert (
        display_band(
            1e-4,
            covariance_source=_SYNTHETIC_TLE,
            dilution=True,
            miss_km=_FAR_MISS_KM,
            sigma_major_km=_SIGMA_MAJOR_KM,
        )
        == "ACT"
    )


def test_display_band_synthetic_tle_no_dilution_far_miss_does_not_inflate() -> None:
    assert (
        display_band(
            0.0,
            covariance_source=_SYNTHETIC_TLE,
            dilution=False,
            miss_km=_FAR_MISS_KM,
            sigma_major_km=_SIGMA_MAJOR_KM,
        )
        == "CLEAR"
    )
    assert (
        display_band(
            1e-7,
            covariance_source=_SYNTHETIC_TLE,
            dilution=False,
            miss_km=_FAR_MISS_KM,
            sigma_major_km=_SIGMA_MAJOR_KM,
        )
        == "MONITOR"
    )
    assert (
        display_band(
            1e-5,
            covariance_source=_SYNTHETIC_TLE,
            dilution=False,
            miss_km=_FAR_MISS_KM,
            sigma_major_km=_SIGMA_MAJOR_KM,
        )
        == "WATCH"
    )
    assert (
        display_band(
            1e-4,
            covariance_source=_SYNTHETIC_TLE,
            dilution=False,
            miss_km=_FAR_MISS_KM,
            sigma_major_km=_SIGMA_MAJOR_KM,
        )
        == "ACT"
    )


def test_display_band_non_synthetic_tle_dilution_does_not_inflate() -> None:
    assert (
        display_band(
            0.0,
            covariance_source=_OTHER_SOURCE,
            dilution=True,
            miss_km=_CLOSE_MISS_KM,
            sigma_major_km=_SIGMA_MAJOR_KM,
        )
        == "CLEAR"
    )
    assert (
        display_band(
            1e-7,
            covariance_source=_OTHER_SOURCE,
            dilution=True,
            miss_km=_CLOSE_MISS_KM,
            sigma_major_km=_SIGMA_MAJOR_KM,
        )
        == "MONITOR"
    )
    assert (
        display_band(
            1e-5,
            covariance_source=_OTHER_SOURCE,
            dilution=True,
            miss_km=_CLOSE_MISS_KM,
            sigma_major_km=_SIGMA_MAJOR_KM,
        )
        == "WATCH"
    )


def test_optional_band_from_pc_thresholds() -> None:
    band_from_pc = getattr(color_mod, "band_from_pc", None)
    if band_from_pc is None:
        return
    assert band_from_pc(0.0) == "CLEAR"
    assert band_from_pc(1e-7) == "MONITOR"
    assert band_from_pc(1e-5) == "WATCH"
    assert band_from_pc(1e-4) == "ACT"
