"""CCSDS CDM/OEM contract (Step 11).

Writes against the public ``aegis.ccsds`` and ``aegis.pipeline`` surfaces
plus allowed imports (core types, ingest, risk). Does not import
``aegis.ccsds`` or ``aegis.pipeline`` submodules. Offline only: no HTTP;
``read_cdm`` must not call ``generate_synthetic``.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from aegis.ccsds import CcsdsError, read_cdm, write_oem
from aegis.constants import MU_EARTH_KM3_S2, R_EARTH_KM, REV_PER_DAY_TO_RAD_PER_S
from aegis.core.conjunction import Conjunction
from aegis.core.maneuver import Maneuver, ManeuverPlan, SatelliteManeuverSet
from aegis.core.objects import ObjectType, OrbitalElements, SpaceObject
from aegis.core.state import CovarianceSource, StateVector
from aegis.core.timebase import parse_epoch
from aegis.ingest import Catalog, DataSource
from aegis.pipeline import (
    PipelineConfig,
    PipelineResult,
    write_plan_oems,
)
import aegis.pipeline as pipeline_pkg
from aegis.risk import AssessedCatalog

_EPOCH = datetime(2010, 1, 1, tzinfo=timezone.utc)
_TCA = "2010-01-01T12:00:00.000"
_MISS_KM = 1.5
_REL_SPEED_KM_S = 0.25
_PRIMARY_ID = "11111"
_PRIMARY_NAME = "PRIMARY-SAT"
_SECONDARY_ID = "22222"
_SECONDARY_NAME = "SECONDARY-SAT"

_MINIMAL_CDM = f"""\
CCSDS_CDM_VERS = 1.0
TCA = {_TCA}
MISS_DISTANCE = {_MISS_KM}
RELATIVE_SPEED = {_REL_SPEED_KM_S}
OBJECT1_NAME = {_PRIMARY_NAME}
OBJECT1_ID = {_PRIMARY_ID}
OBJECT1_X = 7000.0
OBJECT1_Y = 0.0
OBJECT1_Z = 0.0
OBJECT1_X_DOT = 0.0
OBJECT1_Y_DOT = 7.5
OBJECT1_Z_DOT = 0.0
OBJECT2_NAME = {_SECONDARY_NAME}
OBJECT2_ID = {_SECONDARY_ID}
OBJECT2_X = 7001.5
OBJECT2_Y = 0.0
OBJECT2_Z = 0.0
OBJECT2_X_DOT = 0.0
OBJECT2_Y_DOT = 7.5
OBJECT2_Z_DOT = 0.25
"""

_CDM_WITHOUT_TCA = """\
CCSDS_CDM_VERS = 1.0
MISS_DISTANCE = 1.5
RELATIVE_SPEED = 0.25
OBJECT1_NAME = PRIMARY-SAT
OBJECT1_ID = 11111
OBJECT1_X = 7000.0
OBJECT1_Y = 0.0
OBJECT1_Z = 0.0
OBJECT1_X_DOT = 0.0
OBJECT1_Y_DOT = 7.5
OBJECT1_Z_DOT = 0.0
OBJECT2_NAME = SECONDARY-SAT
OBJECT2_ID = 22222
OBJECT2_X = 7001.5
OBJECT2_Y = 0.0
OBJECT2_Z = 0.0
OBJECT2_X_DOT = 0.0
OBJECT2_Y_DOT = 7.5
OBJECT2_Z_DOT = 0.25
"""


def _mean_motion_rev_per_day(altitude_km: float) -> float:
    semi_major_km = R_EARTH_KM + altitude_km
    mean_motion_rad_s = math.sqrt(MU_EARTH_KM3_S2 / semi_major_km**3)
    return mean_motion_rad_s / REV_PER_DAY_TO_RAD_PER_S


def _circular(object_id: str, *, name: str | None = None) -> SpaceObject:
    return SpaceObject(
        object_id=object_id,
        name=name or f"SAT-{object_id}",
        object_type=ObjectType.PAYLOAD,
        elements=OrbitalElements(
            epoch=_EPOCH,
            mean_motion_rev_per_day=_mean_motion_rev_per_day(550.0),
            eccentricity=0.0,
            inclination_deg=53.0,
            raan_deg=0.0,
            arg_perigee_deg=0.0,
            mean_anomaly_deg=0.0,
        ),
        data_source=DataSource.SYNTHETIC,
    )


def _state(
    epoch: datetime,
    position_km: list[float],
    velocity_km_s: list[float],
) -> StateVector:
    return StateVector(
        epoch=epoch,
        position_km=np.asarray(position_km, dtype=float),
        velocity_km_s=np.asarray(velocity_km_s, dtype=float),
        frame="TEME",
    )


def _fail_generate_synthetic(*_a, **_k):
    raise AssertionError("read_cdm must not call generate_synthetic")


def _result_with_one_burn() -> PipelineResult:
    sat = _circular("SAT-A", name="MANEUVER-SAT")
    burn = Maneuver(
        satellite_id=sat.object_id,
        epoch=_EPOCH,
        delta_v_rtn_km_s=[0.0, 1.0e-6, 0.0],
    )
    plan = ManeuverPlan(
        plan_id="ccsds-oem-plan",
        generated_at=datetime(2020, 6, 1, tzinfo=timezone.utc),
        satellite_sets={
            sat.object_id: SatelliteManeuverSet(
                satellite_id=sat.object_id,
                maneuvers=[burn],
            ),
            "SAT-IDLE": SatelliteManeuverSet(satellite_id="SAT-IDLE", maneuvers=[]),
        },
    )
    catalog = Catalog(
        source=DataSource.SYNTHETIC,
        objects=[sat, _circular("SAT-IDLE")],
        fetched_at=datetime.now(timezone.utc),
        query="ccsds-oem",
    )
    return PipelineResult(
        source=DataSource.SYNTHETIC,
        catalog=catalog,
        assessed=AssessedCatalog(source=DataSource.SYNTHETIC, entries=[]),
        plan=plan,
        covariance_source=CovarianceSource.SYNTHETIC_TLE,
        config=PipelineConfig(duration_s=600.0, step_s=60.0, max_iterations=1),
    )


def test_ccsds_public_imports() -> None:
    import aegis.ccsds as ccsds_pkg

    assert {"CcsdsError", "read_cdm", "write_oem"} <= set(ccsds_pkg.__all__)
    assert issubclass(CcsdsError, Exception)
    assert callable(read_cdm)
    assert callable(write_oem)
    assert "write_plan_oems" in pipeline_pkg.__all__
    assert callable(write_plan_oems)


def test_read_cdm_minimal_kvn_returns_conjunction() -> None:
    conjunction = read_cdm(_MINIMAL_CDM)

    assert isinstance(conjunction, Conjunction)
    assert conjunction.tca == parse_epoch(_TCA)
    assert conjunction.miss_distance_km == pytest.approx(_MISS_KM)
    assert conjunction.relative_speed_km_s == pytest.approx(_REL_SPEED_KM_S)

    assert conjunction.primary.object_id == _PRIMARY_ID
    assert conjunction.primary.name == _PRIMARY_NAME
    assert conjunction.secondary.object_id == _SECONDARY_ID
    assert conjunction.secondary.name == _SECONDARY_NAME

    assert conjunction.primary.data_source == DataSource.CELESTRAK
    assert conjunction.secondary.data_source == DataSource.CELESTRAK
    assert conjunction.primary.data_source != DataSource.SYNTHETIC
    assert conjunction.secondary.data_source != DataSource.SYNTHETIC
    assert conjunction.primary.metadata.get("origin") == "CDM"
    assert conjunction.secondary.metadata.get("origin") == "CDM"

    np.testing.assert_allclose(
        conjunction.primary_state.position_km, [7000.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        conjunction.primary_state.velocity_km_s, [0.0, 7.5, 0.0]
    )
    np.testing.assert_allclose(
        conjunction.secondary_state.position_km, [7001.5, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        conjunction.secondary_state.velocity_km_s, [0.0, 7.5, 0.25]
    )
    assert conjunction.primary_state.frame == "TEME"
    assert conjunction.secondary_state.frame == "TEME"


def test_read_cdm_missing_tca_raises() -> None:
    with pytest.raises(CcsdsError):
        read_cdm(_CDM_WITHOUT_TCA)


def test_write_oem_requires_state_and_contains_headers() -> None:
    obj = SpaceObject(object_id="25544", name="ISS", data_source=DataSource.CELESTRAK)
    with pytest.raises(CcsdsError):
        write_oem(obj, [])

    state = _state(_EPOCH, [7000.0, 100.0, -50.0], [0.1, 7.5, -0.2])
    text = write_oem(obj, [state])

    assert "CCSDS_OEM_VERS = 3.0" in text
    assert "OBJECT_NAME" in text
    assert "ISS" in text
    assert "TEME" in text
    assert "UTC" in text
    assert "2010-01-01T00:00:00" in text
    assert "7000" in text
    assert "7.5" in text


def test_write_plan_oems_writes_files_for_maneuvering_sats(tmp_path: Path) -> None:
    result = _result_with_one_burn()
    paths = write_plan_oems(result, tmp_path)

    written = {path.name: path for path in paths}
    assert "SAT-A.oem" in written
    assert written["SAT-A.oem"].is_file()
    body = written["SAT-A.oem"].read_text()
    assert "CCSDS_OEM_VERS = 3.0" in body
    assert "OBJECT_NAME" in body
    assert "TEME" in body
    assert "UTC" in body
    assert "SAT-IDLE.oem" not in written
    assert not (tmp_path / "SAT-IDLE.oem").exists()


def test_read_cdm_does_not_call_generate_synthetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aegis.ingest.generate_synthetic", _fail_generate_synthetic)
    monkeypatch.setattr(
        "aegis.ingest.synthetic.generate_synthetic", _fail_generate_synthetic
    )
    import aegis.ccsds as ccsds_pkg

    if hasattr(ccsds_pkg, "generate_synthetic"):
        monkeypatch.setattr(ccsds_pkg, "generate_synthetic", _fail_generate_synthetic)

    conjunction = read_cdm(_MINIMAL_CDM)
    assert isinstance(conjunction, Conjunction)
    assert conjunction.primary.data_source != DataSource.SYNTHETIC
    assert conjunction.secondary.data_source != DataSource.SYNTHETIC
