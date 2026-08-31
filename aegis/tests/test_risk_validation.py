"""Independent Pc check contract (Step 12).

Writes against the public ``aegis.risk`` surface only.
Does not import ``aegis.risk.validation``.
"""

from __future__ import annotations

import math
from pathlib import Path

from aegis.risk import (
    REFERENCE_CASES,
    evaluate_reference_cases,
    max_collision_probability,
    write_validation_report,
)

_REQUIRED_CASE_NAMES = frozenset(
    {
        "center_small",
        "offset_isotropic",
        "anisotropic_chan_ok",
        "far_tail",
        "pc_max_unit",
        "pc_max_zero_miss",
    }
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMMITTED_REPORT = _REPO_ROOT / "docs" / "alfano-cara-check.md"


def _case_name(case: object) -> str:
    if isinstance(case, dict):
        return str(case["name"])
    return str(getattr(case, "name"))


def test_reference_cases_include_required_names() -> None:
    names = {_case_name(case) for case in REFERENCE_CASES}
    missing = _REQUIRED_CASE_NAMES - names
    assert not missing, f"missing required cases: {sorted(missing)}"


def test_evaluate_reference_cases_all_ok() -> None:
    results = evaluate_reference_cases()
    assert results, "evaluate_reference_cases returned no results"
    failed = [row["name"] for row in results if not row["ok"]]
    assert not failed, f"failed cases: {failed}"


def test_pc_max_unit_matches_published_formula() -> None:
    assert max_collision_probability(1.0, 10.0) == 1.0 / (math.e * 100.0)


def test_write_validation_report_writes_table(tmp_path: Path) -> None:
    dest = tmp_path / "validation.md"
    written = write_validation_report(dest)
    text = Path(written).read_text(encoding="utf-8")
    pipe_rows = [line for line in text.splitlines() if "|" in line]
    assert pipe_rows, "report is not a markdown table"
    assert any("---" in line for line in pipe_rows), "table has no header separator"


def test_committed_alfano_cara_check_exists() -> None:
    assert _COMMITTED_REPORT.is_file(), f"missing {_COMMITTED_REPORT}"
    text = _COMMITTED_REPORT.read_text(encoding="utf-8")
    assert "center_small" in text
    assert "pc_max" in text
