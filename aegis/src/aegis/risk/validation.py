"""Independent reference-case check of the production collision-probability methods.

Alfano Gauss-Chebyshev is the operational estimator. This module compares it
to Chan's analytic method where that envelope is valid, to Alfano Simpson
quadrature otherwise, and to the published geometric ``Pc_max`` formula.

NASA CARA MATLAB is intentionally not a dependency. The referees here are the
closed-form and alternate-quadrature paths already in this package.
"""

from __future__ import annotations

import math
from pathlib import Path

from . import alfano, chan

__all__ = [
    "REFERENCE_CASES",
    "evaluate_reference_cases",
    "write_validation_report",
]

KIND_ALFANO_CHAN = "alfano_chan"
KIND_PC_MAX = "pc_max"

# Alfano GC vs Chan: 1% relative or 1e-8 absolute, whichever is looser.
_CHAN_REL_TOL = 1e-2
_CHAN_ABS_TOL = 1e-8
# Alfano GC vs Simpson when Chan is outside its envelope.
_SIMPSON_REL_TOL = 0.05

_PC_MAX_CITATION = (
    "Alfano, Relating Position Uncertainty to Maximum Conjunction "
    "Probability, JAS 53(2), 2005"
)

REFERENCE_CASES: tuple[dict, ...] = (
    {
        "name": "center_small",
        "source": "Chan isotropic analytic (same inputs); Alfano 2005 GC",
        "sigma_major": 1.0,
        "sigma_minor": 1.0,
        "miss_x": 0.0,
        "miss_z": 0.0,
        "hard_body_radius": 0.01,
        "kind": KIND_ALFANO_CHAN,
    },
    {
        "name": "offset_isotropic",
        "source": "Chan isotropic analytic (same inputs); Alfano 2005 GC",
        "sigma_major": 1.0,
        "sigma_minor": 1.0,
        "miss_x": 1.0,
        "miss_z": 0.0,
        "hard_body_radius": 0.02,
        "kind": KIND_ALFANO_CHAN,
    },
    {
        "name": "anisotropic_chan_ok",
        "source": "Chan 1997 / 2008 when the envelope allows; else Alfano Simpson",
        "sigma_major": 2.0,
        "sigma_minor": 1.0,
        "miss_x": 0.5,
        "miss_z": 0.0,
        "hard_body_radius": 0.01,
        "kind": KIND_ALFANO_CHAN,
    },
    {
        "name": "far_tail",
        "source": "Far-tail; Alfano vs Chan or Simpson",
        "sigma_major": 0.1,
        "sigma_minor": 0.1,
        "miss_x": 2.0,
        "miss_z": 0.0,
        "hard_body_radius": 0.005,
        "kind": KIND_ALFANO_CHAN,
    },
    {
        "name": "pc_max_unit",
        "source": _PC_MAX_CITATION,
        "sigma_major": 1.0,
        "sigma_minor": 1.0,
        "miss_x": 10.0,
        "miss_z": 0.0,
        "hard_body_radius": 1.0,
        "miss_distance": 10.0,
        "kind": KIND_PC_MAX,
    },
    {
        "name": "pc_max_zero_miss",
        "source": _PC_MAX_CITATION,
        "sigma_major": 1.0,
        "sigma_minor": 1.0,
        "miss_x": 0.0,
        "miss_z": 0.0,
        "hard_body_radius": 1.0,
        "miss_distance": 0.0,
        "kind": KIND_PC_MAX,
    },
)


def _relative_error(value: float, referee: float) -> float:
    if referee == 0.0:
        return 0.0 if value == 0.0 else math.inf
    return abs(value - referee) / abs(referee)


def _agrees_chan_tolerance(value: float, referee: float) -> bool:
    """True if relative 1e-2 or absolute 1e-8 holds (whichever is looser)."""
    return abs(value - referee) <= _CHAN_ABS_TOL or _relative_error(
        value, referee
    ) <= _CHAN_REL_TOL


def _is_open_unit_interval(value: float) -> bool:
    return math.isfinite(value) and 0.0 < value <= 1.0


def _evaluate_alfano_chan(case: dict) -> dict:
    args = (
        case["sigma_major"],
        case["sigma_minor"],
        case["miss_x"],
        case["miss_z"],
        case["hard_body_radius"],
    )
    alfano_pc = alfano.collision_probability(*args)

    referee_pc = None
    if chan.is_applicable(case["sigma_major"], case["sigma_minor"], case["hard_body_radius"]):
        chan_pc = chan.collision_probability(*args)
        # Chan's ncx2 path underflows to 0 in the far tail; that is not a
        # usable referee even when the scaled-radius envelope says applicable.
        if _is_open_unit_interval(chan_pc):
            referee_pc = chan_pc
            ok = _agrees_chan_tolerance(alfano_pc, referee_pc)

    if referee_pc is None:
        referee_pc = alfano.collision_probability_simpson(*args)
        ok = _relative_error(alfano_pc, referee_pc) <= _SIMPSON_REL_TOL

    if case["name"] == "far_tail":
        ok = ok and _is_open_unit_interval(alfano_pc) and _is_open_unit_interval(
            referee_pc
        )

    return {
        "name": case["name"],
        "source": case["source"],
        "alfano_pc": alfano_pc,
        "referee_pc": referee_pc,
        "ok": ok,
        "rel_error": _relative_error(alfano_pc, referee_pc),
    }


def _evaluate_pc_max(case: dict) -> dict:
    hard_body_radius = case["hard_body_radius"]
    miss_distance = case["miss_distance"]
    pc_max = alfano.max_collision_probability(hard_body_radius, miss_distance)

    if miss_distance <= 0.0:
        referee_pc = 1.0
        ok = pc_max == 1.0
    else:
        referee_pc = min(
            (hard_body_radius**2) / (math.e * miss_distance**2),
            1.0,
        )
        ok = _agrees_chan_tolerance(pc_max, referee_pc)

    return {
        "name": case["name"],
        "source": case["source"],
        "pc_max": pc_max,
        "referee_pc": referee_pc,
        "ok": ok,
        "rel_error": _relative_error(pc_max, referee_pc),
    }


def evaluate_reference_cases() -> list[dict]:
    """Run every reference case and return one result dict per case."""
    results: list[dict] = []
    for case in REFERENCE_CASES:
        if case["kind"] == KIND_PC_MAX:
            results.append(_evaluate_pc_max(case))
        else:
            results.append(_evaluate_alfano_chan(case))
    return results


def _format_number(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if 1e-3 <= magnitude < 1e4:
        return f"{value:.6g}"
    return f"{value:.6e}"


def write_validation_report(path: str | Path) -> Path:
    """Write the reference-case comparison as a markdown table.

    Parameters
    ----------
    path
        Destination markdown file.

    Returns
    -------
    Path
        The path written.
    """
    destination = Path(path)
    results = evaluate_reference_cases()

    lines = [
        "# Alfano / CARA independent Pc check",
        "",
        "Independent reference-case check of the production Alfano",
        "Gauss-Chebyshev collision probability against Chan's analytic method",
        "(Chan 1997 / 2008, where the envelope allows), Alfano Simpson",
        "quadrature, and the published geometric Pc_max formula.",
        "",
        "NASA CARA MATLAB is not a dependency. These cases are the in-repo",
        "cross-check against independent closed-form and quadrature referees.",
        "",
        "| name | citation | Alfano (or pc_max) | referee | rel error | pass/fail |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for row in results:
        value = row["alfano_pc"] if "alfano_pc" in row else row["pc_max"]
        status = "pass" if row["ok"] else "fail"
        lines.append(
            "| {name} | {source} | {value} | {referee} | {rel_error} | {status} |".format(
                name=row["name"],
                source=row["source"],
                value=_format_number(value),
                referee=_format_number(row["referee_pc"]),
                rel_error=_format_number(row["rel_error"]),
                status=status,
            )
        )

    lines.append("")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination
