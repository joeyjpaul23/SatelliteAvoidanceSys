"""Named collections of ``(family, seed)`` pairs for repeatable test runs.

Three sizes trade coverage for wall-clock time: ``smoke`` is meant to run on
every commit, ``standard`` is the default local/CI regression sweep, and
``full`` is the larger sweep used to build the safety-premium distribution
described in step14 section 0. All three are pure data -- ``expand_suite``
is the only function here that actually calls the generator.
"""

from __future__ import annotations

from ..ingest.synthetic import SyntheticAuthorization
from .generator import FAMILIES, generate
from .spec import Scenario

__all__ = ["SCENARIO_SUITES", "expand_suite"]


def _pairs(seeds: range) -> tuple[tuple[str, int], ...]:
    return tuple((family, seed) for family in FAMILIES for seed in seeds)


#: One scenario per family -- fast enough to run on every commit.
_SMOKE_PAIRS = _pairs(range(1))
#: Seven seeds per family (9 families * 7 = 63) -- the default local/CI sweep.
_STANDARD_PAIRS = _pairs(range(7))
#: Forty-four seeds per family (9 families * 44 = 396) -- the full safety
#: -premium distribution sweep.
_FULL_PAIRS = _pairs(range(44))

SCENARIO_SUITES: dict[str, tuple[tuple[str, int], ...]] = {
    "smoke": _SMOKE_PAIRS,
    "standard": _STANDARD_PAIRS,
    "full": _FULL_PAIRS,
}


def expand_suite(name: str, *, authorization: SyntheticAuthorization | None = None) -> list[Scenario]:
    """Materialise a named suite into concrete, generated scenarios.

    ``authorization`` is forwarded to every synthetic family's
    :func:`~aegis.scenarios.generator.generate` call; ``replay-tle`` entries
    ignore it since that family needs none. Missing authorization surfaces
    as :class:`~aegis.ingest.sources.SyntheticNotAuthorizedError` raised by
    ``generate`` itself -- this function adds no fallback of its own.
    """
    if name not in SCENARIO_SUITES:
        raise KeyError(f"unknown scenario suite: {name!r}; choices are {tuple(SCENARIO_SUITES)}")
    return [generate(family, seed, authorization=authorization) for family, seed in SCENARIO_SUITES[name]]
