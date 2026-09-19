"""Close-approach detection.

``prefilter_pairs``
    Drop pairs whose apogee / perigee bands cannot meet.
``broadphase``
    Coarse RTN-box plus no-miss sieve over a propagation grid.
``broadphase_partitioned``
    Same gate, pairs from a conservative spatial hash (no false negatives).
``refine_tca``
    Bracket and linearly refine time of closest approach for one pair.
``screen`` / ``screen_table``
    Catalog-level pipeline: streaming sweep, batched TCA refinement, box
    acceptance; objects for small catalogs, columns at catalog scale.
    ``partitioned=None`` auto-enables the hash when N is at least
    :data:`~aegis.constants.SCREENING_PARTITION_MIN_OBJECTS`.
"""

from .broadphase import broadphase, broadphase_partitioned
from .engine import screen, screen_table
from .errors import ScreeningError
from .prefilter import prefilter_pairs
from .results import ConjunctionTable
from .tca import refine_tca

__all__ = [
    "screen",
    "screen_table",
    "ConjunctionTable",
    "prefilter_pairs",
    "broadphase",
    "broadphase_partitioned",
    "refine_tca",
    "ScreeningError",
]
