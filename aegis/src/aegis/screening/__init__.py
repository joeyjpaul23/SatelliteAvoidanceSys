"""Close-approach detection.

``screen`` / ``screen_table``
    Catalog-level pipeline: streaming sweep, batched TCA refinement, box
    acceptance; objects for small catalogs, columns at catalog scale.
    ``partitioned=None`` generates candidate pairs from a k-d tree when N is
    at least :data:`~aegis.constants.SCREENING_PARTITION_MIN_OBJECTS`.
``refine_tca``
    Bracket and linearly refine time of closest approach for one pair.
"""

from .engine import screen, screen_table
from .errors import ScreeningError
from .results import ConjunctionTable
from .tca import refine_tca

__all__ = [
    "ConjunctionTable",
    "ScreeningError",
    "refine_tca",
    "screen",
    "screen_table",
]
