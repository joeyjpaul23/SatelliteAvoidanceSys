"""CCSDS message formats: CDM in, OEM out.

Offline KVN only. ``read_cdm`` never calls ``generate_synthetic``.
CDM-ingested objects are labeled ``CELESTRAK`` with ``metadata["origin"]="CDM"``
so they do not invent a third catalog source or cross the synthetic wall.
"""

from .cdm import read_cdm
from .errors import CcsdsError
from .oem import write_oem

__all__ = [
    "CcsdsError",
    "read_cdm",
    "write_oem",
]
