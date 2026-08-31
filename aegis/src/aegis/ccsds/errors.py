"""CCSDS message errors."""

from __future__ import annotations

__all__ = ["CcsdsError"]


class CcsdsError(Exception):
    """Raised when a CCSDS CDM or OEM cannot be parsed or written.

    Distinct from an empty screening result. A missing required KVN field,
    an unparseable epoch, or an OEM with no ephemeris states is a structural
    failure of the message, not a successful empty catalog.
    """
