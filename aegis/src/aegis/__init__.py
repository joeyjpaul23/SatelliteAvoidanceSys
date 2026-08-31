"""AEGIS -- constellation conjunction assessment and collision avoidance.

A pipeline that ingests orbital data for a satellite fleet, screens it for
close approaches (including the intra-fleet ones no public service will
compute), assesses collision probability, and solves for the fuel-optimal set
of trajectory adjustments across the whole fleet at once.

Layer map, outermost first::

    api/          REST surface and the operations console it serves
    pipeline/     end-to-end orchestration
    maneuver/     fleet-wide maneuver optimisation
    risk/         collision probability
    screening/    close-approach detection
    propagation/  orbit propagation
    ingest/       data acquisition
    ccsds/        standard message formats
    core/         domain types
    constants     every tunable number in one place

Dependencies point strictly downward. ``core`` imports only ``constants``;
nothing imports ``api``.
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
