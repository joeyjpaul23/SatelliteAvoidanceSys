"""Physical and operational constants.

Every magic number in AEGIS lives here. If you find a hard-coded constant
anywhere else in the codebase, that is a bug -- move it here and give it a
name and a source.

Units convention for the whole codebase
---------------------------------------
Distances are in KILOMETRES. Velocities are in KM/S. Times are in SECONDS.
The one deliberate exception is the CCSDS layer (``aegis.ccsds``), which must
speak the units the standard mandates (metres for covariance, km for state
vectors). Conversion happens at that boundary and nowhere else.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Earth and gravitational parameters (WGS-84, matching SGP4's internal model)
# ---------------------------------------------------------------------------

#: Earth gravitational parameter, km^3/s^2 (WGS-84).
MU_EARTH_KM3_S2 = 398600.4418

#: Earth equatorial radius, km (WGS-84). SGP4 uses this exact value.
R_EARTH_KM = 6378.137

#: Earth flattening (WGS-84).
EARTH_FLATTENING = 1.0 / 298.257223563

#: Second zonal harmonic. Drives nodal regression and the short-period radial
#: oscillation that forces padding in the perigee/apogee band check.
J2 = 1.08262668e-3

#: Standard gravity, m/s^2. Used only in the Tsiolkovsky rocket equation.
G0_M_S2 = 9.80665


# ---------------------------------------------------------------------------
# Screening volumes
# ---------------------------------------------------------------------------
# All screening volumes are RTN half-widths in kilometres, ordered
# (radial, transverse/along-track, normal/cross-track).
#
# The dramatic asymmetry -- along-track and cross-track being 20-100x the
# radial dimension -- is not arbitrary. Orbit uncertainty is profoundly
# anisotropic: a semi-major-axis error produces a BOUNDED radial error but an
# along-track error that grows secularly as (3/2) * n * delta_a * t. After a
# day or two the along-track 1-sigma is routinely 10-100x the radial. Radial
# separation is therefore both the most trustworthy and the most
# discriminating dimension, which is why it gets the tightest gate.

#: Starlink Space Safety Platform's published screening box: half-widths
#: 2 km radial, 44 km along-track, 51 km out of plane, a box (not an
#: ellipsoid) in each object's own RTN frame, checked from both objects.
#: SpaceX reports a close approach when the local minimum of range falls
#: inside it, then computes 2D Alfano Pc from the submitted covariances.
#: The box is a filter, not a risk measure.
#: Source: https://docs.space-safety.starlink.com/docs/tutorial-basics/architecture/
#: These are exactly the 18/19 SDS "Early Orbit" near-Earth screening volume
#: (Spaceflight Safety Handbook for Operators, Table 7: 2 x 44 x 51 km,
#: 7 days), the most generous standard near-Earth volume. For Starlink's own
#: altitude, 19 SDS's standard operator-ephemeris volume is 2 x 25 x 25 km
#: (Table 4). Measured on TLE data (2026-09-18): passes excluded by the
#: 2 km radial slab reach Pc 3.6e-7 at most, so the thin radial side is sound.
SCREENING_BOX_STARLINK_KM = (2.0, 44.0, 51.0)

#: 19 SDS LEO-1 regime (perigee <= 500 km, e < 0.25), ellipsoid half-axes.
#: Source: USSPACECOM Spaceflight Safety Handbook for Satellite Operators v1.7.
SCREENING_BOX_19SDS_LEO1_KM = (0.4, 44.0, 51.0)
SCREENING_BOX_19SDS_LEO2_KM = (0.4, 25.0, 25.0)   # perigee 500-750 km
SCREENING_BOX_19SDS_LEO3_KM = (0.4, 12.0, 12.0)   # perigee 750-1200 km
SCREENING_BOX_19SDS_LEO4_KM = (0.4, 2.0, 2.0)     # perigee 1200-2000 km

#: NASA CARA operational volume for LEO robotic assets near 700 km.
#: Source: NASA NTRS 20170007928 (screening volume sizing study).
SCREENING_BOX_CARA_LEO_KM = (0.5, 17.0, 20.0)

#: Volume sized for TLE-quality ephemeris rather than operator ephemeris.
#: Applying a 0.4 km radial gate to 7-day TLE propagation is not conservative,
#: it is meaningless -- TLE radial error alone is comparable and along-track
#: error is 20-40x larger. See docs/LIMITATIONS.md for the error model.
SCREENING_BOX_TLE_GRADE_KM = (5.0, 50.0, 50.0)


# ---------------------------------------------------------------------------
# Collision probability
# ---------------------------------------------------------------------------

#: Default hard-body radii by object type, metres. When the secondary's true
#: dimensions are unknown these are the 18th SDS / Space-Track conventions.
#: Radius is measured centre-of-mass to furthest extremity, so it includes
#: deployed solar arrays -- not the bus dimension.
HBR_DEFAULT_M = {
    "PAYLOAD": 5.0,
    "ROCKET BODY": 3.0,
    "DEBRIS": 1.0,
    "UNKNOWN": 3.0,
    "OTHER": 3.0,
}

#: Pc thresholds. Because Pc scales as HBR^2, the hard-body radius is a
#: first-order driver of every number downstream -- it is always logged
#: explicitly with each result rather than buried as a default.
PC_THRESHOLD_ASSESS = 1e-7   # NASA CARA: event enters detailed workflow
PC_THRESHOLD_WATCH = 1e-5    # common "yellow" / watch level
PC_THRESHOLD_ACT = 1e-4      # NASA CARA "red"; ESA avoidance trigger
PC_THRESHOLD_STARLINK = 3e-7  # Starlink's reported operational threshold
PC_TARGET_POST_MANEUVER = 1e-6  # ESA's post-maneuver residual risk target

#: Gauss-Chebyshev quadrature order for the Alfano integral. NASA CARA's
#: default, validated against 1.25e6 operational events. The integrand has an
#: infinite derivative at the disc edge, which Gauss-Chebyshev absorbs exactly;
#: 64 nodes reach machine precision where Simpson's rule needs thousands.
ALFANO_QUAD_ORDER = 64

#: Eigenvalue clipping floor for non-positive-definite 2x2 projected
#: covariances, as a fraction of the hard-body radius. Tying the floor to HBR
#: rather than an absolute epsilon keeps the remediation scale-free.
#: Source: NASA CARA RemediateCovariance2x2.m
COV_CLIP_FRACTIONS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


# ---------------------------------------------------------------------------
# Screening algorithm tuning
# ---------------------------------------------------------------------------

#: Coarse propagation step, seconds. The cost model is
#:     cost ~ A/dt + B*dt^2
#: because neighbour counts grow as the cube of the gate radius. 30 s sits
#: near the optimum for a ~10k object catalog: ~7 neighbours per object,
#: 20160 epochs over 7 days.
SCREENING_STEP_S = 30.0

#: Bounding relative speed for the no-miss gate, km/s. Two circular LEO orbits
#: meeting head-on give 15.4 km/s; 19 km/s covers eccentric objects near
#: perigee. This is what makes the coarse gate provably safe rather than
#: heuristic (see screening/sweep.py).
MAX_RELATIVE_SPEED_KM_S = 19.0

#: Padding for the perigee/apogee band check, km (screening sweep and the
#: ops catalog's debris trim). TLE elements are Brouwer mean
#: elements; the osculating radius oscillates about the mean by the J2
#: short-period term (~9-10 km at a = 7000 km) plus secular drift over a
#: multi-day window. Source: Woodburn, Coppola & Stoner, AAS 09-372.
PERIGEE_APOGEE_PAD_KM = 30.0

#: Below this relative speed the 2D Pc assumptions break down and the TCA
#: search can find spurious roots. Directly relevant to intra-constellation
#: screening, where co-planar satellites have very low relative velocity.
LOW_RELATIVE_VELOCITY_KM_S = 0.5

#: Encounter duration beyond which the short-encounter (2D) assumption is
#: suspect and the result should be flagged for 3D treatment, seconds.
MAX_SHORT_ENCOUNTER_DURATION_S = 500.0

#: Catalog size at which ``screen(..., partitioned=None)`` generates
#: candidate pairs from a k-d tree instead of all pairs. Both give the same
#: result.
SCREENING_PARTITION_MIN_OBJECTS = 50


# ---------------------------------------------------------------------------
# Maneuver planning
# ---------------------------------------------------------------------------

#: Specific impulse by propulsion type, seconds.
ISP_ARGON_HALL_S = 2500.0    # Starlink V2 Mini argon Hall thruster
ISP_KRYPTON_HALL_S = 1500.0  # Starlink V1.0 / V1.5 krypton Hall thruster
ISP_MONOPROP_S = 220.0       # conventional hydrazine, for comparison

#: Maximum impulsive delta-v per burn slot, km/s (0.5 m/s).
MAX_DV_PER_BURN_KM_S = 0.5e-3

#: Default per-satellite delta-v budget for a single planning cycle, km/s
#: (1 m/s). For scale: annual LEO station-keeping is 10-100 m/s, while a
#: typical collision avoidance maneuver is 1-10 cm/s. Fuel is emphatically
#: not the binding constraint -- a 5 cm/s maneuver on an 800 kg satellite at
#: Isp 2500 s costs about 1.6 grams of argon. What actually costs is service
#: disruption and the re-screening burden each maneuver imposes.
DEFAULT_DV_BUDGET_KM_S = 1.0e-3

#: Slack penalty weight in the fleet LP. Must dominate any realistic delta-v
#: so the optimizer only accepts an unresolved conjunction when it is
#: genuinely infeasible. Slack variables are MANDATORY, not optional: without
#: them the fleet LP is infeasible whenever any conjunction requires radial or
#: cross-track displacement, because those responses are bounded and periodic
#: while only the along-track response grows secularly.
LP_SLACK_PENALTY = 1e4

#: Minimum maneuver lead time, expressed in orbital periods. Below half an
#: orbit the periodic term in the Clohessy-Wiltshire along-track response has
#: not averaged out and the familiar 3*dv*dt rule is wrong by up to 85%.
#: The code always uses the exact (4*sin(nt) - 3nt)/n expression, but planning
#: burns at least one orbit ahead also buys fuel efficiency (cost scales as
#: 1/(3*n*dt)) and leaves time to re-screen the perturbed ephemeris.
MIN_LEAD_TIME_ORBITS = 1.0

#: Number of candidate burn slots offered to the optimizer per satellite,
#: spaced half an orbit apart working backwards from the earliest TCA.
DEFAULT_BURN_SLOTS = 6

#: Station-keeping box half-widths, km (in-track, cross-track). Representative
#: achievable LEO slot tolerances. Source: ISSFD 2012 autonomous LEO
#: station-keeping. A collision avoidance maneuver perturbs the semi-major
#: axis, which drifts the satellite out of its slot -- constrain it or you
#: trade a conjunction for a slot violation.
STATION_KEEPING_BOX_KM = (2.0, 1.0)


# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------

#: CelesTrak refreshes GP data every 2 hours; polling faster gains nothing and
#: risks the firewall. Their stated policy: more than 1000 HTTP errors from one
#: IP in a day (or 100 in 2 hours) results in an IP ban.
CELESTRAK_CACHE_TTL_S = 2 * 3600

#: Maximum retries per CelesTrak request. Their guidance is 2-3 attempts, then
#: stop. Repeated failures are a hard stop, never a signal to retry harder --
#: your own error rate is what triggers the ban.
CELESTRAK_MAX_RETRIES = 3

#: Identify the client so CelesTrak's maintainer can contact rather than block.
CELESTRAK_USER_AGENT = "AEGIS-ConjunctionAssessment/1.0 (+research; contact via repository)"

#: Space-Track's published API throttle is under 30 requests per minute and
#: under 300 per hour per account; violations suspend the account. The client
#: throttle sits below both so a burst of console reloads can never trip them.
SPACETRACK_MAX_PER_MINUTE = 20
SPACETRACK_MAX_PER_HOUR = 200

#: Space-Track asks that GP data be pulled at most once per hour. Readers
#: (the console) accept a cache up to two hours old; the scheduled refresh
#: re-downloads once the cache passes 50 minutes, so it alone keeps the data
#: fresh and nothing else downloads in the same hour.
SPACETRACK_CACHE_TTL_S = 2 * 3600
SPACETRACK_REFRESH_AGE_S = 50 * 60

#: Retries per Space-Track request. Low on purpose: failed requests still
#: count against the account throttle.
SPACETRACK_MAX_RETRIES = 2

#: Server-side LEO band for the Space-Track debris query, km. Deliberately
#: wider than any Starlink shell; ``overlapping_debris`` then trims to the
#: fleet's actual perigee/apogee window plus the band-check pad.
SPACETRACK_DEBRIS_PERIGEE_MAX_KM = 1000.0
SPACETRACK_DEBRIS_APOGEE_MIN_KM = 200.0

#: GP element sets older than this many days are dropped server-side;
#: SGP4 error growth makes them useless for a three-day screen.
SPACETRACK_MAX_EPOCH_AGE_DAYS = 10


# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------

#: Seconds in a day, for mean-motion conversions.
SECONDS_PER_DAY = 86400.0

#: Default console / ops-catalog screening horizon. 19 SDS screens LEO
#: 7 days out; operators decide on burns inside ~3 days. Three days is
#: long enough to catch debris conjunctions and short enough that TLE
#: along-track error is not yet fully diluted.
SCREENING_HORIZON_S = 3.0 * SECONDS_PER_DAY

#: Coarsest console globe track step, seconds: a window over two hours
#: screened at a coarser step still draws smooth 90-minute tracks.
CONSOLE_TRACK_STEP_S = 300.0

#: Console screening step, seconds. Coarser than ``SCREENING_STEP_S``
#: because the no-miss gate remains conservative at 19 km/s.
CONSOLE_STEP_S = 60.0

#: CelesTrak GP ``GROUP`` for the test fleet.
CELESTRAK_FLEET_GROUP = "starlink"

#: CelesTrak GP ``NAME`` query for catalogued debris. There is no
#: ``GROUP=debris``; ``NAME=DEB`` is the public debris dump.
CELESTRAK_DEBRIS_NAME = "DEB"

#: Space-Track ``OBJECT_NAME`` prefix for the test fleet.
SPACETRACK_FLEET_PREFIX = "STARLINK"

#: Convert TLE mean motion (revolutions per day) to radians per second.
REV_PER_DAY_TO_RAD_PER_S = 2.0 * 3.141592653589793 / SECONDS_PER_DAY
