# AEGIS limitations

AEGIS is a research prototype for constellation conjunction assessment and
fleet maneuver planning. This page states what the system does **not** claim.

## 1. TLEs carry no covariance

Two-line elements do not include uncertainty. AEGIS synthesizes a covariance
and tags it `SYNTHETIC_TLE`. That tag is provenance, not a calculated
orbit-determination result.

## 2. Synthetic covariance is for ranking, not operations

The synthetic TLE covariance supports ranking and triage of close approaches.
It is **not** for operational maneuver decisions. Do not treat a planned burn
as flight-ready because a synthesized Pc dropped below a threshold.

## 3. Intra-fleet Pc from TLEs is not CDM quality

Collision probabilities computed from TLE-grade ephemeris and synthetic
covariance are not operator-grade Conjunction Data Message quality. They are
a screening and research signal, not a replacement for owner/operator CDMs.

## 4. The maneuver plan is not flight commands

The maneuver plan is a prototype / research output. It is not a set of flight
commands. Burns are along-track impulses from a linear program on Clohessy–
Wiltshire responses; they are not validated guidance products.

## 5. Synthetic catalogs are opt-in and distinct from CelesTrak

Generated catalogs require `AEGIS_ALLOW_SYNTHETIC=1` plus an explicit
authorization. That path is opt-in. Synthetic objects must not be confused
with CelesTrak data.

## 6. CelesTrak failure is not a silent fallback

If CelesTrak ingest fails, the pipeline raises. It does **not** silently
become synthetic data. A refused or failed live fetch never substitutes a
generated catalog.
