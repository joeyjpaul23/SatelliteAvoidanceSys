# AEGIS console UI — plan (review before build)

Decisions from the operator (2026-08-30):

- **Catalog:** live CelesTrak Starlink **plus overlapping catalog debris** (`NAME=DEB`) when the network works; **cached slices** if it fails. Never synthetic as a fallback.
- **Horizon:** **3 days** (`SCREENING_HORIZON_S`). Not 90 minutes.
- **Scope:** visualize everything the backend already computes (tracks, risk, covariance, screening geometry, flags, plan, provenance).
- **Scale:** object-count slider, default **40** (split ~half Starlink / half debris), hard cap **200**. Slider is how many objects are **screened**. The globe and lists only show objects in MONITOR / WATCH / ACT events.
- **Display:** omit CLEAR. One event per pair (closest miss). Tracks are ~one orbit around the first at-risk TCA, not the full 3-day window.

This is an operations console over the existing pipeline, not a new physics layer.

## What you see

A single local web app: Earth globe, satellites at SGP4 TEME positions for the catalog epoch (and a time scrubber across the screening window), orbital tracks, and a right-hand engineering panel.

**Track / sat color** (worst conjunction that involves that object):

| Band | When | Color |
|---|---|---|
| CLEAR | no event ≥ `PC_THRESHOLD_ASSESS` (1e-7) | green `#3DFF8A` (dim, not neon candy) |
| MONITOR | Pc ≥ 1e-7 | yellow `#E8C547` |
| WATCH | Pc ≥ 1e-5 | orange `#E07A3D` |
| ACT | Pc ≥ 1e-4 | red `#E23B2F` |

**Uncertainty (TLE / `SYNTHETIC_TLE`):** color is not the raw Pc band alone.

- If the object’s covariance source is `SYNTHETIC_TLE` **and** any of its conjunctions has `dilution_flag` or miss < 3 × projected σ_major, **raise the displayed band by one** (CLEAR→MONITOR, MONITOR→WATCH, WATCH→ACT). The panel must say `DISPLAY BAND inflates one step — TLE covariance`.
- Draw a faint along-track **uncertainty ribbon** (tube or line pair) scaled from `TleCovarianceModel` σ_transverse at that lead time — not a pretty glow, a dimensioned envelope. Label in the panel: σ_R / σ_T / σ_N km.

**Also drawn / listed (all from existing types):**

- Conjunction chords (primary–secondary at TCA), colored by `RiskLevel`.
- Screening-box half-widths in the selected event inspector (2 × 44 × 51 km Starlink default) as numbers, not a giant 3D box on every pair.
- Selected event: Pc (Alfano), Chan cross-check if present, miss, TCA, relative speed, Mahalanobis, dilution / remediated / short-encounter / low-rel-vel flags, `covariance_source`.
- Maneuver plan: burns as short RTN ticks on the sat; unresolved table sorted by shortfall; `plan.summary()`.
- Provenance strip: `source` (CELESTRAK), `query`, `fetched_at`, live vs slice fallback, `SYNTHETIC_TLE` honesty line. If live fetch failed, show `FALLBACK SLICE` — never imply live.

Synthetic catalog is **not** in the default path. A hidden/advanced control may exist only if `AEGIS_ALLOW_SYNTHETIC=1` **and** the user ticks an explicit acknowledge box (same dual gate). Off by default.

## Architecture

```
browser (vanilla JS + Three.js r160+, no React)
    GET /api/scene
FastAPI  aegis.api
    run_pipeline / screen / assess_catalog / plan_maneuvers
    Sgp4Propagator.propagate_grid
```

- `src/aegis/api/` — FastAPI app, CORS localhost only by default.
- `src/aegis/ui/` — static `index.html`, `css/console.css`, `js/scene.js`, `js/console.js`. Three.js from a pinned CDN or vendored file (pin version).
- `python -m aegis.api` serves UI at `/` and JSON under `/api/`.
- One **scene** payload so the front end does not re-derive Pc.

### `GET /api/scene`

Query: `max_objects` (1–200, default 40), `duration_s` (default `SCREENING_HORIZON_S` = 259200 / 3 days), `step_s` (default 60), `live=1` (default true).

Ingest:

1. If `live=1`, try `fetch_celestrak("starlink")` and `fetch_celestrak("DEB", field="NAME")`. On `CelesTrakError` / network fail for a half → that half’s committed slice (`starlink_slice.tle` / `debris_slice.tle`) and set `fallback="slice"`.
2. Keep debris whose altitude band can meet the Starlink slice (same pad as the apogee/perigee prefilter). Cap ~half fleet / ~half debris.
3. Never call `generate_synthetic` on this path.
4. Tag Starlink as maneuverable PAYLOAD; debris as DEBRIS (not maneuverable).

Then: screen with `keep_pair` (skip debris–debris); assess_catalog; drop CLEAR (keep MONITOR / WATCH / ACT, using `display_band` so TLE inflation still shows); one closest event per pair; propagate **only those objects** for ~one orbit around the first TCA; plan_maneuvers on the at-risk set. Build JSON:

```
{
  source, fallback, covariance_source, fetched_at, query,
  epoch, duration_s, step_s, max_objects,
  objects: [{ id, name, track: [[x,y,z],...], color_band, sigma_rtn_km, conjunction_ids }],
  conjunctions: [{ id, primary_id, secondary_id, tca, miss_km, pc, risk_level,
                   relative_speed_km_s, mahalanobis, dilution, remediated,
                   short_encounter_valid, low_relative_velocity, flags[] }],
  plan: { summary, burns[], unresolved[] },
  honesty: ["SYNTHETIC_TLE ...", ...]
}
```

Positions in **km**, TEME as used internally (no silent frame conversion). Front end maps TEME xyz to Three.js (Earth radius 6378.137, same as `R_EARTH_KM`).

### Other endpoints (thin)

- `GET /api/health` → `{ok: true, version}`
- `GET /api/scene` as above

No extra physics.

## Visual language (SpaceX-ops, not “AI space”)

- Background `#07080A`. Type: `IBM Plex Mono` + `IBM Plex Sans` (or Inter + JetBrains Mono). Tight tracking, small caps for labels (`TCA`, `PC`, `RTN`).
- Hairline borders `#2A2D32`, text `#E6E4DF`, muted `#8B8E93`. No purple, no glassmorphism, no gradient orbs, no “mission control HUD” clipart.
- Left: globe (majority). Right: 320–380px rail — SOURCE, OBJECTS, EVENTS, PLAN. Numbers in tabular lining.
- Wordmark: `AEGIS` + `CONJUNCTION` in 11px, not a logo.
- Earth: NASA Blue Marble or similar **textured** sphere + simple atmosphere limb; stars as a sparse point field. Satellites: 1–2 px nodes + thin track. No cartoon ISS models.
- Interaction: click sat → select; click event row → highlight both tracks + TCA chord. Time slider scrubs the grid.

## Tests (tester-owned; builder does not write them)

1. **API:** live fail → scene `source=CELESTRAK`, `fallback=slice`, no SYNTHETIC objects (mock `fetch_celestrak` to raise).
2. **API:** `max_objects=5` → ≤5 objects.
3. **API:** `max_objects=201` → 400.
4. **API:** displayed objects have `color_band` in {MONITOR,WATCH,ACT} (CLEAR omitted); honesty mentions SYNTHETIC_TLE and the risk-only rule. `screened_objects` is the catalog size; `objects` is the at-risk subset.
5. **Color helper** (pure function in `aegis.api.color` used by scene builder): band from Pc; +1 if synthetic TLE and (dilution or miss < 3σ). Unit tests on that function.
6. **Static:** `GET /` returns HTML containing a globe mount and no “AI” marketing copy.
7. **Optional:** TestClient loads `/js/scene.js`.

No browser E2E required for the loop; orchestrator verifies in the browser after green.

## Out of scope

GNN, live TraCSS, authenticating Starlink Space Safety, mobile layout, React rewrite.

## Build order

1. `aegis.api.color` + scene DTO from existing pipeline.
2. FastAPI `/api/health`, `/api/scene`, `/`.
3. Three.js console UI.
4. Tester from this spec; analyzer/builder loop until green.
5. Orchestrator browser pass: slider, fallback honesty, click event, colors.
