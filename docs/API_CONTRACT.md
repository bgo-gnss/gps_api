# gps_api — API Contract v0

> **Status: v0 — REVIEWED 2026-07-08 (BGÓ)** — Phase 0 artifact of the
> post-processing revamp (`gpslibrary_new/PLAN-postprocessing-revamp.md`
> §10.5). The five open questions are resolved (see *Decisions* below).
> Phase 1 builds against this contract; the thin Dash QC tool must survive
> as a real consumer before it is considered stable.

## Principles

1. **Read-only.** A scheduled precompute job writes products to the store
   (Postgres for metadata/velocities/catalogs; Parquet/GeoJSON files for bulk
   series and rasters). The API only reads. Web UIs only query the API.
2. **GeoJSON for anything mappable.** Stations, velocity vectors, InSAR
   footprints, model nodes — all served as `FeatureCollection`s so MapLibre
   layers consume them directly. Vector payloads carry `magnitude` + `azimuth`
   in `properties` alongside the component values.
3. **Catalog / payload split.** `GET /stations` is small and cacheable;
   `GET /stations/{marker}/series` is on-demand and supports downsampling.
   (skjalftalisa lesson — never merge the two.)
4. **Complex queries are one POST body.** `POST /query` takes regions,
   polygons, station lists and time windows as JSON — not dozens of repeated
   query parameters.
5. **Normalized shapes only.** UTC ISO-8601 `Z` timestamps; explicit unit
   fields; raw GAMIT/processing formats never leak through the API.
6. **Uniform errors.** Every non-2xx response body is `{"detail": …}`
   (FastAPI convention), including 422 validation errors.
7. **Typed, versioned layer catalog.** `GET /layers` drives data-driven map
   overlays (GeoJSON/WMS/TMS/image) so clients add layers without code changes.

## Conventions

| Topic | Rule |
|---|---|
| Time | ISO-8601 UTC with `Z` suffix, e.g. `2026-07-08T00:00:00Z` |
| Coordinates | GeoJSON order `[lon, lat]` (optionally `[..., height_m]`), WGS84 |
| Displacements | millimetres (`"units": "mm"` field in payload) |
| Velocities | mm/yr; `azimuth` degrees clockwise from north; `sigma_*` mm/yr |
| Depths | kilometres (`depth_km` model parameter) |
| Errors | `{"detail": "<message>"}` for all 4xx/5xx |
| Caching | `/v1/stations`, `/v1/layers` cacheable; series endpoints short/no cache |
| Versioning | all data endpoints under `/v1`; `/healthz` unversioned |
| Access | fully public read; no auth; QC-internal products never enter this API |

## Endpoints (v0 surface)

| Method | Path | Returns | Status | Notes |
|---|---|---|---|---|
| GET | `/healthz` | `{"status","version"}` | live | liveness probe (unversioned) |
| GET | `/v1/stations` | `StationCollection` (GeoJSON) | **wired** | cacheable catalog; fleet runs span all regions, `properties.regions` merged |
| GET | `/v1/stations/{marker}/series` | `SeriesResponse` | **wired** | `start`, `end`, `max_points` (target count, LTTB), `detrended` params; server-side ceiling — see Amendment A3 |
| GET | `/v1/velocities` | `VelocityCollection` (GeoJSON) | **wired** | `region`, `window_years` filters |
| GET | `/v1/models/{region}` | `ModelResult` | **wired** | latest model products; `kind="breakpoints"` (GBIS4TS) first — see Amendment A2; Mogi reserved |
| GET | `/v1/models/{region}/history` | `ModelHistory` | 501 | fit time-lapse; reserved in v0 (Decisions #5), needs run accumulation (Postgres slice) |
| GET | `/v1/layers` | `LayerCatalog` | 501 | typed, versioned overlay catalog |
| POST | `/v1/query` | `QueryResponse` | 501 | complex selections via JSON body |

Authoritative field-level schemas live in `src/gps_api/schemas.py` and in the
generated OpenAPI document (`/openapi.json`) — this file explains intent; the
code is the source of truth for shapes.

### Example: `GET /v1/velocities?region=reykjanes`

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": { "type": "Point", "coordinates": [-22.4353, 63.8721] },
      "properties": {
        "marker": "SENG",
        "east": 12.3, "north": -4.1, "up": 31.9,
        "sigma_east": 0.4, "sigma_north": 0.3, "sigma_up": 1.1,
        "magnitude": 12.97, "azimuth": 108.4,
        "method": "wls",
        "window_start": "2025-11-01T00:00:00Z",
        "window_end": "2026-07-01T00:00:00Z"
      }
    }
  ]
}
```

### Example: `POST /v1/query`

```json
{
  "regions": ["reykjanes"],
  "start": "2024-01-01T00:00:00Z",
  "end": "2026-07-01T00:00:00Z",
  "products": ["series", "velocities"]
}
```

## Decisions (contract review, BGÓ, 2026-07-08)

1. **Versioning: `/v1` prefix from day one.** One-line router prefix now; the
   API is destined for the public `aflogun.vedur.is`, and renaming paths after
   the SPA ships would be a coordinated break. `/healthz` stays unversioned.
2. **Downsampling: `max_points` + LTTB.** The parameter expresses intent
   (target point count); the server guarantees a visually faithful reduction
   (LTTB — peaks, offsets and trend shape survive). Phase 1 may implement
   naively; the contract never has to change when the internals improve.
   Plain every-Nth decimation was rejected — it can erase exactly the
   transients the portal exists to show.
3. **No pagination.** Station catalog and velocity collections are one small
   feature per station (173 stations); series endpoints are already windowed
   by `start`/`end`. Revisit only if per-window feature products multiply.
4. **Fully public read; no auth.** Matches the incumbent public aflogun.
   QC-internal products (flags, station health) never enter this API — the
   Dash QC tool reads the store directly or gets an internal service later.
5. **`/v1/models/{region}/history` reserved now.** 501 stub + `ModelHistory`
   schema in v0; Phase 2 wires it for the Svartsengi volume time-lapse and
   the reconciliation against Vincent's `inv_volume_mogi.dat`.

## Amendments (fleet precompute rollout, 2026-07-11)

Additive contract changes landed with the Phase-2 fleet slice; `schemas.py`
changed in the same commit (the two are always changed together).

- **A1 — velocity `method` values: `"wls" | "gbis"` (`"mle"` reserved).**
  PLAN-analysis-lane §1: WLS is the fleet-wide baseline (fast, formal σ);
  GBIS4TS is the honest-σ upgrade (joint break + colored-noise estimation),
  selective per `breakpoints.enabled_regions` — never fleet-wide.
- **A2 — `ModelResult.kind` values: `"mogi" | "breakpoints"`.** The
  `"breakpoints"` kind serves the GBIS4TS break/rate-change catalog
  (`models/<region>_breaks.json`) losslessly: `entries` is a list of
  `BreakEntry` (marker, component, BPD1/BPD2 model tag, posterior-optimal
  parameters incl. κ and noise amplitude, break epoch as UTC time,
  `wn_amp_mm`, `y_ref_mm`, `n_runs`); `parameters` stays empty for this
  kind (it belongs to the reserved `"mogi"` source-model kind).
  `ModelResult.provenance`/`ModelFit.provenance` widened to carry the
  structured provenance object products are stamped with.
- **A3 — server-side `max_points` ceiling.** `analysis.yaml api.max_points`
  is recorded by the precompute run in the store's `meta/run.json`; the
  series endpoint applies it as the default LTTB target when the client
  sends no `max_points` and clamps the client's value when it does
  (`target = min(max_points, ceiling)`). A store without the key has no
  ceiling. LTTB always keeps the first/last epoch of the served window,
  and selects *real* observed points — served `sigma_*` stay the
  observation uncertainties of exactly the points shown.
- **A4 — fleet stores.** A `--fleet` precompute run writes one combined
  `stations.geojson` across all configured regions (a station in several
  regions lists them all, sorted, in `properties.regions`), per-region
  velocity/break products, and a single fleet-shaped `meta/run.json`
  (per-region + per-station success/failure counts). The API surface is
  unchanged — the same endpoints serve single-region and fleet stores.

---

*Drafted + reviewed 2026-07-08 (Phase 0). Amended 2026-07-11 (fleet slice:
endpoint statuses, A1–A4). Owner: BGÓ.*
