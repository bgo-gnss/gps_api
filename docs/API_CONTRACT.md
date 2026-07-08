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

| Method | Path | Returns | Notes |
|---|---|---|---|
| GET | `/healthz` | `{"status","version"}` | liveness probe (unversioned) |
| GET | `/v1/stations` | `StationCollection` (GeoJSON) | cacheable catalog |
| GET | `/v1/stations/{marker}/series` | `SeriesResponse` | `start`, `end`, `max_points` (target count, LTTB), `detrended` params |
| GET | `/v1/velocities` | `VelocityCollection` (GeoJSON) | `region`, `window_years` filters |
| GET | `/v1/models/{region}` | `ModelResult` | latest source model; Mogi first |
| GET | `/v1/models/{region}/history` | `ModelHistory` | fit time-lapse; reserved in v0, wired Phase 2 |
| GET | `/v1/layers` | `LayerCatalog` | typed, versioned overlay catalog |
| POST | `/v1/query` | `QueryResponse` | complex selections via JSON body |

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

---

*Drafted + reviewed 2026-07-08 (Phase 0). Owner: BGÓ.*
