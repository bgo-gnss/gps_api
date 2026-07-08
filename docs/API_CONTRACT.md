# gps_api — API Contract v0

> **Status: DRAFT v0** — Phase 0 artifact of the post-processing revamp
> (`gpslibrary_new/PLAN-postprocessing-revamp.md` §10.5). Review of this
> document is part of the Phase 0 exit gate. Phase 1 builds against it and
> the thin Dash QC tool must survive as a real consumer before the contract
> is considered stable.

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
| Caching | `/stations`, `/layers` cacheable; series endpoints short/no cache |

## Endpoints (v0 surface)

| Method | Path | Returns | Notes |
|---|---|---|---|
| GET | `/healthz` | `{"status","version"}` | liveness probe |
| GET | `/stations` | `StationCollection` (GeoJSON) | cacheable catalog |
| GET | `/stations/{marker}/series` | `SeriesResponse` | `start`, `end`, `downsample` (every Nth), `detrended` params |
| GET | `/velocities` | `VelocityCollection` (GeoJSON) | `region`, `window_years` filters |
| GET | `/models/{region}` | `ModelResult` | latest source model; Mogi first |
| GET | `/layers` | `LayerCatalog` | typed, versioned overlay catalog |
| POST | `/query` | `QueryResponse` | complex selections via JSON body |

Authoritative field-level schemas live in `src/gps_api/schemas.py` and in the
generated OpenAPI document (`/openapi.json`) — this file explains intent; the
code is the source of truth for shapes.

### Example: `GET /velocities?region=reykjanes`

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

### Example: `POST /query`

```json
{
  "regions": ["reykjanes"],
  "start": "2024-01-01T00:00:00Z",
  "end": "2026-07-01T00:00:00Z",
  "products": ["series", "velocities"]
}
```

## Open questions for review (decide before Phase 1 freezes the contract)

1. **Versioning** — path prefix (`/v1/...`) vs. unversioned-until-breaking.
   Proposal: stay unversioned through Phase 1 (single internal consumer),
   introduce `/v1` only if the SPA (Phase 4) needs a breaking change.
2. **Downsampling semantics** — plain every-Nth (current param) vs.
   LTTB/min-max for visual fidelity on long series.
3. **Pagination** — needed for `FeatureCollection`s at 173 stations? Proposal:
   no; the catalog is small and velocities are one feature per station.
4. **Auth** — public read for everything, or internal-only layers (e.g. QC
   flags)? The incumbent aflogun is public; proposal: keep the API fully
   public-read, keep QC-internal products out of it.
5. **Model history** — `/models/{region}` returns the latest fit; do we need
   `/models/{region}/history` for time-lapse (e.g. Svartsengi volume series)?
   Likely yes in Phase 2 — reconcile with Vincent's `inv_volume_mogi.dat` lane.

---

*Drafted 2026-07-08 (Phase 0). Owner: BGÓ.*
