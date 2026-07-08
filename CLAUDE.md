# CLAUDE.md — gps_api

Tier-2 service (plan §3): **read-only FastAPI** over the precomputed GNSS
analysis store. The precompute job writes (Postgres: metadata/velocities/
catalogs; files: Parquet series, GeoJSON, rasters); this API only reads.
Consumers: thin Dash QC tool (Phase 1), aflogun SPA (Phase 4), gps_plot.

> **Read first:** `../PLAN-postprocessing-revamp.md` (§10.5 = this service's
> contract rules; §6 = Phase 1 DoD it must satisfy).

## Contract

- `docs/API_CONTRACT.md` is the reviewable contract v0; `src/gps_api/schemas.py`
  is the typed source of truth for shapes. Change them **together**.
- Non-negotiables: GeoJSON FeatureCollections for anything mappable; UTC
  ISO-8601 `Z`; explicit unit fields (mm, mm/yr); `{"detail": …}` errors;
  `/stations` (cacheable catalog) split from `/stations/{marker}/series`
  (on-demand); complex selections via `POST /query`; typed versioned `/layers`.
- All data endpoints are **501 stubs** until Phase 1 wires the store — keep
  the `not_implemented()` helper pattern so error shape stays uniform.

## Layout & commands

```
src/gps_api/{main.py, schemas.py, routers/{stations,velocities,models,layers,query}.py}
tests/test_app.py        # contract-shape tests (routes, 501+detail, OpenAPI)
```

```bash
uv sync --all-groups && uv run gps-api    # dev server :8000, docs at /docs
uv run ruff check src tests && uv run black --check src tests
uv run mypy src tests && uv run pytest
```

## Rules

- Python ≥3.13, hatchling, uv; ruff+black+mypy(strict) zero warnings.
- Home: **GitLab** (git.vedur.is, services) — not GitHub. CI: `.gitlab-ci.yml`
  (self-contained; org template include documented inside it).
- The API never imports `gps_analysis` or `geo_dataread` — it reads the
  store the precompute job filled. The precompute job (Phase 1) may live in
  this repo as a sibling module or its own repo; decide when it lands.

---
*Last reviewed: 2026-07-08*
