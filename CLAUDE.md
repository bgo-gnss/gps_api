# CLAUDE.md — gps_api

Tier-2 service (plan §3): **read-only FastAPI** over the precomputed GNSS
analysis store. The precompute job writes (Postgres: metadata/velocities/
catalogs; files: Parquet series, GeoJSON, rasters); this API only reads.
Consumers: thin Dash QC tool (Phase 1), aflogun SPA (Phase 4), gps_plot.

> **Read first:** `../PLAN-postprocessing-revamp.md` (§10.5 = this service's
> contract rules; §6 = Phase 1 DoD it must satisfy).

## Contract

- `docs/API_CONTRACT.md` is the contract (v0, **reviewed 2026-07-08** — its
  Decisions section is binding); `src/gps_api/schemas.py` is the typed source
  of truth for shapes. Change them **together**.
- Non-negotiables: GeoJSON FeatureCollections for anything mappable; UTC
  ISO-8601 `Z`; explicit unit fields (mm, mm/yr); `{"detail": …}` errors;
  `/v1/stations` (cacheable catalog) split from `/v1/stations/{marker}/series`
  (on-demand, `max_points`/LTTB); complex selections via `POST /v1/query`;
  typed versioned `/v1/layers`. Data endpoints live under `/v1` (`/healthz`
  unversioned); fully public read, no auth, QC products stay out.
- **Store-wired** (fleet slice, 2026-07-11): `/v1/stations`,
  `/v1/stations/{marker}/series` (LTTB via `downsample.py`; `api.max_points`
  from the run meta is default + clamp — contract Amendment A3),
  `/v1/velocities`, `/v1/models/{region}` (`kind="breakpoints"`, GBIS4TS).
  Still **501 stubs**: `/models/{region}/history` (reserved, Decisions #5),
  `/layers`, `/query` — keep the `not_implemented()` helper pattern so
  error shape stays uniform. `method` values: `wls | gbis` (`mle` reserved);
  see the contract's Amendments A1–A4.

## Precompute job (decision: lives here, `gps_api.precompute`)

The Phase-1 slice landed the scheduled precompute as a module of this repo
(plan §3 "in gps_api or a sibling" — decided 2026-07-11). It calls the
`gps_analysis` public API (fit/detrend, WLS velocity, GBIS4TS break points —
series passed straight in; the leaf auto-zero-references) and writes the
file store the API serves; Postgres is the next slice. Config comes from
`analysis.yaml` + `stations.cfg` via `gps_parser` (`$GPS_CONFIG_PATH`) —
zero hardcoded paths/stations. Store root: `$GPS_API_STORE` →
`~/.cache/gps_analysis` (`settings.py`, shared by writer + routers). Layout:
`stations.geojson`, `velocities/<region>.geojson`, `series/<MARKER>.parquet`,
`models/<region>_breaks.json`, `meta/run.json` — GeoJSON validated through
`schemas.py` before writing; every product carries provenance (method,
frame, software versions, `fitted_at`, source). The API routers still never
import `gps_analysis`/`gps_parser` — only `gps_api.precompute` does (deps in
the `precompute` dependency group; the API runtime reads the store with
numpy/pyarrow only; editable sibling paths via `[tool.uv.sources]`, so
GitLab CI needs the git-dep switch once the analysis-lane branch
merges/publishes).

**Fleet runs** (`run_fleet` / `--fleet`, Phase-2 rollout): every region in
`cfg.regions` through the same per-region chain into ONE coherent store —
combined `stations.geojson` (multi-region membership merged), per-region
velocity/break products, one fleet `meta/run.json` (per-region +
per-station success/failure counts). Fault tolerance at both levels: a bad
station is skipped inside its region, a bad region is skipped by the fleet.
GBIS4TS stays gated by `breakpoints.enabled_regions` — WLS is the
fleet-wide baseline; never run the 1e6 chains across all stations.

**Parallel breaks + triage** (`precompute/breaks.py`, perf-audit #1/#6 +
plan §10.7): the gated GBIS4TS chains fan out over a
`ProcessPoolExecutor` (spawn; workers return 256-byte scalar
`BreakSummary`s, never the ~64 MB kept chain — memory stays ~workers×1
chain). Optional triage→confirm: `breakpoints.triage_n_runs > 0` screens
every gated station with a short chain and confirms only stations whose
trend-change posterior `|mean|/std` ≥ `triage_sigma`; flagged/screened
counts are logged and stamped into the breaks-product provenance (never a
silent cap). Config keys: `triage_n_runs` (0 = off, default),
`triage_t_runs`, `triage_sigma`, `max_workers` (absent → cpu count,
0 = inline); CLI `--triage-runs/--triage-t-runs/--workers`. Same seed →
identical summaries to the old serial path (tests pin exact equality).

## Layout & commands

```
src/gps_api/{main.py, schemas.py, settings.py, downsample.py,
             routers/{stations,velocities,models,layers,query}.py,
             precompute/{config,sources,products,job}.py}
tests/test_app.py         # contract-shape tests (routes, 404/501+detail, OpenAPI)
tests/test_precompute.py  # end-to-end: config → precompute (region + fleet) → store → wired endpoints
tests/test_breaks_parallel.py  # pool==serial parity, triage flags, bounded summaries, fault tolerance
tests/test_downsample.py  # LTTB property tests + single-channel reference parity
```

```bash
uv sync --all-groups && uv run gps-api    # dev server :8000, docs at /docs
uv run gps-api-precompute --synthetic --runs 2000  # foreground batch (dev chain)
uv run gps-api-precompute --neu-dir <dir>          # real .NEU products, one region
uv run gps-api-precompute --fleet --neu-dir <dir>  # all configured regions, one store
uv run ruff check src tests && uv run black --check src tests
uv run mypy src tests && uv run pytest
```

## Rules

- Python ≥3.13, hatchling, uv; ruff+black+mypy(strict) zero warnings.
- Home: **GitLab** (git.vedur.is, services) — not GitHub. CI: `.gitlab-ci.yml`
  (self-contained; org template include documented inside it).
- The API never imports `gps_analysis` or `geo_dataread` — it reads the
  store the precompute job filled. The precompute module is the only place
  those imports are allowed.

---
*Last reviewed: 2026-07-12 (fleet-parallel-mcmc: pooled GBIS4TS chains +
triage→confirm; prior review 2026-07-11 fleet rollout — stations/series/models
endpoints, LTTB, contract Amendments A1–A4)*
