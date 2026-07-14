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
- **Store-wired** (fleet slice 2026-07-11; Mogi/MLE slice 2026-07-12):
  `/v1/stations`, `/v1/stations/{marker}/series` (LTTB via `downsample.py`;
  `api.max_points` from the run meta is default + clamp — contract
  Amendment A3; `clean` param + per-epoch `outlier` flags — Amendment A8,
  raw is the default truth, `clean=true` drops flagged epochs BEFORE
  LTTB; pre-A8 stores serve nullable fields), `/v1/velocities`, `/v1/models/{region}`
  (`kind="breakpoints"`, GBIS4TS), `/v1/deformation/{region}` (Mogi ΔV(t)
  time series + Bayesian posterior — A6; **or** Okada distributed-slip
  distribution — A7; `source_type`-discriminated union, one source/region).
  Still **501 stubs**: `/models/{region}/history` (reserved, Decisions #5),
  `/layers`, `/query` — keep the `not_implemented()` helper pattern so
  error shape stays uniform. Velocity `method` values: `wls | mle` live
  (`mle` per-region via `velocity_method`, honest colored-noise σ + `noise`
  provenance — Amendment A5), `gbis` reserved; see Amendments A1–A6.

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

**Mogi deformation + MLE velocity** (2026-07-12, Amendments A5/A6): the
job now runs `gps_analysis.estimate_velocity_mle` for regions configuring
`velocity_method: mle` (WLS stays the fleet baseline) and, for regions in
`deformation.enabled_regions` (gated like breakpoints), the Mogi stage
(`precompute/deformation.py`): per grid epoch, station displacements
relative to the trailing-window start → `mogi_invert` in a local
tangent-plane frame → `models/<region>_deformation.json` (ΔV(t)/depth/
position + σ; optional `mogi_invert_bayes` posterior for the newest epoch
when `deformation.bayes.n_runs > 0`). A stage failure is recorded
(`deformation_failed` in `meta/run.json`) without sinking the region. The
product is an **independent GNSS-only** analog of Vincent's operational
Mogi ΔV(t) (`insar.vedur.is:.../inv_volume_mogi.dat`) — cross-checked, never
derived. CLI: `--no-deformation`. Per-epoch fits are multi-start (warm+cold)
with bound-pinned optima rejected (`_invert_epoch`/`_is_interior`) —
real-data robustness fix.

**Okada distributed slip** (2026-07-12, Amendment A7, `precompute/slip.py`):
`deformation.source: okada` inverts a **single-window** slip distribution on
an **operator-supplied fixed plane** (`deformation.okada` → `OkadaPlaneConfig`;
config-driven per intrusion, NOT auto-found). Net window displacement →
`discretize_fault`→`okada_greens`→`okada_invert_slip` (Laplacian-reg ± NNLS;
λ fixed or `slip_lcurve` corner) → `models/<region>_slip.json`
(`SlipDistributionResult`: per-`FaultPatch` slip/σ + potency + norms), served
on the same `/v1/deformation/{region}` endpoint, `source_type`-discriminated
from Mogi (mogi XOR okada). Per-patch σ = unconstrained linear-Gaussian formal
cov via the public G/L operators (`_slip_formal_cov`; not exact for NNLS-pinned
patches — provenance `sigma_note`). Degenerate solves → `deformation_failed`.

**Real-data validation** (2026-07-12): `gps_api.validation.realdata` +
`gps-api-validate-deformation` reconcile the pipeline on real Svartsengi
`.NEU` (CDN) against the operational model (read-only SSH,
`/mnt/scratch/vincent/model/svartsengi/`). Baseline inflation08: ΔV(t)
r=0.993, verdict + numbers in `docs/VALIDATION_svartsengi_deformation.md`.
Fixture `tests/fixtures/realdata/` is gitignored (`fetch` rebuilds);
`tests/test_validation_realdata.py` is skipped without it.

**Outlier detection** (2026-07-13, Amendment A8, `precompute/outliers.py`;
full detail: contract A8 + `gps_analysis/docs/DESIGN_outlier_detection.md`
§5/§9): gated by the `analysis.yaml` `outliers:` block (`OutlierConfig` —
5/5/10 mm H/H/V floors + validated per-station `overrides`; CLI
`--no-outliers`; absent block = off). Calls `gps_analysis.detect_outliers`
(branch `outlier-detection-leaf`) with **step-augmented** inputs from the
deployed `steps.csv` (`load_step_catalog`, `N|E|U|ALL` → per-component
lists — the SENG lesson: a stepless model over-flags active stations).
NON-destructive: raw parquet columns byte-identical; additive `*_outlier`
/ `*_outlier_reason` / `*_outlier_protected` / `outlier_epoch` columns +
`outliers` provenance (params echo, counts, events, abort, `params_hash`).
Downstream estimates fit on the INLIERS (velocity/deformation union mask;
GBIS4TS per-component masks + outlier-config hash in breaks provenance —
Q8). `meta/suspected_steps.csv` = protected `SuspectedEvent` clusters for
operator review (Q5); aborts (`outliers_aborted`) and failures
(`outliers_failed`) land in `meta/run.json`, station proceeds unmasked —
loud, never silently clipped.

**Detrend-parameter estimation** (2026-07-14, `precompute/detrend.py`;
design: `gps_analysis/docs/DESIGN_live_detrending.md` §0): the WRITER half
of the geo_dataread handshake — gated by the `analysis.yaml`
`detrend.estimation:` block (`DetrendConfig`; CLI `--no-detrend-params`;
absent block = off). Per station: window policy (whole series by default,
"as long as possible" §0.7; trailing `fit_window_years`; per-station
`fit_windows` override) → `gps_analysis.estimate_detrend` (leaf gates +
`steps.csv` step augmentation, union over components + the resolved
`outliers:` thresholds/floors) → `DetrendEstimate.to_record` →
`params/detrend_params.json` (`{"schema_version": 1, "stations": {STA:
record}}` — byte-compatible with `geo_dataread.gps_views.read_detrend_params`;
fleet runs merge one document). Frame tag `plate:<region frame>` (plate-first
§0.5 — the `.NEU` inputs are plate products); `pinned: {STA: keep|path}` =
honor verbatim, never refit (§0.7); `use_sta: {borrower: donor}` = self-
contained borrowed record + `borrowed` provenance (§0.6). Graceful + LOUD
(§0.4): gate failures → station absent + reason in `meta/run.json`
`detrend_params.skipped`; outlier abort OR `max_rms_mm` excess (real-data
SENG finding: an unrest-spanning window does NOT abort — the robust fit
swallows the transient, rms 100s of mm; the rms gate catches it) →
`degraded`, written only under `write_degraded: true` with `refs.degraded`.
The store document is a CANDIDATE — deploy to gpsconfig (where geo_dataread
resolves it) stays BGÓ's reviewed act (§3.3).

**Parallel breaks + triage** (`precompute/breaks.py`, perf-audit #1/#6 +
plan §10.7): the gated GBIS4TS chains fan out over a
`ProcessPoolExecutor` (spawn; workers return 256-byte scalar
`BreakSummary`s, never the ~64 MB kept chain — memory stays ~workers×1
chain). Optional triage→confirm: `breakpoints.triage_n_runs > 0` screens
every gated station with a short chain and confirms only stations whose
trend-change posterior `|mean|/std` ≥ `triage_sigma`; flagged/screened
counts are logged and stamped into the breaks-product provenance (never a
silent cap). Config keys `triage_n_runs` (0 = off) / `triage_t_runs` /
`triage_sigma` / `max_workers` (absent → cpu count, 0 = inline); CLI
`--triage-runs/--triage-t-runs/--workers`. Same seed → identical summaries
to the old serial path (tests pin exact equality).

## Layout & commands

```
src/gps_api/{main.py, schemas.py, settings.py, downsample.py,
             routers/{stations,velocities,models,deformation,layers,query}.py,
             precompute/{config,sources,products,job,breaks,outliers,detrend,deformation,slip}.py,
             validation/realdata.py}  # real-data harness (precompute-side)
tests/test_app.py         # contract-shape tests (routes, 404/501+detail, OpenAPI)
tests/test_precompute.py  # end-to-end: config → precompute (region + fleet) → store → wired endpoints
tests/test_outliers_wiring.py  # A8 slice: byte-identical raw columns, additive flags, declared-step/abort behavior, suspected_steps.csv, cleaned GBIS input + hash, overrides, clean param, fault tolerance
tests/test_breaks_parallel.py  # pool==serial parity, triage flags, bounded summaries, fault tolerance
tests/test_deformation.py # Mogi ΔV(t) recovery + MLE velocities + gating + endpoint + fault tolerance
tests/test_slip.py        # Okada distributed-slip recovery + σ faithfulness + L-curve + gating + endpoint + fault tolerance
tests/test_detrend_params.py  # detrend-params writer: geo_dataread round-trip handshake, schema-v2 refusal, pinning/UseSTA/degrade, real SENG pre-unrest + unrest behavior (imports geo_dataread — editable sibling; module skips without it)
tests/test_downsample.py  # LTTB property tests + single-channel reference parity
tests/test_validation_realdata.py  # env-gated real-data reconciliation (skipped w/o fixture)
```

```bash
uv sync --all-groups && uv run gps-api    # dev server :8000, docs at /docs
uv run gps-api-precompute --synthetic --runs 2000  # foreground batch (dev chain)
uv run gps-api-precompute --neu-dir <dir>          # real .NEU products, one region
uv run gps-api-precompute --fleet --neu-dir <dir>  # all configured regions, one store
uv run gps-api-precompute --no-outliers ...        # skip the outlier stage (A8)
uv run gps-api-precompute --no-detrend-params ...  # skip detrend-params estimation
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
*Last reviewed: 2026-07-14 (detrend-estimation-precompute: `precompute/detrend.py`
+ `DetrendConfig` (`detrend.estimation:` block), `params/detrend_params.json`
writer byte-compatible with `geo_dataread.gps_views.read_detrend_params` —
pinning/UseSTA/degrade per DESIGN_live_detrending §0; real-data SENG rms-gate
finding. Prior 2026-07-13: wire-outlier-detection: `precompute/outliers.py`
+ `OutlierConfig`/`load_step_catalog`, additive parquet flag columns +
`suspected_steps.csv`, inlier-fitted downstream estimates, series `clean`
param + `outlier` flags — Amendment A8; develops against gps_analysis
branch `outlier-detection-leaf`. Prior 2026-07-12: productize-okada-slip
A7, validate-deformation-realdata Svartsengi reconciliation + Mogi
multi-start/interior-guard fix, productize-mogi-mle A5–A6,
fleet-parallel-mcmc pooled GBIS4TS chains + triage→confirm; 2026-07-11
fleet rollout A1–A4)*
