# gps_api

Read-only FastAPI service serving precomputed GNSS analysis products for the
IMO 173-station network: station catalog, N/E/U displacement series,
velocity fields and deformation-source models (Mogi → Okada → joint).

Part of the post-processing revamp
(`gpslibrary_new/PLAN-postprocessing-revamp.md`): a scheduled precompute job
writes to the store (Postgres + Parquet/GeoJSON files); this API reads and
serves; the Dash QC tool and the aflogun SPA are its consumers.

**Status: Phase 0 scaffold.** All data endpoints are 501 stubs; the contract
they will honour is `docs/API_CONTRACT.md` + `src/gps_api/schemas.py`.

## Development

```bash
uv sync --all-groups     # install package + dev tools
uv run gps-api           # dev server on http://127.0.0.1:8000 (docs at /docs)

uv run ruff check src tests
uv run black --check src tests
uv run mypy src tests
uv run pytest
```

## Layout

```
src/gps_api/
├── main.py       # app factory + /healthz + dev entry point
├── schemas.py    # pydantic contract types (GeoJSON collections, series, query)
└── routers/      # stations, velocities, models, layers, query
docs/API_CONTRACT.md   # contract v0 — the reviewable artifact
```

Repo home: git.vedur.is (GitLab — services live there; group TBD with the
aut/ut-dev team). CI: `.gitlab-ci.yml`.
