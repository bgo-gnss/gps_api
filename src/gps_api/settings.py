"""Where the API (and the precompute job) find the product file store.

Phase-1 store = files only (plan §6): Parquet for bulk series, GeoJSON/JSON
for velocities, station catalog and model/break catalogs. Postgres joins in
a later slice; this module is the single place both the writer (precompute
job) and the reader (routers) resolve the store root, so they always meet
at the same directory.

Resolution order:

1. ``$GPS_API_STORE`` — explicit override (tests, ad-hoc runs, deployment).
2. ``$XDG_CACHE_HOME/gps_analysis`` or ``~/.cache/gps_analysis`` — the
   cache location the revamp plan assigns to precomputed analysis products
   (PLAN-postprocessing-revamp.md §3, "Self-containment on prod").

The environment is read per call (not at import) so tests can point the
running app at a temporary store without rebuilding it.
"""

import os
from pathlib import Path

STORE_ENV = "GPS_API_STORE"

#: Store subdirectories (shared vocabulary of writer and readers).
VELOCITIES_DIR = "velocities"
SERIES_DIR = "series"
MODELS_DIR = "models"
META_DIR = "meta"

#: Store file names (shared vocabulary of writer and readers).
STATIONS_FILE = "stations.geojson"
RUN_META_FILE = "run.json"
#: Operator-review deliverable of the outlier stage (design §5.1 / BGÓ Q5):
#: protected SuspectedEvent clusters as candidate steps.csv entries.
SUSPECTED_STEPS_FILE = "suspected_steps.csv"

#: Parquet schema-metadata key carrying product provenance. Lives here (not
#: in ``precompute.products``) so the series router can read it without
#: importing the precompute package (which pulls in ``gps_analysis``).
PROVENANCE_METADATA_KEY = b"gps_api_provenance"


def store_path() -> Path:
    """Resolve the product-store root directory (see module docstring)."""
    explicit = os.environ.get(STORE_ENV)
    if explicit:
        return Path(explicit).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return base / "gps_analysis"
