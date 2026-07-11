"""Product writers for the file store (Phase-1: Parquet + GeoJSON/JSON).

Layout under the store root (:func:`gps_api.settings.store_path`)::

    stations.geojson                 StationCollection (catalog)
    velocities/<region>.geojson      VelocityCollection + provenance member
    series/<MARKER>.parquet          raw + detrended N/E/U series (bulk)
    models/<region>_breaks.json      GBIS4TS break/rate-change catalog
    meta/run.json                    provenance summary of the last run

Contract mapping (``docs/API_CONTRACT.md`` / :mod:`gps_api.schemas`):

- ``stations.geojson`` and ``velocities/*.geojson`` are built through the
  pydantic schemas (:class:`~gps_api.schemas.StationCollection`,
  :class:`~gps_api.schemas.VelocityCollection`) **before** writing, so a
  product file that exists is by construction contract-shaped. The extra
  top-level ``provenance`` member is a legal GeoJSON foreign member and is
  ignored by the schema validators on read.
- ``series/*.parquet`` backs ``GET /v1/stations/{marker}/series``
  (:class:`~gps_api.schemas.SeriesResponse`): the ``time`` column is UTC
  timestamps, displacement columns are mm, ``*_detrended`` carries the
  trajectory-model residuals. Provenance travels in the Parquet schema
  metadata (key ``gps_api_provenance``).
- ``models/*_breaks.json`` is served by ``GET /v1/models/{region}`` as a
  :class:`~gps_api.schemas.ModelResult` with ``kind="breakpoints"``; each
  entry validates as a :class:`~gps_api.schemas.BreakEntry` (schema named
  after the ``kind`` this writer emits, so the file round-trips losslessly).

Every product carries provenance: estimation method, reference frame,
software versions, ``fitted_at`` (UTC), input source.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
from pathlib import Path
from typing import Any

import gps_analysis
import pyarrow as pa
import pyarrow.parquet as pq

from gps_api import __version__, settings
from gps_api.precompute.sources import COMPONENTS, FloatArray, StationSeries
from gps_api.schemas import StationCollection, VelocityCollection

# Re-exported from settings (the shared writer/reader vocabulary) so the
# series router can read it without importing this gps_analysis-dependent
# module; kept here as the historical import location.
PROVENANCE_METADATA_KEY = settings.PROVENANCE_METADATA_KEY


@dataclasses.dataclass(frozen=True)
class Provenance:
    """Provenance stamped on every product (MATH_STANDARDS §6, plan §10.5).

    Attributes:
        method: Estimator tag of the product (``"wls"``, ``"gbis"``,
            ``"lineperiodic"`` for the detrend products, ...).
        frame: Reference frame / plate fix of the input series.
        fitted_at: When the precompute produced the product (UTC).
        source: Input data tag (``"neu:<dir>"`` / ``"synthetic:..."``).
        extra: Free-form additions (window, model, run settings, ...).
    """

    method: str
    frame: str
    fitted_at: datetime.datetime
    source: str
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready mapping, software versions included."""
        return {
            "method": self.method,
            "frame": self.frame,
            "fitted_at": _iso_z(self.fitted_at),
            "source": self.source,
            "software": {
                "gps_api": __version__,
                "gps_analysis": gps_analysis.__version__,
            },
            **self.extra,
        }


def _iso_z(moment: datetime.datetime) -> str:
    """ISO-8601 with ``Z`` suffix (contract time convention)."""
    if moment.tzinfo is None:
        raise ValueError("product timestamps must be timezone-aware UTC")
    return (
        moment.astimezone(datetime.UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return path


def write_stations_geojson(
    store: Path, catalog: StationCollection, provenance: Provenance
) -> Path:
    """Write the station catalog (validated ``StationCollection``)."""
    payload = catalog.model_dump(mode="json")
    payload["provenance"] = provenance.as_dict()
    return _write_json(store / settings.STATIONS_FILE, payload)


def write_velocities_geojson(
    store: Path, region: str, collection: VelocityCollection, provenance: Provenance
) -> Path:
    """Write one region's velocity field (validated ``VelocityCollection``)."""
    payload = collection.model_dump(mode="json")
    payload["provenance"] = provenance.as_dict()
    return _write_json(store / settings.VELOCITIES_DIR / f"{region}.geojson", payload)


def write_series_parquet(
    store: Path,
    series: StationSeries,
    detrended: FloatArray,
    times: list[datetime.datetime],
    provenance: Provenance,
) -> Path:
    """Write one station's bulk series product (raw + detrended, mm).

    Columns: ``time`` (UTC timestamp), ``north``/``east``/``up`` (raw
    displacement, mm), ``sigma_*`` (1-σ, mm; omitted when the source has
    none), ``north_detrended``/... (trajectory-model residuals, mm).
    Provenance rides in the Parquet schema metadata under
    :data:`PROVENANCE_METADATA_KEY`.
    """
    columns: dict[str, Any] = {
        "time": pa.array(times, type=pa.timestamp("ms", tz="UTC")),
    }
    for i, component in enumerate(COMPONENTS):
        columns[component] = pa.array(series.y[i], type=pa.float64())
    if series.sigma is not None:
        for i, component in enumerate(COMPONENTS):
            columns[f"sigma_{component}"] = pa.array(series.sigma[i], type=pa.float64())
    for i, component in enumerate(COMPONENTS):
        columns[f"{component}_detrended"] = pa.array(detrended[i], type=pa.float64())
    table = pa.table(columns).replace_schema_metadata(
        {PROVENANCE_METADATA_KEY: json.dumps(provenance.as_dict()).encode()}
    )
    path = store / settings.SERIES_DIR / f"{series.marker}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def write_breaks_json(
    store: Path,
    region: str,
    entries: list[dict[str, Any]],
    provenance: Provenance,
) -> Path:
    """Write the region break/rate-change catalog (GBIS4TS products).

    Each entry mirrors :class:`~gps_api.schemas.ModelFit` (``fitted_at``,
    ``parameters``) plus marker/component/model tags; see the module
    docstring for the ``/v1/models`` follow-up.
    """
    payload: dict[str, Any] = {
        "region": region,
        "kind": "breakpoints",
        "provenance": provenance.as_dict(),
        "entries": entries,
    }
    return _write_json(store / settings.MODELS_DIR / f"{region}_breaks.json", payload)


def write_run_meta(store: Path, summary: dict[str, Any]) -> Path:
    """Write the run-level provenance summary (``meta/run.json``)."""
    return _write_json(store / settings.META_DIR / settings.RUN_META_FILE, summary)
