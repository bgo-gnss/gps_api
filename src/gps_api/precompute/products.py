"""Product writers for the file store (Phase-1: Parquet + GeoJSON/JSON).

Layout under the store root (:func:`gps_api.settings.store_path`)::

    stations.geojson                 StationCollection (catalog)
    velocities/<region>.geojson      VelocityCollection + provenance member
    series/<MARKER>.parquet          raw + detrended N/E/U series (bulk)
    models/<region>_breaks.json      GBIS4TS break/rate-change catalog
    models/<region>_deformation.json Mogi ΔV(t) source time series (A6)
    models/<region>_slip.json        Okada distributed-slip distribution (A7)
    meta/run.json                    provenance summary of the last run
    meta/suspected_steps.csv         operator-review step candidates (outlier stage)

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

import csv
import dataclasses
import datetime
import json
from pathlib import Path
from typing import Any

import gps_analysis
import pyarrow as pa
import pyarrow.parquet as pq

from gps_api import __version__, settings
from gps_api.precompute.outliers import SUSPECTED_STEPS_COLUMNS, StationOutliers
from gps_api.precompute.sources import COMPONENTS, FloatArray, StationSeries
from gps_api.schemas import (
    DeformationResult,
    SlipDistributionResult,
    StationCollection,
    VelocityCollection,
)

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


def _unlink_if_exists(path: Path) -> None:
    """Remove a stale sibling product (store hygiene; missing_ok)."""
    path.unlink(missing_ok=True)


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
    outliers: StationOutliers | None = None,
) -> Path:
    """Write one station's bulk series product (raw + detrended, mm).

    Columns: ``time`` (UTC timestamp), ``north``/``east``/``up`` (raw
    displacement, mm), ``sigma_*`` (1-σ, mm; omitted when the source has
    none), ``north_detrended``/... (trajectory-model residuals, mm).
    Provenance rides in the Parquet schema metadata under
    :data:`PROVENANCE_METADATA_KEY`.

    When the outlier stage ran (``outliers`` given), the **additive** flag
    columns of design §5.2 are appended — the raw columns above stay
    byte-identical to a no-outlier run (requirement 3: non-destructive):
    ``<component>_outlier`` (bool, final per-component flags),
    ``<component>_outlier_reason`` (uint8 ``REASON_*`` bitmask),
    ``<component>_outlier_protected`` (uint8 ``PROTECT_*`` bitmask) and
    ``outlier_epoch`` (bool, union over components — the serving
    convenience the ``clean`` API parameter drops on). Absent columns mean
    the product predates the feature (or the stage was off/aborted-free —
    the provenance ``outliers`` object says which).
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
    if outliers is not None:
        for i, component in enumerate(COMPONENTS):
            columns[f"{component}_outlier"] = pa.array(
                outliers.flags[i], type=pa.bool_()
            )
            columns[f"{component}_outlier_reason"] = pa.array(
                outliers.reasons[i], type=pa.uint8()
            )
            columns[f"{component}_outlier_protected"] = pa.array(
                outliers.protected[i], type=pa.uint8()
            )
        columns["outlier_epoch"] = pa.array(outliers.union_flags, type=pa.bool_())
    table = pa.table(columns).replace_schema_metadata(
        {PROVENANCE_METADATA_KEY: json.dumps(provenance.as_dict()).encode()}
    )
    path = store / settings.SERIES_DIR / f"{series.marker}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def write_suspected_steps_csv(store: Path, rows: list[dict[str, Any]]) -> Path:
    """Write the operator-review step-candidate file (design §5.1 / BGÓ Q5).

    ``meta/suspected_steps.csv`` — the protected ``SuspectedEvent``
    clusters of every station the outlier stage processed, as candidate
    ``steps.csv`` entries / suspected-icing hints for visual assessment.
    Written whenever the stage ran (header-only when nothing was
    protected, so "empty file" reads as "stage ran, nothing suspected",
    never as "stage skipped").
    """
    path = store / settings.META_DIR / settings.SUSPECTED_STEPS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUSPECTED_STEPS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
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


def write_deformation_json(
    store: Path,
    region: str,
    result: DeformationResult,
    provenance: Provenance,
) -> Path:
    """Write one region's Mogi ΔV(t) deformation product (Amendment A6).

    Validated :class:`~gps_api.schemas.DeformationResult` (built by
    :func:`gps_api.precompute.deformation.compute_mogi_series`), served by
    ``GET /v1/deformation/{region}``. The structured provenance is stamped
    into the payload's ``provenance`` member (a schema field here, unlike
    the GeoJSON foreign member of the velocity products).

    A region configures ``source: mogi`` XOR ``source: okada``; the endpoint
    serves whichever product file is present, so any stale Okada sibling from
    a previous ``source: okada`` run is removed here (store hygiene — a flipped
    source must never leave a shadowing product behind).
    """
    payload = result.model_dump(mode="json")
    payload["provenance"] = provenance.as_dict()
    _unlink_if_exists(store / settings.MODELS_DIR / f"{region}_slip.json")
    return _write_json(
        store / settings.MODELS_DIR / f"{region}_deformation.json", payload
    )


def write_slip_json(
    store: Path,
    region: str,
    result: SlipDistributionResult,
    provenance: Provenance,
) -> Path:
    """Write one region's Okada distributed-slip product (Amendment A7).

    Validated :class:`~gps_api.schemas.SlipDistributionResult` (built by
    :func:`gps_api.precompute.slip.compute_slip_distribution`), served by
    ``GET /v1/deformation/{region}`` alongside the Mogi product (discriminated
    by ``source_type``). The structured provenance is stamped into the
    payload's ``provenance`` member (a schema field, like the Mogi product).

    Any stale Mogi sibling from a previous ``source: mogi`` run is removed
    here — the endpoint serves whichever product file is present, so a flipped
    source must not leave a shadowing product behind (store hygiene).
    """
    payload = result.model_dump(mode="json")
    payload["provenance"] = provenance.as_dict()
    _unlink_if_exists(store / settings.MODELS_DIR / f"{region}_deformation.json")
    return _write_json(store / settings.MODELS_DIR / f"{region}_slip.json", payload)


def write_run_meta(store: Path, summary: dict[str, Any]) -> Path:
    """Write the run-level provenance summary (``meta/run.json``)."""
    return _write_json(store / settings.META_DIR / settings.RUN_META_FILE, summary)
