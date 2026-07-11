"""Station catalog and per-station series (store-wired).

Contract split (skjalftalisa lesson): ``GET /stations`` is the small,
cacheable catalog; ``GET /stations/{marker}/series`` is the on-demand,
potentially large payload with downsampling support.

Both endpoints follow the ``velocities`` pattern: read the file the
precompute job wrote, re-validate through the pydantic schema on the way
out (a hand-edited or truncated product cannot leak a malformed payload),
drop the ``provenance`` foreign member from GeoJSON payloads.

Series specifics:

- ``series/<MARKER>.parquet`` carries raw *and* detrended columns; the
  ``detrended`` query parameter picks which set is served (the observation
  ``sigma_*`` columns apply to both — detrending shifts the model, not the
  measurement uncertainty).
- ``max_points`` triggers LTTB downsampling (:mod:`gps_api.downsample`,
  contract Decisions #2). The store may also carry a server-side ceiling
  (``analysis.yaml`` ``api.max_points``, recorded by the precompute job in
  ``meta/run.json``): it acts as the default target when the client sends
  no ``max_points`` and clamps the client's value when it does.
"""

import json
from datetime import UTC, datetime
from typing import Annotated, Any

import numpy as np
import pyarrow.parquet as pq
from fastapi import APIRouter, HTTPException, Query
from fastapi import Path as PathParam

from gps_api import settings
from gps_api.downsample import lttb_indices
from gps_api.schemas import SeriesResponse, StationCollection

router = APIRouter(tags=["stations"])

#: Station markers are plain identifiers — no path characters reach the store.
MARKER_PATTERN = r"^[A-Za-z0-9_-]+$"

_COMPONENTS = ("north", "east", "up")


def _as_utc(moment: datetime) -> datetime:
    """Interpret naive query datetimes as UTC (all contract times are UTC)."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def _store_max_points() -> int | None:
    """Server-side LTTB ceiling recorded by the precompute run, if any.

    The precompute job copies ``analysis.yaml`` ``api.max_points`` into
    ``meta/run.json`` (single-region and fleet runs alike); a store without
    run metadata — or without the key — has no ceiling.
    """
    meta_path = settings.store_path() / settings.META_DIR / settings.RUN_META_FILE
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (ValueError, OSError):
        return None
    api = meta.get("api") if isinstance(meta, dict) else None
    value = api.get("max_points") if isinstance(api, dict) else None
    return value if isinstance(value, int) and value >= 2 else None


@router.get("/stations", response_model=StationCollection)
def list_stations() -> StationCollection:
    """Station catalog as a GeoJSON FeatureCollection (small, cacheable)."""
    path = settings.store_path() / settings.STATIONS_FILE
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="no station catalog in the store — run gps-api-precompute",
        )
    try:
        return StationCollection.model_validate(json.loads(path.read_text()))
    except (ValueError, OSError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"corrupt station catalog {path.name!r}: {exc}",
        ) from exc


@router.get("/stations/{marker}/series", response_model=SeriesResponse)
def station_series(
    marker: Annotated[str, PathParam(pattern=MARKER_PATTERN)],
    start: datetime | None = None,
    end: datetime | None = None,
    max_points: Annotated[
        int | None,
        Query(
            ge=2,
            description=(
                "target point count; the server downsamples visually "
                "faithfully (LTTB — peaks, offsets and trend shape survive)"
            ),
        ),
    ] = None,
    detrended: bool = True,
) -> SeriesResponse:
    """N/E/U displacement series for one station (on-demand, downsampleable)."""
    path = settings.store_path() / settings.SERIES_DIR / f"{marker}.parquet"
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"no series product for station {marker!r}",
        )
    try:
        table = pq.read_table(path)
        metadata: dict[bytes, bytes] = dict(table.schema.metadata or {})
        provenance_raw = metadata[settings.PROVENANCE_METADATA_KEY]
        provenance: dict[str, Any] = json.loads(provenance_raw)
        frame = str(provenance["frame"])
        columns = set(table.column_names)
        wanted = {"time", *_COMPONENTS}
        if detrended:
            wanted |= {f"{c}_detrended" for c in _COMPONENTS}
        if not wanted <= columns:
            raise ValueError(f"missing columns {sorted(wanted - columns)}")
        times: list[datetime] = list(table.column("time").to_pylist())
        suffix = "_detrended" if detrended else ""
        values = np.asarray(
            [table.column(f"{c}{suffix}").to_pylist() for c in _COMPONENTS],
            dtype=np.float64,
        )
        has_sigma = all(f"sigma_{c}" in columns for c in _COMPONENTS)
        sigmas = (
            np.asarray(
                [table.column(f"sigma_{c}").to_pylist() for c in _COMPONENTS],
                dtype=np.float64,
            )
            if has_sigma
            else None
        )
    except (KeyError, ValueError, OSError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"corrupt series product {path.name!r}: {exc}",
        ) from exc

    # Window filter (before downsampling — LTTB targets the served window).
    seconds = np.asarray([t.timestamp() for t in times], dtype=np.float64)
    mask = np.ones(seconds.shape, dtype=bool)
    if start is not None:
        mask &= seconds >= _as_utc(start).timestamp()
    if end is not None:
        mask &= seconds <= _as_utc(end).timestamp()
    keep = np.flatnonzero(mask)
    seconds = seconds[keep]
    values = values[:, keep]
    if sigmas is not None:
        sigmas = sigmas[:, keep]
    times = [times[int(i)] for i in keep]

    # LTTB target: client intent clamped by the store's configured ceiling.
    ceiling = _store_max_points()
    target = max_points
    if ceiling is not None:
        target = ceiling if target is None else min(target, ceiling)
    if target is not None and target < seconds.size:
        idx = lttb_indices(seconds, values, target)
        values = values[:, idx]
        if sigmas is not None:
            sigmas = sigmas[:, idx]
        times = [times[int(i)] for i in idx]

    # Row order fixed by _COMPONENTS = (north, east, up).
    return SeriesResponse(
        marker=marker,
        frame=frame,
        detrended=detrended,
        time=times,
        north=values[0].tolist(),
        east=values[1].tolist(),
        up=values[2].tolist(),
        sigma_north=sigmas[0].tolist() if sigmas is not None else None,
        sigma_east=sigmas[1].tolist() if sigmas is not None else None,
        sigma_up=sigmas[2].tolist() if sigmas is not None else None,
    )
