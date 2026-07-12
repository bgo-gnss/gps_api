"""Real-data validation harness: Svartsengi Mogi ΔV(t) vs the operational model.

Closes the loop on "auto Mogi on live GNSS": runs the productized pipeline
(:func:`gps_api.precompute.run_precompute` → Mogi stage → store product) on
**real** published ``.NEU`` displacement series over one Svartsengi
inflation cycle, and quantitatively reconciles our cumulative ΔV(t) against
the operational Mogi model maintained by Vincent Drouin on
``insar.vedur.is`` (read-only — his files are never modified, and our
product is never derived from them; cross-check only, Amendment A6).

Operational-model file formats (characterized 2026-07-12 by heading the
files at ``<remote>=bgo@insar.vedur.is:/mnt/scratch/vincent/model/
svartsengi``; all read-only):

- ``inflation.list`` — ``inflationNN YYYYMMDD YYYYMMDD`` per inflation
  cycle (start/end; gaps between cycles are eruptions/dike intrusions).
- ``inv_volume_mogi.dat`` (top level) — ``YYYY-MM-DDTHH:MM value``: the
  **net** cumulative Mogi ΔV since 2023-10-25 in **10⁶ m³**, daily,
  stitched across cycles *including* co-eruptive drawdowns (hence its
  long-run deflation look). A different quantity from a per-cycle curve.
- ``inflationNN/inv_volume_mogi.dat`` — ``idx YYYYMMDD dv_m3``: cumulative
  ΔV **within the cycle** [m³], zero at cycle start, daily. **This is the
  apples-to-apples comparand** for our trailing-window ΔV(t) (both are
  zero-referenced at the cycle/window start).
- ``flowrate_ts_mogi.dat`` (top level + per cycle) — ``time m3_per_s
  sigma``: daily reservoir inflow rate (m³/s; day-1 12.38 m³/s × 86400 s
  ≈ the 1.0e6 m³ day-1 cumulative volume — unit cross-checked).
- ``flowrate_total_mogi.dat`` / ``flowrate__mogi.dat`` — ``start end mean
  lower upper``: per-cycle mean inflow [m³/s] with confidence bounds.
- ``inflationNN/dayNNN/mogi/model_best.conf`` — ``mogi svartsengi X Y
  depth_m dv_m3`` with X/Y in ISN93 / EPSG:3057 (easting/northing, m).
  Position **and depth are held fixed** (330265, 378552, 4000 m ≈
  63.869°N −22.454°E); only ΔV is estimated, by grid search
  (``run_grid.var``: −10e6 … 30e6 step 1e5 m³).
- ``inflationNN/dayNNN/cgnss_*.dat`` — ``lon lat dE dN dU sigE sigN sigU
  MARKER`` [mm]: cumulative GNSS displacement since cycle start — the
  model input. ``data_inv.conf`` reads ``gnss cgnss`` only: the
  operational ΔV(t) is **GNSS-only** too (InSAR constrains the fixed
  geometry, not the daily volume series).

Harness flow (``gps-api-validate-deformation``):

1. ``fetch`` — resolve the station cluster from deployed config (the
   ``--area`` regional area in ``station_areas.yaml``, path from
   ``postprocess.cfg [PATHS] station_areas_file``; no hand-listed
   stations), download each station's published ``.NEU`` from the CDN
   base (``postprocess.cfg [PATHS] aflogun_neu_base_url``), and copy the
   operational reference series for ``--cycle`` over SSH into a local
   fixture directory (gitignored; ``manifest.json`` records provenance).
2. ``run`` — drive the real precompute on the fixture with the region's
   trailing window set to the inflation cycle (series clipped at cycle
   end so the pipeline's newest-epoch window lands on the cycle), read
   the served product back, and reconcile: correlation / RMS / scale /
   bias of ΔV(t), depth stability vs the fixed 4 km, source-position
   agreement. Results go to ``reconciliation.json`` next to the fixture.

The env-gated test ``tests/test_validation_realdata.py`` reruns step 2
from the cached fixture (skipped when the fixture has not been fetched),
making the reconciliation a standing regression check.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from gps_analysis import local_coordinates
from gtimes.timefunc import dTimetoYearf

from gps_api.precompute.config import (
    AnalysisConfig,
    BreakpointConfig,
    DeformationConfig,
    RegionConfig,
)
from gps_api.precompute.job import SeriesLoader, run_precompute
from gps_api.precompute.sources import StationSeries, load_neu
from gps_api.schemas import DeformationResult

#: The operational Svartsengi model home (read-only; parameterizable via
#: ``--remote``). Same location the Mogi product provenance points at.
DEFAULT_REMOTE = "bgo@insar.vedur.is:/mnt/scratch/vincent/model/svartsengi"

#: Regional area (station_areas.yaml) whose stations form the cluster —
#: the live config since the dedicated svartsengi volcanic area was folded
#: into it (station_areas.yaml note, 2026-07-10).
DEFAULT_AREA = "reykjanes"

#: Default inflation cycle: 2024-08-24 → 2024-11-19, the longest completed
#: cycle of the 2023–2024 unrest with day-level operational snapshots.
DEFAULT_CYCLE = "inflation08"

#: Published .NEU reference-frame tag (``{marker}-{frame}.NEU`` on the CDN,
#: the aflogun URL convention).
DEFAULT_FRAME = "plate"

#: Environment override for the fixture directory (tests + CLI).
FIXTURE_ENV = "GPS_API_REALDATA_DIR"

#: Region key used for the validation run (product + endpoint path).
REGION_NAME = "svartsengi"

_MANIFEST = "manifest.json"
_SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=15")


def default_fixture_dir() -> Path:
    """Fixture location: ``$GPS_API_REALDATA_DIR`` or the repo test tree."""
    import os

    env = os.environ.get(FIXTURE_ENV)
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "realdata"


@dataclasses.dataclass(frozen=True)
class InflationCycle:
    """One row of the operational ``inflation.list``."""

    name: str
    start: datetime.date
    end: datetime.date

    @property
    def days(self) -> int:
        """Cycle length in days."""
        return (self.end - self.start).days


# ---------------------------------------------------------------------------
# fetch — build the local fixture (CDN .NEU + operational reference files)
# ---------------------------------------------------------------------------


def _gpsconfig() -> Any:
    """gps_parser config reader (untyped upstream — Any at the boundary)."""
    from gps_parser import ConfigParser

    return ConfigParser()


def _resolve_area_stations(parser: Any, area: str) -> tuple[str, ...]:
    """Station markers of one area from the deployed ``station_areas.yaml``.

    The file path comes from ``postprocess.cfg [PATHS] station_areas_file``
    (resolved against the gpsconfig directory when relative); the area is
    searched across every ``*_areas`` category. No stations are hand-listed
    anywhere in this harness (plan §10.4 no-hardcoding rule).
    """
    areas_file = Path(str(parser.getPostProcessDir("station_areas_file")))
    if not areas_file.is_absolute():
        areas_file = Path(str(parser.config_path)) / areas_file
    doc = yaml.safe_load(areas_file.read_text())
    if not isinstance(doc, dict):
        raise ValueError(f"{areas_file}: expected a mapping document")
    for key, section in doc.items():
        if not (str(key).endswith("_areas") and isinstance(section, dict)):
            continue
        body = section.get(area)
        if isinstance(body, dict) and body.get("stations"):
            return tuple(str(s) for s in body["stations"])
    raise KeyError(f"area {area!r} not found in any *_areas category of {areas_file}")


def _ssh_read(remote: str, command_suffix: str) -> str:
    """Run a read-only command on the remote model host and return stdout.

    ``remote`` is ``user@host:root``; ``command_suffix`` is appended after
    ``cd <root> && `` so relative paths (and globs) resolve inside the
    model directory. Read-only by construction: this harness only ever
    issues ``cat``.
    """
    host, root = remote.split(":", 1)
    proc = subprocess.run(
        ["ssh", *_SSH_OPTS, host, f"cd {root} && {command_suffix}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ssh {host} failed for {command_suffix!r}: {proc.stderr.strip()}"
        )
    return proc.stdout


def _parse_inflation_list(text: str) -> tuple[InflationCycle, ...]:
    """Parse ``inflation.list`` rows (``name YYYYMMDD YYYYMMDD``)."""
    cycles = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        cycles.append(
            InflationCycle(
                name=fields[0],
                start=datetime.datetime.strptime(fields[1], "%Y%m%d").date(),
                end=datetime.datetime.strptime(fields[2], "%Y%m%d").date(),
            )
        )
    if not cycles:
        raise ValueError("inflation.list: no cycle rows parsed")
    return tuple(cycles)


def _isn93_to_lonlat(x: float, y: float) -> tuple[float, float] | None:
    """ISN93 / EPSG:3057 easting/northing → (lon, lat), via ``cs2cs``.

    Coordinate math is not derived here (MATH_STANDARDS rule) — PROJ does
    the transform. Returns ``None`` when ``cs2cs`` is unavailable so the
    fetch degrades to a fixture without a geographic source position
    (position comparison is then skipped, never guessed).
    """
    try:
        proc = subprocess.run(
            [
                "cs2cs",
                "+init=epsg:3057",
                "+to",
                "+proj=longlat",
                "+datum=WGS84",
                "-f",
                "%.8f",
            ],
            input=f"{x} {y}\n",
            capture_output=True,
            text=True,
            check=True,
        )
        lon, lat = (float(v) for v in proc.stdout.split()[:2])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None
    return lon, lat


def fetch_fixture(
    fixture_dir: Path,
    *,
    area: str = DEFAULT_AREA,
    cycle_name: str = DEFAULT_CYCLE,
    remote: str = DEFAULT_REMOTE,
    frame: str = DEFAULT_FRAME,
) -> dict[str, Any]:
    """Build (or refresh) the real-data fixture and write its manifest.

    Everything is resolved from deployed config or the remote model
    directory: stations from the ``area`` in ``station_areas.yaml``
    (intersected with ``stations.cfg`` so the pipeline can place them),
    the CDN base from ``postprocess.cfg``, cycle dates from the remote
    ``inflation.list``. Stations whose ``.NEU`` is not published (HTTP
    404) are recorded, not fatal.
    """
    parser = _gpsconfig()
    stations = _resolve_area_stations(parser, area)
    known = {str(s) for s in parser.getStationInfo()}
    missing_cfg = tuple(s for s in stations if s not in known)
    stations = tuple(s for s in stations if s in known)
    base_url = str(parser.getPostProcessDir("aflogun_neu_base_url")).rstrip("/")

    neu_dir = fixture_dir / "neu"
    neu_dir.mkdir(parents=True, exist_ok=True)
    fetched: list[str] = []
    missing_neu: list[str] = []
    for marker in stations:
        url = f"{base_url}/timeseries/{marker}-{frame}.NEU"
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                (neu_dir / f"{marker}.NEU").write_bytes(response.read())
            fetched.append(marker)
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            missing_neu.append(marker)
        print(
            f"[{marker}] {'fetched' if marker in fetched else 'no published .NEU'}",
            flush=True,
        )

    vincent_dir = fixture_dir / "vincent"
    vincent_dir.mkdir(parents=True, exist_ok=True)
    cycles = _parse_inflation_list(_ssh_read(remote, "cat inflation.list"))
    by_name = {c.name: c for c in cycles}
    if cycle_name not in by_name:
        raise KeyError(
            f"cycle {cycle_name!r} not in remote inflation.list "
            f"(have: {', '.join(sorted(by_name))})"
        )
    cycle = by_name[cycle_name]
    (vincent_dir / "inflation.list").write_text(
        "\n".join(f"{c.name} {c.start:%Y%m%d} {c.end:%Y%m%d}" for c in cycles) + "\n"
    )
    for rel, local in (
        ("inv_volume_mogi.dat", "inv_volume_net_mogi.dat"),
        ("flowrate_ts_mogi.dat", "flowrate_ts_mogi.dat"),
        (f"{cycle.name}/inv_volume_mogi.dat", f"{cycle.name}_inv_volume_mogi.dat"),
    ):
        (vincent_dir / local).write_text(_ssh_read(remote, f"cat {rel}"))

    # Operational source geometry: the newest day snapshot's best model
    # (`mogi svartsengi X Y depth_m dv_m3`; position/depth fixed by design).
    best = _ssh_read(
        remote, f"cat {cycle.name}/day*/mogi/model_best.conf | tail -1"
    ).split()
    if len(best) != 6 or best[0] != "mogi":
        raise ValueError(f"unexpected model_best.conf shape: {' '.join(best)!r}")
    x_isn93, y_isn93, depth_m, dv_final = (float(v) for v in best[2:6])
    lonlat = _isn93_to_lonlat(x_isn93, y_isn93)

    manifest: dict[str, Any] = {
        "retrieved_at": datetime.datetime.now(datetime.UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        "area": area,
        "frame": frame,
        "neu_base_url": base_url,
        "remote": remote,
        "stations_fetched": fetched,
        "stations_missing_neu": missing_neu,
        "stations_missing_cfg": list(missing_cfg),
        "cycle": {
            "name": cycle.name,
            "start": cycle.start.isoformat(),
            "end": cycle.end.isoformat(),
        },
        "vincent_source": {
            "x_isn93": x_isn93,
            "y_isn93": y_isn93,
            "depth_m": depth_m,
            "dv_m3_final": dv_final,
            "lon": lonlat[0] if lonlat else None,
            "lat": lonlat[1] if lonlat else None,
            "geometry": "fixed (grid search over dV only)",
        },
        "files": {
            "cycle_volume": f"vincent/{cycle.name}_inv_volume_mogi.dat",
            "net_volume": "vincent/inv_volume_net_mogi.dat",
            "flowrate_ts": "vincent/flowrate_ts_mogi.dat",
        },
    }
    (fixture_dir / _MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"fixture ready under {fixture_dir}: {len(fetched)} station(s), "
        f"cycle {cycle.name} {cycle.start} → {cycle.end}",
        flush=True,
    )
    return manifest


# ---------------------------------------------------------------------------
# run — drive the real pipeline on the fixture and reconcile
# ---------------------------------------------------------------------------


def _clipped_neu_loader(neu_dir: Path, end: datetime.date) -> SeriesLoader:
    """Loader for fixture ``.NEU`` files, clipped at the cycle end.

    The pipeline's trailing window always ends at the newest epoch; the
    clip is what points that window at the historical inflation cycle —
    data preparation, not a pipeline modification.
    """
    t_max = float(
        dTimetoYearf(datetime.datetime(end.year, end.month, end.day, 23, 59, 59))
    )

    def load(marker: str) -> StationSeries:
        path = neu_dir / f"{marker}.NEU"
        if not path.is_file():
            raise FileNotFoundError(f"no {marker}.NEU under {neu_dir}")
        series = load_neu(path, marker=marker)
        mask = series.t <= t_max
        if not bool(np.any(mask)):
            raise ValueError(f"{marker}: no samples on or before {end.isoformat()}")
        return dataclasses.replace(
            series,
            t=series.t[mask],
            y=series.y[:, mask],
            sigma=None if series.sigma is None else series.sigma[:, mask],
            source=f"{series.source} (clipped <= {end.isoformat()})",
        )

    return load


def _load_cycle_volume(path: Path, start: datetime.date) -> tuple[Any, Any]:
    """Operational per-cycle ΔV series → (days-since-start, ΔV [m³])."""
    days: list[float] = []
    dv: list[float] = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        date = datetime.datetime.strptime(fields[1], "%Y%m%d").date()
        days.append(float((date - start).days))
        dv.append(float(fields[2]))
    if not days:
        raise ValueError(f"no rows parsed from {path}")
    return np.asarray(days, dtype=np.float64), np.asarray(dv, dtype=np.float64)


@dataclasses.dataclass(frozen=True)
class Reconciliation:
    """Quantitative agreement between our ΔV(t) and the operational model.

    ``scale`` is the least-squares factor a in ``ours ≈ a · operational``
    (through the origin — both series are zero at the cycle start by
    construction), the single most telling number for the depth/ΔV
    trade-off between our free-geometry fits and the fixed-4 km model.
    """

    cycle: str
    window_start: str
    window_end: str
    stations_used: tuple[str, ...]
    n_epochs: int
    n_aligned: int
    pearson_r: float
    rms_m3: float
    bias_m3: float
    scale: float
    final_ours_m3: float
    final_operational_m3: float
    final_ratio: float
    depth_mean_km: float
    depth_std_km: float
    depth_operational_km: float
    source_lon_mean: float
    source_lat_mean: float
    source_scatter_km: float
    source_offset_km: float | None
    chi2_reduced_median: float
    rms_mm_median: float
    n_stations_median: float

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready mapping (written to ``reconciliation.json``)."""
        payload = dataclasses.asdict(self)
        payload["stations_used"] = list(self.stations_used)
        return payload

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        offset = (
            f"{self.source_offset_km:.2f} km"
            if self.source_offset_km is not None
            else "n/a (no geographic operational position)"
        )
        return "\n".join(
            (
                f"cycle {self.cycle}: {self.window_start} → {self.window_end}",
                f"stations: {len(self.stations_used)}; epochs fitted: "
                f"{self.n_epochs} (aligned: {self.n_aligned})",
                f"dV(t) vs operational: r={self.pearson_r:.4f}  "
                f"scale={self.scale:.3f}  rms={self.rms_m3 / 1e6:.2f}e6 m3  "
                f"bias={self.bias_m3 / 1e6:.2f}e6 m3",
                f"final dV: ours {self.final_ours_m3 / 1e6:.2f}e6 m3, "
                f"operational {self.final_operational_m3 / 1e6:.2f}e6 m3 "
                f"(ratio {self.final_ratio:.3f})",
                f"depth: {self.depth_mean_km:.2f} ± {self.depth_std_km:.2f} km "
                f"(operational fixed {self.depth_operational_km:.1f} km)",
                f"source: ({self.source_lat_mean:.4f}N, "
                f"{self.source_lon_mean:.4f}E) ± {self.source_scatter_km:.2f} km; "
                f"offset from operational: {offset}",
                f"fit quality (medians): chi2_red={self.chi2_reduced_median:.2f}, "
                f"rms={self.rms_mm_median:.1f} mm, "
                f"stations/epoch={self.n_stations_median:.0f}",
            )
        )


def reconcile(
    result: DeformationResult,
    cycle: InflationCycle,
    op_days: Any,
    op_dv: Any,
    *,
    op_depth_m: float,
    op_lon: float | None,
    op_lat: float | None,
) -> Reconciliation:
    """Align the two ΔV(t) series and compute the agreement metrics.

    The operational series is daily; ours sits on the precompute grid.
    The daily series is linearly interpolated to our epochs (both are
    referenced to the cycle start, so no offset is removed — bias is a
    reported metric, not a fitted parameter).
    """
    t0 = datetime.datetime.combine(cycle.start, datetime.time(12), tzinfo=datetime.UTC)
    our_days = np.asarray(
        [(fit.time - t0).total_seconds() / 86400.0 for fit in result.fits],
        dtype=np.float64,
    )
    our_dv = np.asarray([fit.dv_m3 for fit in result.fits], dtype=np.float64)
    aligned = (our_days >= float(op_days.min())) & (our_days <= float(op_days.max()))
    if int(np.count_nonzero(aligned)) < 3:
        raise RuntimeError(
            f"only {int(np.count_nonzero(aligned))} of {our_days.size} fitted "
            "epochs fall inside the operational series — window misaligned?"
        )
    ours = our_dv[aligned]
    theirs = np.interp(our_days[aligned], op_days, op_dv)

    depths = np.asarray([fit.depth_km for fit in result.fits], dtype=np.float64)
    east = np.asarray([fit.east_m for fit in result.fits], dtype=np.float64)
    north = np.asarray([fit.north_m for fit in result.fits], dtype=np.float64)
    lon_mean = float(np.mean([fit.lon for fit in result.fits]))
    lat_mean = float(np.mean([fit.lat for fit in result.fits]))
    offset_km: float | None = None
    if op_lon is not None and op_lat is not None:
        de, dn = local_coordinates(lon_mean, lat_mean, op_lon, op_lat)
        offset_km = float(np.hypot(float(de), float(dn))) / 1000.0

    return Reconciliation(
        cycle=cycle.name,
        window_start=cycle.start.isoformat(),
        window_end=cycle.end.isoformat(),
        stations_used=tuple(result.stations),
        n_epochs=len(result.fits),
        n_aligned=int(ours.size),
        pearson_r=float(np.corrcoef(ours, theirs)[0, 1]),
        rms_m3=float(np.sqrt(np.mean((ours - theirs) ** 2))),
        bias_m3=float(np.mean(ours - theirs)),
        scale=float(ours @ theirs / (theirs @ theirs)),
        final_ours_m3=float(ours[-1]),
        final_operational_m3=float(theirs[-1]),
        final_ratio=float(ours[-1] / theirs[-1]),
        depth_mean_km=float(depths.mean()),
        depth_std_km=float(depths.std()),
        depth_operational_km=op_depth_m / 1000.0,
        source_lon_mean=lon_mean,
        source_lat_mean=lat_mean,
        source_scatter_km=float(np.hypot(east.std(), north.std())) / 1000.0,
        source_offset_km=offset_km,
        chi2_reduced_median=float(np.median([fit.chi2_reduced for fit in result.fits])),
        rms_mm_median=float(np.median([fit.rms_mm for fit in result.fits])),
        n_stations_median=float(np.median([fit.n_stations for fit in result.fits])),
    )


def run_validation(
    fixture_dir: Path | None = None,
    *,
    store: Path | None = None,
    step_days: float = 2.0,
    epoch_mean_days: float = 3.0,
    min_stations: int = 4,
    seed: int | None = 0,
) -> tuple[Reconciliation, DeformationResult]:
    """Rerun the real-data reconciliation from the cached fixture.

    Drives the real chain — :func:`gps_api.precompute.run_precompute` with
    the fixture ``.NEU`` loader (clipped at the cycle end) and a region
    whose ``deformation:`` window equals the inflation cycle — then reads
    the store product back through the contract schema and reconciles it
    against the operational per-cycle ΔV series.

    Args:
        fixture_dir: Fixture location (default:
            :func:`default_fixture_dir`).
        store: Store root for the run's products (default: a fresh temp
            directory).
        step_days / epoch_mean_days: Grid spacing / averaging window of
            the Mogi stage — finer than the production defaults so the
            comparison against the daily operational series is dense.
        min_stations: Minimum stations per fitted epoch.
        seed: RNG seed (Bayesian tail is off; kept for reproducibility).

    Returns:
        ``(reconciliation, our product)``.
    """
    fixture = fixture_dir or default_fixture_dir()
    manifest_path = fixture / _MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"no fixture manifest at {manifest_path} — run "
            "`gps-api-validate-deformation fetch` first (needs CDN + SSH access)"
        )
    manifest = json.loads(manifest_path.read_text())
    cycle = InflationCycle(
        name=str(manifest["cycle"]["name"]),
        start=datetime.date.fromisoformat(manifest["cycle"]["start"]),
        end=datetime.date.fromisoformat(manifest["cycle"]["end"]),
    )
    stations = tuple(str(s) for s in manifest["stations_fetched"])
    if not stations:
        raise RuntimeError(f"fixture manifest lists no fetched stations: {fixture}")

    cfg = AnalysisConfig(
        config_dir=fixture,
        analysis_yaml=manifest_path,
        regions={
            REGION_NAME: RegionConfig(
                name=REGION_NAME,
                stations=stations,
                reference_frame=f"{manifest['frame']}-fixed (.NEU product)",
                description=(
                    f"real-data validation cluster (area {manifest['area']!r})"
                ),
            )
        },
        velocity_window_years=2.0,
        velocity_method="wls",
        detrend_model="lineperiodic",
        detrend_overrides={},
        breakpoints=BreakpointConfig(
            enabled_regions=(), n_breaks=1, n_runs=100_000, t_runs=100
        ),
        deformation=DeformationConfig(
            enabled_regions=(REGION_NAME,),
            series="raw",
            window_years=cycle.days / 365.25,
            step_days=step_days,
            epoch_mean_days=epoch_mean_days,
            min_stations=min_stations,
        ),
    )
    if store is None:
        store = Path(tempfile.mkdtemp(prefix="gps_api_validation_"))
    summary = run_precompute(
        cfg,
        REGION_NAME,
        _clipped_neu_loader(fixture / "neu", cycle.end),
        store,
        source=f"neu:{fixture / 'neu'} (clipped <= {cycle.end.isoformat()})",
        detect_breaks=False,
        seed=seed,
    )
    if summary.deformation_failed is not None:
        raise RuntimeError(f"deformation stage failed: {summary.deformation_failed}")

    product_path = store / "models" / f"{REGION_NAME}_deformation.json"
    result = DeformationResult.model_validate(json.loads(product_path.read_text()))

    op_days, op_dv = _load_cycle_volume(
        fixture / str(manifest["files"]["cycle_volume"]), cycle.start
    )
    source_info = manifest["vincent_source"]
    reconciliation = reconcile(
        result,
        cycle,
        op_days,
        op_dv,
        op_depth_m=float(source_info["depth_m"]),
        op_lon=(
            float(source_info["lon"]) if source_info.get("lon") is not None else None
        ),
        op_lat=(
            float(source_info["lat"]) if source_info.get("lat") is not None else None
        ),
    )
    return reconciliation, result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gps-api-validate-deformation",
        description=(
            "Real-data validation of the Mogi deformation pipeline: fetch a "
            "Svartsengi GNSS fixture (published .NEU + the operational model "
            "reference series) and reconcile our dV(t) against it."
        ),
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        help=f"fixture directory (default: ${FIXTURE_ENV} or tests/fixtures/realdata)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser(
        "fetch", help="build the fixture (needs CDN HTTP + read-only SSH)"
    )
    fetch.add_argument("--area", default=DEFAULT_AREA, help="station_areas.yaml area")
    fetch.add_argument(
        "--cycle", default=DEFAULT_CYCLE, help="inflation cycle (inflation.list name)"
    )
    fetch.add_argument(
        "--remote", default=DEFAULT_REMOTE, help="user@host:path of the model dir"
    )
    fetch.add_argument(
        "--frame", default=DEFAULT_FRAME, help=".NEU reference-frame tag on the CDN"
    )

    run = sub.add_parser("run", help="rerun the reconciliation from the fixture")
    run.add_argument("--store", type=Path, help="store root (default: temp dir)")
    run.add_argument("--step-days", type=float, default=2.0)
    run.add_argument("--epoch-mean-days", type=float, default=3.0)
    run.add_argument("--min-stations", type=int, default=4)
    run.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point (``gps-api-validate-deformation``)."""
    args = _build_parser().parse_args(argv)
    fixture = args.fixture_dir or default_fixture_dir()
    if args.command == "fetch":
        fetch_fixture(
            fixture,
            area=args.area,
            cycle_name=args.cycle,
            remote=args.remote,
            frame=args.frame,
        )
        return 0
    reconciliation, _ = run_validation(
        fixture,
        store=args.store,
        step_days=args.step_days,
        epoch_mean_days=args.epoch_mean_days,
        min_stations=args.min_stations,
        seed=args.seed,
    )
    out = fixture / "reconciliation.json"
    out.write_text(json.dumps(reconciliation.as_dict(), indent=2) + "\n")
    print(reconciliation.summary())
    print(f"metrics written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
