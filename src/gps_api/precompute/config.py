"""Configuration for the precompute job — everything via ``gps_parser``.

No-hardcoding rule (plan §10.4): every region, station list, window, model
choice and path the job uses comes from the deployed gpsconfig directory,
which ``gps_parser.ConfigParser`` resolves (``$GPS_CONFIG_PATH`` or
``~/.config/gpsconfig``). Two files feed this module:

- ``analysis.yaml`` — the analysis-lane sidecar (regions, velocity window +
  method, detrend model + overrides, break-point settings, optional
  ``store.path`` / ``data.neu_dir``). Template:
  ``gpslibrary/config-templates/analysis-lane/analysis.yaml``; it deploys
  through ``gps-config-data`` like every other cfg.
- ``stations.cfg`` — station coordinates (``latitude``/``longitude``/
  ``height``) and display names, read through
  ``ConfigParser.getStationInfo`` so inline-comment stripping and duplicate
  handling stay in one place.

``gps_parser`` ships no type information; its objects are handled as ``Any``
at this boundary and coerced to concrete types immediately.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


def _gpsconfig() -> Any:
    """Instantiate the gps_parser config reader (untyped upstream)."""
    from gps_parser import ConfigParser

    return ConfigParser()


@dataclasses.dataclass(frozen=True)
class StationMeta:
    """Coordinates + display name of one station (from ``stations.cfg``)."""

    marker: str
    lon: float
    lat: float
    height: float | None = None
    name: str | None = None


@dataclasses.dataclass(frozen=True)
class RegionConfig:
    """One region grouping from ``analysis.yaml``."""

    name: str
    stations: tuple[str, ...]
    reference_frame: str
    description: str | None = None


@dataclasses.dataclass(frozen=True)
class BreakpointConfig:
    """GBIS4TS break-point detection settings (``breakpoints:`` block)."""

    enabled_regions: tuple[str, ...]
    n_breaks: int
    n_runs: int
    t_runs: int

    def enabled_for(self, region: str) -> bool:
        """Whether break detection is configured for ``region``."""
        return region in self.enabled_regions


@dataclasses.dataclass(frozen=True)
class AnalysisConfig:
    """Everything the precompute job reads from configuration."""

    config_dir: Path
    analysis_yaml: Path
    regions: dict[str, RegionConfig]
    velocity_window_years: float
    velocity_method: str
    detrend_model: str
    detrend_overrides: dict[str, str]
    breakpoints: BreakpointConfig
    store_path: Path | None = None
    neu_dir: Path | None = None

    def region(self, name: str) -> RegionConfig:
        """Return one region or raise with the configured alternatives."""
        try:
            return self.regions[name]
        except KeyError:
            known = ", ".join(sorted(self.regions)) or "<none>"
            raise KeyError(
                f"region {name!r} is not configured in {self.analysis_yaml} "
                f"(configured regions: {known})"
            ) from None

    def detrend_model_for(self, marker: str) -> str:
        """Trajectory model name for one station (override or default)."""
        return self.detrend_overrides.get(marker, self.detrend_model)


def _as_mapping(value: object, what: str) -> dict[str, Any]:
    """Narrow a YAML node to a mapping (empty mapping when absent)."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"analysis.yaml: {what} must be a mapping, got {value!r}")
    return value


def load_analysis_config(analysis_yaml: Path | None = None) -> AnalysisConfig:
    """Load the analysis-lane configuration.

    Args:
        analysis_yaml: Explicit path to an ``analysis.yaml``; default is
            ``<gpsconfig dir>/analysis.yaml`` where the gpsconfig directory
            is resolved by ``gps_parser`` (``$GPS_CONFIG_PATH`` or
            ``~/.config/gpsconfig``).

    Raises:
        FileNotFoundError: When the sidecar is not deployed — the message
            points at the template so the fix is actionable.
        ValueError: On a structurally invalid sidecar (no regions, region
            without stations, bad node types).
    """
    config_dir = Path(str(_gpsconfig().config_path))
    yaml_path = analysis_yaml or config_dir / "analysis.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(
            f"analysis config not found: {yaml_path} — deploy the analysis-lane "
            "sidecar (template: gpslibrary/config-templates/analysis-lane/"
            "analysis.yaml) or pass --config/analysis_yaml explicitly"
        )
    raw = _as_mapping(yaml.safe_load(yaml_path.read_text()), "document")

    regions_raw = _as_mapping(raw.get("regions"), "regions")
    if not regions_raw:
        raise ValueError(f"analysis.yaml: no regions configured in {yaml_path}")
    regions: dict[str, RegionConfig] = {}
    for name, body_raw in regions_raw.items():
        body = _as_mapping(body_raw, f"regions.{name}")
        stations = tuple(str(s) for s in body.get("stations") or ())
        if not stations:
            raise ValueError(f"analysis.yaml: region {name!r} lists no stations")
        regions[str(name)] = RegionConfig(
            name=str(name),
            stations=stations,
            reference_frame=str(body.get("default_reference_frame", "ITRF2014")),
            description=(str(body["description"]) if body.get("description") else None),
        )

    velocity = _as_mapping(raw.get("velocity"), "velocity")
    detrend = _as_mapping(raw.get("detrend"), "detrend")
    breaks = _as_mapping(raw.get("breakpoints"), "breakpoints")
    store = _as_mapping(raw.get("store"), "store")
    data = _as_mapping(raw.get("data"), "data")

    overrides_raw = _as_mapping(detrend.get("overrides"), "detrend.overrides")
    overrides: dict[str, str] = {}
    for marker, override in overrides_raw.items():
        body = _as_mapping(override, f"detrend.overrides.{marker}")
        overrides[str(marker)] = str(body.get("model", ""))

    return AnalysisConfig(
        config_dir=config_dir,
        analysis_yaml=yaml_path,
        regions=regions,
        velocity_window_years=float(velocity.get("default_window_years", 2.0)),
        velocity_method=str(velocity.get("default_method", "wls")),
        detrend_model=str(detrend.get("default_model", "lineperiodic")),
        detrend_overrides=overrides,
        breakpoints=BreakpointConfig(
            enabled_regions=tuple(str(r) for r in breaks.get("enabled_regions") or ()),
            n_breaks=int(breaks.get("n_breaks_default", 1)),
            n_runs=int(breaks.get("n_runs", 1_000_000)),
            t_runs=int(breaks.get("t_runs", 1000)),
        ),
        store_path=Path(str(store["path"])).expanduser() if store.get("path") else None,
        neu_dir=(
            Path(str(data["neu_dir"])).expanduser() if data.get("neu_dir") else None
        ),
    )


def load_station_meta(markers: tuple[str, ...] | list[str]) -> dict[str, StationMeta]:
    """Coordinates for ``markers`` from ``stations.cfg`` via gps_parser.

    Raises:
        KeyError: When a marker has no section in ``stations.cfg``.
        ValueError: When a station section carries no usable
            latitude/longitude.
    """
    parser = _gpsconfig()
    known = {str(s) for s in parser.getStationInfo()}
    meta: dict[str, StationMeta] = {}
    for marker in markers:
        if marker not in known:
            raise KeyError(
                f"station {marker!r} has no section in stations.cfg "
                f"({parser.get_stations_config_path()})"
            )
        # getStationInfo(marker) wraps the section as {"station": {...}}.
        info = dict(dict(parser.getStationInfo(marker))["station"])
        try:
            lat = float(info["latitude"])
            lon = float(info["longitude"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                f"station {marker!r}: stations.cfg carries no usable "
                "latitude/longitude — cannot place it on the map"
            ) from None
        height_raw = info.get("height")
        meta[marker] = StationMeta(
            marker=marker,
            lon=lon,
            lat=lat,
            height=float(height_raw) if height_raw not in (None, "") else None,
            name=str(info["station_name"]) if info.get("station_name") else None,
        )
    return meta
