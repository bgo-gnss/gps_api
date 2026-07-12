"""Configuration for the precompute job — everything via ``gps_parser``.

No-hardcoding rule (plan §10.4): every region, station list, window, model
choice and path the job uses comes from the deployed gpsconfig directory,
which ``gps_parser.ConfigParser`` resolves (``$GPS_CONFIG_PATH`` or
``~/.config/gpsconfig``). Two files feed this module:

- ``analysis.yaml`` — the analysis-lane sidecar (regions, velocity window +
  method incl. per-region ``velocity_method`` overrides, detrend model +
  overrides, break-point settings, the ``deformation:`` block gating the
  Mogi source-inversion stage, optional ``store.path`` / ``data.neu_dir``
  / ``api.max_points``). Template:
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
import math
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
    #: Per-region velocity estimator override (``velocity_method:`` key);
    #: ``None`` falls back to ``velocity.default_method``. ``"mle"`` selects
    #: the colored-noise MLE (honest σ) where the region's noise earns it —
    #: WLS stays the fleet-wide baseline (contract Amendment A5).
    velocity_method: str | None = None


@dataclasses.dataclass(frozen=True)
class BreakpointConfig:
    """GBIS4TS break-point detection settings (``breakpoints:`` block).

    ``n_runs``/``t_runs`` are the **confirm** chain lengths (production
    1e6). The triage stage (plan §10.7 two-stage screen -> confirm) is
    driven by ``triage_n_runs``: when > 0, every gated station is first
    screened with a short ``triage_n_runs`` chain and only stations whose
    trend-change posterior significance reaches ``triage_sigma`` get the
    full confirm chain; ``0`` (the default) disables triage — every gated
    station is confirmed, the pre-triage behavior. ``max_workers`` sizes
    the break-detection process pool (``null``/absent -> one worker per
    CPU core, capped at the number of station x component chains; ``0``
    -> inline serial execution).
    """

    enabled_regions: tuple[str, ...]
    n_breaks: int
    n_runs: int
    t_runs: int
    triage_n_runs: int = 0
    triage_t_runs: int = 100
    triage_sigma: float = 3.0
    max_workers: int | None = None

    def enabled_for(self, region: str) -> bool:
        """Whether break detection is configured for ``region``."""
        return region in self.enabled_regions


#: Velocity estimators the precompute job implements (Amendment A5).
VELOCITY_METHODS: tuple[str, ...] = ("wls", "mle")

#: Deformation source models the precompute job implements (Amendments A6/A7:
#: ``mogi`` = point-source ΔV(t) time series; ``okada`` = distributed-slip on
#: an operator-supplied fault/dike plane, single window).
DEFORMATION_SOURCES: tuple[str, ...] = ("mogi", "okada")

#: Okada slip directions the distributed-slip stage may estimate.
SLIP_COMPONENTS: tuple[str, ...] = ("strike_slip", "dip_slip", "opening")

#: Default log-spaced (lo, hi, count) L-curve λ scan for slip regularization.
DEFAULT_SLIP_SCAN: tuple[float, float, int] = (1.0e4, 1.0e10, 13)


@dataclasses.dataclass(frozen=True)
class OkadaPlaneConfig:
    """Operator-supplied fault/dike plane for Okada distributed slip (A7).

    Okada distributed-slip inversion **fixes the plane and inverts slip on
    it** — the plane is NOT found automatically (that is the nonlinear
    ``okada_invert`` / Bayesian lane). Because a dike/fault is event-specific,
    the operator supplies the plane per intrusion in the ``deformation.okada``
    block of ``analysis.yaml`` (config-driven, never hardcoded geometry).

    The horizontal ``origin`` is the plane **centroid** (WGS84); ``strike`` is
    the trace azimuth (deg clockwise from north, dip to its right) and ``dip``
    the down-dip angle (0 < dip ≤ 90). ``top_depth_km`` is the depth of the
    plane's shallow (up-dip) edge — the operator-friendly quantity — and is
    converted to the centroid depth Okada 1985 uses via
    :meth:`centroid_depth_km`. ``n_strike`` × ``n_dip`` tiles the plane
    (:func:`gps_analysis.discretize_fault`); ``components`` names the slip
    directions to estimate (default pure ``opening`` — a dike/sill).
    ``smoothing`` is a fixed Laplacian weight λ, or ``None`` to pick it at the
    L-curve corner over ``smoothing_scan`` (lo, hi, count log-spaced).
    ``nonneg`` imposes one-signed slip (NNLS, Jónsson et al. 2002 — the dike
    default); ``edge`` is the Laplacian boundary treatment.
    """

    origin_lon: float
    origin_lat: float
    strike: float
    dip: float
    length_km: float
    width_km: float
    top_depth_km: float
    n_strike: int
    n_dip: int
    components: tuple[str, ...] = ("opening",)
    smoothing: float | None = None
    smoothing_scan: tuple[float, float, int] = DEFAULT_SLIP_SCAN
    nonneg: bool = True
    edge: str = "zero"

    def __post_init__(self) -> None:
        if not 0.0 < self.dip <= 90.0:
            raise ValueError(
                f"deformation.okada.dip must be in (0, 90], got {self.dip}"
            )
        for key in ("length_km", "width_km"):
            if getattr(self, key) <= 0:
                raise ValueError(f"deformation.okada.{key} must be > 0")
        # top_depth_km >= 0 guarantees the discretize_fault surface-breach
        # guard (centroid_depth >= sin(dip)*width/2) can never fire mid-stage.
        if self.top_depth_km < 0.0:
            raise ValueError(
                f"deformation.okada.top_depth_km must be >= 0, got {self.top_depth_km}"
            )
        if self.n_strike < 1 or self.n_dip < 1:
            raise ValueError(
                f"deformation.okada grid must be >= 1x1, "
                f"got {self.n_strike}x{self.n_dip}"
            )
        if not self.components:
            raise ValueError(
                "deformation.okada.components must name >= 1 slip direction"
            )
        unknown = set(self.components) - set(SLIP_COMPONENTS)
        if unknown:
            raise ValueError(
                f"deformation.okada.components: unknown {sorted(unknown)}; "
                f"choose from {SLIP_COMPONENTS}"
            )
        if len(set(self.components)) != len(self.components):
            raise ValueError(
                f"deformation.okada.components has duplicates: {self.components}"
            )
        if self.smoothing is not None and self.smoothing <= 0.0:
            raise ValueError(
                "deformation.okada.smoothing must be > 0 (or omit / 'lcurve' "
                f"to select at the L-curve corner), got {self.smoothing}"
            )
        lo, hi, count = self.smoothing_scan
        if not 0.0 < lo < hi or count < 3:
            raise ValueError(
                "deformation.okada.smoothing_scan must be (lo, hi, count) with "
                f"0 < lo < hi and count >= 3, got {self.smoothing_scan}"
            )
        if self.edge not in ("zero", "free"):
            raise ValueError(
                f"deformation.okada.edge must be 'zero' or 'free', got {self.edge!r}"
            )

    def centroid_depth_km(self) -> float:
        """Plane-centroid depth = top-edge depth + sin(dip)·width/2 [km].

        The :class:`gps_analysis.OkadaSource` convention places the centroid
        at ``(x, y, −depth)``; the config's ``top_depth_km`` is the shallow
        up-dip edge, so the centroid sits half the down-dip projection deeper.
        """
        return (
            self.top_depth_km + math.sin(math.radians(self.dip)) * self.width_km / 2.0
        )


@dataclasses.dataclass(frozen=True)
class DeformationConfig:
    """Deformation-source inversion settings (``deformation:`` block).

    Config-gated exactly like break detection: the stage runs only for
    ``enabled_regions``. Two sources (``source``):

    - ``"mogi"`` (Amendment A6) — per grid epoch, station displacements
      relative to the start of a trailing ``window_years`` window are
      averaged over ``epoch_mean_days`` around each grid epoch (spaced
      ``step_days``) and inverted for one Mogi source (``mogi_invert``) — the
      ΔV(t)/depth/position time series. When ``bayes_n_runs > 0`` the newest
      epoch also gets a Bayesian posterior (``mogi_invert_bayes``; requires
      finite ``dv_bounds_m3`` priors). ``origin_lon``/``origin_lat`` pin the
      local tangent-plane frame; absent → the participating-station centroid.
    - ``"okada"`` (Amendment A7) — a single-window distributed-slip inversion
      on the operator-supplied :class:`OkadaPlaneConfig` (``okada`` field):
      net displacement over the trailing ``window_years`` window is inverted
      for smoothed slip/opening on the fixed plane
      (``discretize_fault`` → ``okada_greens`` → ``okada_invert_slip``). The
      local-frame origin is the plane centroid; the mogi-only fields
      (``depth_bounds_km``, ``dv_bounds_m3``, ``bayes_*``, ``origin_*``) are
      ignored.
    """

    enabled_regions: tuple[str, ...]
    source: str = "mogi"
    series: str = "raw"
    window_years: float = 1.0
    step_days: float = 7.0
    epoch_mean_days: float = 10.0
    min_stations: int = 3
    nu: float = 0.25
    depth_bounds_km: tuple[float, float] = (0.1, 20.0)
    dv_bounds_m3: tuple[float, float] | None = None
    origin_lon: float | None = None
    origin_lat: float | None = None
    bayes_n_runs: int = 0
    bayes_t_runs: int = 100
    #: Operator-supplied fault/dike plane — required when ``source == "okada"``
    #: (the distributed-slip stage inverts slip on a fixed plane; Amendment A7).
    okada: OkadaPlaneConfig | None = None

    def __post_init__(self) -> None:
        if self.source not in DEFORMATION_SOURCES:
            raise ValueError(
                f"deformation.source={self.source!r} — implemented sources: "
                f"{DEFORMATION_SOURCES}"
            )
        if self.source == "okada" and self.okada is None:
            raise ValueError(
                "deformation.source='okada' requires an 'okada:' fault-plane "
                "block — the plane is operator-supplied per intrusion "
                "(distributed slip fixes the plane, it is not auto-found)"
            )
        if self.series not in ("raw", "detrended"):
            raise ValueError(
                f"deformation.series={self.series!r} — must be 'raw' or 'detrended'"
            )
        for key in ("window_years", "step_days", "epoch_mean_days"):
            if getattr(self, key) <= 0:
                raise ValueError(f"deformation.{key} must be > 0")
        if self.min_stations < 2:
            raise ValueError(
                "deformation.min_stations must be >= 2 (mogi needs >= 2 "
                "stations for 4 parameters; distributed slip needs a "
                "resolvable network)"
            )
        lo, hi = self.depth_bounds_km
        if not 0 < lo < hi:
            raise ValueError(
                f"deformation.depth_bounds_km must satisfy 0 < lower < upper, "
                f"got {self.depth_bounds_km}"
            )
        if self.dv_bounds_m3 is not None and not (
            self.dv_bounds_m3[0] < self.dv_bounds_m3[1]
        ):
            raise ValueError(
                f"deformation.dv_bounds_m3 must satisfy lower < upper, "
                f"got {self.dv_bounds_m3}"
            )
        if (self.origin_lon is None) != (self.origin_lat is None):
            raise ValueError("deformation.origin needs both lon and lat (or neither)")
        if self.bayes_n_runs > 0:
            if self.dv_bounds_m3 is None:
                raise ValueError(
                    "deformation.bayes needs finite dv_bounds_m3 — uniform "
                    "priors of the Bayesian inversion cannot be unbounded"
                )
            if 16 * self.bayes_t_runs >= self.bayes_n_runs:
                raise ValueError(
                    f"deformation.bayes: n_runs ({self.bayes_n_runs}) must "
                    f"exceed the annealing span 16*t_runs "
                    f"({16 * self.bayes_t_runs})"
                )

    def enabled_for(self, region: str) -> bool:
        """Whether the deformation stage is configured for ``region``."""
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
    deformation: DeformationConfig = dataclasses.field(
        default_factory=lambda: DeformationConfig(enabled_regions=())
    )
    velocity_kappa_bounds: tuple[float, float] | None = None
    store_path: Path | None = None
    neu_dir: Path | None = None
    api_max_points: int | None = None

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

    def velocity_method_for(self, region_name: str) -> str:
        """Velocity estimator for one region (override or default).

        Raises:
            ValueError: When the resolved method is not implemented
                (Amendment A5: ``wls`` baseline, ``mle`` per region;
                ``gbis`` velocities are a later slice).
        """
        region = self.region(region_name)
        method = region.velocity_method or self.velocity_method
        if method not in VELOCITY_METHODS:
            raise ValueError(
                f"region {region_name!r}: velocity method {method!r} is not "
                f"implemented — configure one of {VELOCITY_METHODS} "
                "('gbis' velocities are a later slice)"
            )
        return method


def _as_mapping(value: object, what: str) -> dict[str, Any]:
    """Narrow a YAML node to a mapping (empty mapping when absent)."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"analysis.yaml: {what} must be a mapping, got {value!r}")
    return value


def _as_pair(value: object, what: str) -> tuple[float, float] | None:
    """Narrow a YAML node to a (lower, upper) float pair (None when absent)."""
    if value is None:
        return None
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(
            f"analysis.yaml: {what} must be a [lower, upper] pair, got {value!r}"
        )
    return float(value[0]), float(value[1])


def _parse_okada(body: dict[str, Any]) -> OkadaPlaneConfig | None:
    """Build the :class:`OkadaPlaneConfig` from a ``deformation.okada`` mapping.

    Returns ``None`` when the block is absent (disabled). The plane is
    operator-supplied and validated in :meth:`OkadaPlaneConfig.__post_init__`;
    here we only narrow YAML nodes and report missing required keys.
    """
    if not body:
        return None
    origin = _as_mapping(body.get("origin"), "deformation.okada.origin")
    required = {
        "strike",
        "dip",
        "length_km",
        "width_km",
        "top_depth_km",
        "n_strike",
        "n_dip",
    }
    missing = sorted(required - body.keys())
    if not origin or "lon" not in origin or "lat" not in origin:
        missing.append("origin.{lon,lat}")
    if missing:
        raise ValueError(f"deformation.okada missing required key(s): {missing}")

    smoothing_raw = body.get("smoothing")
    if smoothing_raw is None or smoothing_raw == "lcurve":
        smoothing: float | None = None
    else:
        smoothing = float(smoothing_raw)

    scan_raw = body.get("smoothing_scan")
    if scan_raw is None:
        scan = DEFAULT_SLIP_SCAN
    else:
        if not isinstance(scan_raw, list | tuple) or len(scan_raw) != 3:
            raise ValueError(
                "deformation.okada.smoothing_scan must be [lo, hi, count], "
                f"got {scan_raw!r}"
            )
        scan = (float(scan_raw[0]), float(scan_raw[1]), int(scan_raw[2]))

    components = tuple(str(c) for c in body.get("components") or ("opening",))
    return OkadaPlaneConfig(
        origin_lon=float(origin["lon"]),
        origin_lat=float(origin["lat"]),
        strike=float(body["strike"]),
        dip=float(body["dip"]),
        length_km=float(body["length_km"]),
        width_km=float(body["width_km"]),
        top_depth_km=float(body["top_depth_km"]),
        n_strike=int(body["n_strike"]),
        n_dip=int(body["n_dip"]),
        components=components,
        smoothing=smoothing,
        smoothing_scan=scan,
        nonneg=bool(body.get("nonneg", True)),
        edge=str(body.get("edge", "zero")),
    )


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
            velocity_method=(
                str(body["velocity_method"]) if body.get("velocity_method") else None
            ),
        )

    velocity = _as_mapping(raw.get("velocity"), "velocity")
    detrend = _as_mapping(raw.get("detrend"), "detrend")
    breaks = _as_mapping(raw.get("breakpoints"), "breakpoints")
    deformation = _as_mapping(raw.get("deformation"), "deformation")
    bayes = _as_mapping(deformation.get("bayes"), "deformation.bayes")
    origin = _as_mapping(deformation.get("origin"), "deformation.origin")
    okada_cfg = _parse_okada(_as_mapping(deformation.get("okada"), "deformation.okada"))
    store = _as_mapping(raw.get("store"), "store")
    data = _as_mapping(raw.get("data"), "data")
    api = _as_mapping(raw.get("api"), "api")

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
            # Two-stage triage -> confirm (plan §10.7): 0 keeps triage off.
            triage_n_runs=int(breaks.get("triage_n_runs", 0)),
            triage_t_runs=int(breaks.get("triage_t_runs", 100)),
            triage_sigma=float(breaks.get("triage_sigma", 3.0)),
            # Break-detection pool size; absent/null -> min(cpu_count, chains).
            max_workers=(
                int(breaks["max_workers"])
                if breaks.get("max_workers") is not None
                else None
            ),
        ),
        # Mogi deformation stage (Amendment A6): absent block → disabled.
        deformation=DeformationConfig(
            enabled_regions=tuple(
                str(r) for r in deformation.get("enabled_regions") or ()
            ),
            source=str(deformation.get("source", "mogi")),
            series=str(deformation.get("series", "raw")),
            window_years=float(deformation.get("window_years", 1.0)),
            step_days=float(deformation.get("step_days", 7.0)),
            epoch_mean_days=float(deformation.get("epoch_mean_days", 10.0)),
            min_stations=int(deformation.get("min_stations", 3)),
            nu=float(deformation.get("nu", 0.25)),
            depth_bounds_km=(
                _as_pair(
                    deformation.get("depth_bounds_km"), "deformation.depth_bounds_km"
                )
                or (0.1, 20.0)
            ),
            dv_bounds_m3=_as_pair(
                deformation.get("dv_bounds_m3"), "deformation.dv_bounds_m3"
            ),
            origin_lon=(
                float(origin["lon"]) if origin.get("lon") is not None else None
            ),
            origin_lat=(
                float(origin["lat"]) if origin.get("lat") is not None else None
            ),
            bayes_n_runs=int(bayes.get("n_runs", 0)),
            bayes_t_runs=int(bayes.get("t_runs", 100)),
            okada=okada_cfg,
        ),
        # Optional κ search bounds for method="mle" regions (Amendment A5).
        velocity_kappa_bounds=_as_pair(
            velocity.get("kappa_bounds"), "velocity.kappa_bounds"
        ),
        store_path=Path(str(store["path"])).expanduser() if store.get("path") else None,
        neu_dir=(
            Path(str(data["neu_dir"])).expanduser() if data.get("neu_dir") else None
        ),
        # Serving hint (template `api:` block): LTTB ceiling the precompute
        # records in the run provenance for the series router to honor.
        api_max_points=(
            int(api["max_points"]) if api.get("max_points") is not None else None
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
