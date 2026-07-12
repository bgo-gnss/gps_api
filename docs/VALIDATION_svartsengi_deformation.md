# Real-data validation: Svartsengi Mogi ΔV(t) vs the operational model

**Date:** 2026-07-12 · **Branch:** `validate-deformation-realdata` ·
**Pipeline:** `gps_api.precompute.deformation` (Amendment A6) driven end-to-end
on real published GNSS, reconciled against the operational Svartsengi Mogi
model (V. Drouin, `insar.vedur.is:/mnt/scratch/vincent/model/svartsengi/`,
read-only).

**Verdict up front: the productized GNSS-only Mogi pipeline tracks the
operational model.** Over inflation cycle 08 (2024-08-24 → 2024-11-19) the two
cumulative ΔV(t) series correlate at **r = 0.993**; our free-geometry inversion
independently recovers the source **0.32 km** from the operationally fixed
position and at **3.7 ± 1.0 km** depth vs the fixed 4.0 km — from nothing but
`stations.cfg` coordinates and published `.NEU` series. Amplitudes agree to a
**0.80–0.85 factor** that is quantitatively explained by the depth–ΔV
trade-off, not by any unit/sign/frame error. One genuine robustness bug was
found and fixed along the way (see §5) — without it the real-data run was
garbage (r = 0.20), which is exactly why synthetic-only validation was not
enough.

---

## 1. The operational model's files (characterized, not assumed)

All formats determined by heading the files over read-only SSH; nothing of
Vincent's was modified, and our product is never derived from his files
(cross-check only).

| File | Format | Quantity |
|---|---|---|
| `inflation.list` | `inflationNN YYYYMMDD YYYYMMDD` | Cycle registry: 12 inflation cycles 2023-10-25 → today; gaps between cycles are eruptions/dike intrusions |
| `inv_volume_mogi.dat` (top level) | `YYYY-MM-DDTHH:MM v` | **Net** cumulative ΔV since 2023-10-25 in **10⁶ m³**, daily, stitched across cycles *including* co-eruptive drawdowns |
| `inflationNN/inv_volume_mogi.dat` | `idx YYYYMMDD dv` | Cumulative ΔV **within the cycle** [m³], zero at cycle start, daily |
| `flowrate_ts_mogi.dat` | `time flow sigma` | Daily reservoir inflow [m³/s] |
| `flowrate_total_mogi.dat`, `flowrate__mogi.dat` | `start end mean lo hi` | Per-cycle mean inflow [m³/s] + confidence bounds |
| `inflationNN/dayNNN/mogi/model_best.conf` | `mogi svartsengi X Y depth dV` | Best model per day snapshot; X/Y in ISN93 (EPSG:3057), depth m, ΔV m³ |
| `inflationNN/dayNNN/cgnss_*.dat` | `lon lat dE dN dU σE σN σU MARKER` | Cumulative GNSS displacement since cycle start [mm] — the model input |
| `conf_pred_intervals.var` | 4 numbers | Variance/df inputs of the interval machinery |

**Unit cross-checks.** inflation01 day 26 = 1.71 × 10⁷ m³ matches the top-level
step +17.1 (10⁶ m³); flow day 1 of inflation01 (12.38 m³/s × 86 400 s ≈
1.07 × 10⁶ m³) matches the day-1 cumulative volume; inflation08's top-level
segment −47.3 → −27.6 (= +19.7 × 10⁶ m³) equals the per-cycle final 1.97 × 10⁷ m³
exactly.

**Model design (from `dayNNN/mogi/`):** position (ISN93 330265, 378552 ≈
63.8690° N, −22.4540° E) **and depth (4000 m) are held fixed** for every day of
every cycle; only ΔV is estimated, by grid search (`run_grid.var`: −10 × 10⁶ →
30 × 10⁶ m³, step 10⁵ m³). `data_inv.conf` reads `gnss cgnss` only — **the
operational ΔV(t) is GNSS-only too** (InSAR informs the fixed geometry, not
the daily series). His inflation08 input set is 26 stations, all inside the
`reykjanes` area our config resolves.

### Which series is the correct comparand (resolving the earlier confusion)

The earlier port flagged the top-level `inv_volume_mogi.dat` as "a deflation
curve, a different quantity" — confirmed and resolved: the top-level file is
the **net** stitched series across all cycles including co-eruptive drawdowns
(hence its long-run decline; the 2023-11-10 Grindavík dike alone steps it
10.7 → −67.1, i.e. −78 × 10⁶ m³). Our product is cumulative ΔV(t) relative to a
trailing-window start, so the apples-to-apples comparand is the **per-cycle**
`inflationNN/inv_volume_mogi.dat` [m³] — both series are zero-referenced at the
cycle/window start. (The flow-rate series would require differencing ours and
was not used.)

## 2. Setup

- **Cycle / window:** `inflation08`, 2024-08-24 → 2024-11-19 (87 days) — the
  longest completed cycle of the 2023–2024 unrest. Our `.NEU` input is clipped
  at the cycle end so the pipeline's trailing window (`window_years` =
  87/365.25) lands exactly on the cycle; grid `step_days = 2`,
  `epoch_mean_days = 3`.
- **Stations:** resolved from deployed config only — the `reykjanes` regional
  area of `station_areas.yaml` (path from `postprocess.cfg [PATHS]
  station_areas_file`), 46 markers, all with published
  `{marker}-plate.NEU` at the CDN base (`aflogun_neu_base_url`). 39 enter the
  epoch fits (SUND's `.NEU` is header-only upstream; six others lack
  reference-window coverage and are excluded by the stage, recorded in
  provenance). No station or path is hand-listed anywhere in the harness.
- **Chain:** real `run_precompute` (fit/detrend → WLS velocity → Mogi stage) →
  store product `models/svartsengi_deformation.json` → read back through
  `gps_api.schemas.DeformationResult` → served by
  `GET /v1/deformation/svartsengi` (asserted in the gated test).
- **Input parity check:** our window-mean displacement extraction was compared
  directly against his day-87 `cgnss` input, e.g. THOB ours (E,N,U) =
  (72.7, −64.8, 248.7) mm vs his (70, −68.5, 264.3) mm; SENG (39.1, 77.8,
  244.5) vs (37.2, 76.9, 260.1) — agreement at the few-mm level expected from
  the 3-day averaging difference. Data ingestion is correct.

## 3. Reconciliation numbers (2026-07-12 baseline)

40 of 43 grid epochs fitted (3 rejected by the interior-solution guard, §5);
all 40 align with his daily series.

| Metric | Ours vs operational |
|---|---|
| Pearson r of ΔV(t) | **0.9934** |
| Scale (ours ≈ a·his, through origin) | **0.797** |
| RMS difference | 2.67 × 10⁶ m³ |
| Mean bias | −2.54 × 10⁶ m³ |
| Final ΔV (2024-11-18 vs his interp.) | 16.37 vs 19.30 × 10⁶ m³ (**ratio 0.848**) |
| Depth (free) | **3.71 ± 1.05 km** vs fixed 4.0 km |
| Source position (mean of 40 free fits) | 63.8718° N, −22.4523° E — **0.32 km** from his fixed source; scatter 1.67 km |
| Fit quality (medians) | χ²_red 13.3, residual RMS 9.1 mm, 38.5 stations/epoch |

### Where and why they diverge (expected differences, not bugs)

1. **Depth–ΔV trade-off (the 0.80–0.85 amplitude factor).** His depth is fixed
   at 4.0 km; ours floats and settles at 3.5–3.7 km through the well-resolved
   part of the cycle. A shallower Mogi source needs less ΔV for the same
   surface field — (3.71/4.0)² ≈ 0.86 matches the observed final ratio 0.848.
   With geometry fixed at his values, the amplitudes would close further; we
   deliberately do **not** do that — the free-geometry recovery is the point
   of the validation.
2. **Co-eruptive early window (the −2.5 × 10⁶ m³ bias).** The cycle opens
   during the Aug 22 – Sep 5 eruption. Both models go negative; his
   fixed-geometry series bottoms at −1.7 × 10⁶ m³, ours dips to −5.7 × 10⁶ m³
   (free depth wanders 4–9 km absorbing the co-eruptive, non-Mogi field).
   From mid-September the two rise in lockstep.
3. **Station sets and reference epochs.** 39 stations (ours) vs 26 (his);
   our reference is a 3-day mean at the window start vs his day-0 sample;
   his ΔV is quantized to the 10⁵ m³ grid step. All minor at these scales.
4. **χ²_red ≫ 1 for both.** A single Mogi source under-fits a real field with
   mm-level formal σ (plant subsidence at the geothermal field, dike-adjacent
   stations); our residual RMS of ~9 mm against 100–300 mm signals is healthy.

## 4. Repeatable harness

```bash
# one-time fixture build (CDN HTTP + read-only SSH to insar.vedur.is):
uv run gps-api-validate-deformation fetch          # --area/--cycle/--remote/--frame
# rerun the reconciliation any time (no network needed):
uv run gps-api-validate-deformation run            # writes reconciliation.json
uv run pytest tests/test_validation_realdata.py    # gated regression check
```

- Module: `src/gps_api/validation/realdata.py` (formats documented in its
  docstring). Fixture: `tests/fixtures/realdata/` — **gitignored** (~4.5 MB
  real data + operational reference; `manifest.json` records provenance);
  the `fetch` subcommand is the rebuild script.
- The pytest module is **skipped automatically** when the fixture is absent,
  and asserts regression guards derived from the baseline above with generous
  margins (r > 0.97, scale/final ratio in (0.5, 1.6), depth 2–8 km stable to
  <2 km, source within 5 km) plus the served endpoint. The margins catch sign
  flips, unit slips, frame errors and epoch misalignment — they are **not**
  tuned agreement targets.

## 5. Genuine pipeline bug found and fixed (this branch)

**Symptom (first real-data run):** r = 0.195, scale = −0.58, source scatter
±28 km, χ²_red rising monotonically to 838 — while the same data hand-fed to
`mogi_invert` at the final epoch recovered the source cleanly.

**Root cause:** the per-epoch inversion warm-started *only* from the previous
epoch's optimum. On real (non-ideal) fields the problem is non-convex: one
epoch converging into a pathological corner (depth pinned at its 0.1 km bound)
trapped **every subsequent epoch** in that basin — a failure mode synthetic
Mogi-ramp data can never expose.

**Fix (`precompute/deformation.py`):**

1. `_invert_epoch` — multi-start: every epoch is inverted from both the warm
   start and the `mogi_invert` cold (network-footprint) start; lowest reduced
   χ² wins. No estimator change — both candidates come from the same
   `gps_analysis.mogi_invert`.
2. `_is_interior` — candidates pinned at a finite parameter bound are
   rejected (a bound-riding NLLS optimum is not a credible source; on real
   data it is a phantom far/deep source absorbing common-mode noise, e.g.
   +889 × 10⁶ m³ at the 20 km depth bound, 92 km east). An epoch with no
   interior optimum is skipped and counted (3 of 43 here: 2024-09-01/-07/-13,
   during/just after the eruption) — recorded in provenance, never silent.

Unit regression tests: `test_invert_epoch_recovers_from_pathological_warm_start`
and `test_invert_epoch_rejects_bound_pinned_optima` in
`tests/test_deformation.py`. No result was tuned toward the operational
model: the fix restores basic optimizer hygiene, and the remaining
0.80-factor and early-cycle differences are reported above as what they are.

## 6. What this validates — and what it does not

**The inversion was genuinely free.** The harness fed no priors from the
operational model: `depth_bounds_km` = default (0.1, 20.0) km (wide open, not
around 4 km), `dv_bounds_m3` = None (ΔV unbounded), no `origin_lon/lat` (the
tangent frame is the participating-station centroid), and the cold-start initial
guess (`gps_analysis._mogi_start`) is derived entirely from the observed field —
uplift-weighted centroid for position, median station→centroid distance for
depth, the peak-uplift relation for ΔV. Evidence of real freedom: depth wandered
to 3.71 ± 1.05 km (not pinned to 4.0), source scattered 1.67 km, ΔV landed at
16.4e6 (operational 19.7e6). A biased/pinned inversion would show ~0 scatter and
depth ≈ 4.0. It *disagreed* on depth — that is independence, not alignment.

**But this is a method/code cross-validation on shared data, not an
independent-data validation.** Both inversions fit GNSS from the *same station
network* (we read the `.NEU` products; the operational model reads its `cgnss`),
with the *same Mogi model*. So:
- It **does** confirm our forward model + inversion machinery is *correct* —
  independently coded, it recovers the same source position (0.32 km) and the
  same surface field (r=0.99) as an established operational implementation.
- It **does not** independently confirm the geophysical source: two correct Mogi
  fits to the same data *should* agree, so a high r is partly built in, and a
  shared model inadequacy would not show up.
- The depth/ΔV **difference** (3.71 free vs 4.0 fixed; ratio 0.85) is the
  informative part: the classic **depth–volume trade-off** — the surface cannot
  cleanly separate depth from ΔV, so both fit at different points on the same
  trade-off curve (see the vector comparison in `gps_plot/examples`).

**Independent confirmation requires the joint GPS+InSAR lane** (`geo_dataread`
InSAR groundwork): InSAR brings genuinely independent observations — a different
line-of-sight geometry and dense spatial coverage — that break the depth–ΔV
trade-off and constrain the source without relying on the same GNSS.

---

*Baseline run 2026-07-12, gps_api `validate-deformation-realdata`. Rebuild the
fixture and rerun §4 to refresh; update the test docstring baseline if the
fixture cycle changes.*
