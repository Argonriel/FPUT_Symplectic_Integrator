# FPUT-β analysis pipeline (`analysis/`)

A clean, reproducible pipeline that summarizes the **existing** FPUT-β trajectories
produced by the 4th-order Yoshida symplectic integrator. It never runs
simulations and never modifies raw data — it only reads CSVs and writes reports
and figures.

> Scope: FPUT-**β** / **Yoshida4** only. Alpha data, Lyapunov exponents, Toda
> integrals, and threshold fitting are intentionally out of scope.

## Where the raw data live (this machine)

`data/` contains four subdirectories:

| Directory                     | Status                                           |
| ----------------------------- | ------------------------------------------------ |
| `data/yoshida_threshold_v2`   | **Canonical** long runs (`t_max ≈ 1.4e8`). Default input. |
| `data/yoshida_threshold`      | Older, short (`t_max ≈ 1e5`). Excluded from primary analysis. |
| `data/yoshida`                | Older. Excluded.                                 |
| `data/legacy`                 | Legacy/short (`t_max ≈ 1e5`). Excluded.          |

The default `--input-dir` is `data/yoshida_threshold_v2`. **Directory name is
never trusted on its own**: every file is validated by header metadata and by its
actual saved times, so short, non-β, or non-Yoshida runs are rejected with a
recorded reason even if the input is pointed elsewhere. The three older
directories are a known source of *confusable duplicates* — e.g.
`data/yoshida_threshold/beta_N1024_A10.1053.csv` has the same filename as the
canonical file but `Stride=5000, NumSegments=200` (≈1e5), so it is rejected by
`--min-saved-time` and is distinguished from the canonical run because the
duplicate key includes `stride` and `num_segments`.

## What Git ignores

`.gitignore` ignores (globally) `*.csv`, `data/`, `results/`, `__pycache__/`,
`*.log`, and `.pytest_cache/`. Consequences:

- Raw and generated **CSVs are not committed** — this is intended.
- Generated **figures and reports** land under `results/` (also ignored).
- **Test fixtures are generated in-code** (into `tmp_path`), never committed,
  precisely because `*.csv` is ignored globally.
- Source code under `analysis/` **is** committed (only `*.py[cod]` byte-code is
  ignored).

## Running the pipeline

```bash
pip install -r requirements.txt   # numpy, scipy, pandas, matplotlib, ...
python -m analysis.summarize_beta summarize-beta \
    --input-dir data/yoshida_threshold_v2 \
    --output-dir results
```

`python -m analysis summarize-beta ...` also works. All defaults are overridable:

| Flag                          | Default                      | Meaning |
| ----------------------------- | ---------------------------- | ------- |
| `--input-dir`                 | `data/yoshida_threshold_v2`  | Root searched recursively for `*.csv`. |
| `--output-dir`                | `results`                    | Where reports + `figures/` are written. |
| `--tail-fraction`             | `0.20`                       | Fraction of trailing samples for tail statistics. |
| `--min-saved-time`            | `1e8`                        | Reject runs whose actual last saved time is below this. |
| `--energy-drift-warning`      | `1e-4`                       | Flag runs whose max relative energy error exceeds this (Yoshida-4 drift is ~3e-6..5e-5, so this flags only true anomalies). |
| `--analytic-mismatch-warning` | `1e-2`                       | Flag runs whose analytic-H0 relative discrepancy exceeds this. |
| `--grid-points`               | `50`                         | Shared log-ε grid points for the collapse metric. |
| `--include-legacy`            | *(off)*                      | Do **not** exclude legacy paths. Off by default. |
| `--no-plots`                  | *(off)*                      | Skip figure generation. |

### How legacy / short runs are excluded

1. Directories named `summary`, `results`, `output(s)`, `figures`, `__pycache__`
   are skipped during discovery.
2. Any path containing a `legacy` segment is skipped (unless `--include-legacy`).
3. Each remaining file must have `Integrator == Yoshida4` and `Model == beta`,
   the required `Beta`/`Amplitude` header keys, all required columns
   (`Time, Mode1, TotalEnergy, Eta`), and an **actual** last saved time
   `≥ --min-saved-time`. Every rejection reason is recorded in
   `beta_rejected_files.csv`.

## Output directory structure

```
results/
├── beta_runs.csv              # one row per accepted, selected β run
├── beta_rejected_files.csv    # every skipped file + reason
├── beta_duplicates.csv        # duplicate groups + which member is selected
├── beta_data_quality.csv      # per-run quality flags
├── beta_collapse_metrics.csv  # per-grid-point across-N spread
└── figures/
    ├── fig1_entropy_tailmean_vs_epsilon.{png,pdf}
    ├── fig2_entropy_tailstd_vs_epsilon.{png,pdf}
    ├── fig3_mode1_fraction_vs_epsilon.{png,pdf}
    ├── fig4_max_energy_error_vs_epsilon.{png,pdf}
    ├── fig5_amplitude_vs_epsilon.{png,pdf}
    └── fig6_finite_size_collapse.{png,pdf}
```

## Definitions

### Energy density `epsilon` (the main control variable)

```
epsilon = H(0) / (N - 1)
```

`H(0)` is the **initial `TotalEnergy` value stored in the CSV** — the primary
source of truth. `N - 1` is the number of **moving interior particles** in this
solver's fixed-boundary convention.

> **Convention note.** This `epsilon` differs from the `E/N` convention used in
> some papers by a factor `N/(N-1)`. Keep this in mind before comparing absolute
> values across sources.

### Analytic cross-check `H0_analytic`

For the pure mode-1 β initial condition (verified against the solver potential):

```
z           = A**2 * sin**2(pi / (2 N))
H0_analytic = N * (z + (3/2) * beta * z**2)
```

> **Deliberate asymmetry.** `H0_analytic` sums over `N` bonds, while `epsilon`
> divides by `N - 1` moving particles. Both are intentional. The pipeline records
> `rel_analytic_energy_discrepancy = (H0_analytic - H(0)) / H(0)`; it is ~1e-15
> at low amplitude and grows to ~1e-5 as nonlinearity increases, which is
> expected.

### Tail statistics

The tail window is the last `--tail-fraction` of the saved samples. At the
default `0.20`, the tail-mean of `Eta` equals the existing `aggregate_eta.py`
convention `df["Eta"].iloc[int(0.8*len(df)):].mean()`; that logic is refactored
into `analysis/statistics.py` (`tail_start_index` / `tail_stats`) so there is a
single implementation. The **tail-mean of `Eta`** is the primary entropy
statistic (not just the final row). `Eta` is used verbatim — it already stores
**normalized** spectral entropy and is never re-normalized.

### Energy drift

```
max_abs_rel_energy_error = max_t |E(t) - E(0)| / |E(0)|
final_rel_energy_error   =      |E(last) - E(0)| / |E(0)|
```

For a healthy Yoshida-4 run both are tiny; `--energy-drift-warning` (default
`1e-4`) flags only true anomalies.

### Snapshot-time conventions

This solver writes a snapshot **before** each advance. Metadata-derived and
actual times generally differ, so both are exported:

```
metadata_last_saved_time = (NumSegments - 1) * Stride * dt   # from header
nominal_duration         =  NumSegments      * Stride * dt   # from header
first_saved_time         = df["Time"].iloc[0]                # actual
last_saved_time          = df["Time"].iloc[-1]               # actual (source of truth)
```

## Per-run quantities (`beta_runs.csv`)

`source_path`, `sha256`, `integrator`, `model`, `N`, `beta`, `amplitude`, `dt`,
`stride`, `NumSegments`, `first_saved_time`, `last_saved_time`, `n_saved_rows`,
`nominal_duration`, `metadata_last_saved_time`, `H0`, `epsilon`, `H0_analytic`,
`rel_analytic_energy_discrepancy`, `max_abs_rel_energy_error`,
`final_rel_energy_error`, `final_eta`, `eta_tail_mean`, `eta_tail_std`,
`eta_tail_min`, `eta_tail_max`, `mode1_tail_mean`, `mode1_tail_std`,
`totalenergy_tail_mean`, `totalenergy_tail_std`, `has_nonfinite`,
`selected_for_plot`, `duplicate_conflict`.

## Data quality (`beta_data_quality.csv`)

Boolean flags (plus the underlying values) that make it easy to spot problems:
`excessive_energy_drift`, `incomplete_run`, `analytic_energy_mismatch`,
`missing_entropy`, `nonfinite_values`, `suspicious_metadata`.

## Duplicate handling (`beta_duplicates.csv`)

Two runs are duplicates when they share the full physical key
`(model, N, beta, amplitude, dt, stride, num_segments)`. Duplicates are never
silently discarded — every duplicate group is reported with all member paths,
their SHA-256 hashes, whether the group is byte-identical, and which member is
selected for plotting.

**Deterministic selection rule:**

- Single-member group → that member.
- Byte-identical members → the lexicographically-first path (`conflict = False`).
- Conflicting members (same key, different hashes) → the member with the largest
  `last_saved_time` (longest completed run), tie-broken lexicographically by
  path; `conflict = True`. Conflicting runs are flagged and **excluded from the
  aggregate plots and collapse metric** (`selected_for_plot & ~duplicate_conflict`).

## Quantifying collapse (`beta_collapse_metrics.csv`) — descriptive only

1. Find the common overlapping ε interval across the selected `N`:
   `[max_i min(ε_i), min_i max(ε_i)]`.
2. Build a shared log-spaced ε grid inside that interval.
3. Interpolate each `N` curve onto the grid **only within its own data range**
   (linear in `log ε`; grid points outside a curve's range are `NaN` — never
   extrapolated).
4. At each grid point compute, across the `N` that cover it: `eta_mean`,
   `eta_std`, `eta_maxmin`, and `eta_cv` (coefficient of variation, when ≥2
   curves and mean ≠ 0).
5. Report an aggregate **RMS spread** = root-mean-square of the across-`N` std
   over the interval.

No stochasticity threshold is fitted and no universal collapse is auto-declared.
The common interval is expected to be **narrow** because the `N=4096` β runs
start at high ε and barely overlap the low-ε end of the smaller-`N` runs. On the
current canonical data the common interval is roughly `ε ∈ [3.7e-4, 5.5e-3]`.

## Reproducing the figures

The figures are a deterministic function of the accepted, selected runs; re-run
the command above. Every figure uses matplotlib only (no seaborn), a log ε axis
where appropriate, explicit axis labels, a legend keyed by `N` (ascending), and
is saved as both PNG and PDF at `dpi=300, bbox_inches="tight"`. The global
matplotlib style is never modified.

## Module layout

| Module                      | Responsibility |
| --------------------------- | -------------- |
| `analysis/metadata.py`      | Typed header parsing (reuses `visualization/plot_utils.py::get_metadata`). |
| `analysis/statistics.py`    | Tail window, `epsilon`, analytic `H0`, energy drift, per-run stats. |
| `analysis/validation.py`    | Accept/reject a candidate trajectory with a recorded reason. |
| `analysis/duplicates.py`    | Duplicate detection + deterministic selection. |
| `analysis/collapse.py`      | Common-interval interpolation (no extrapolation) + spread metrics. |
| `analysis/plotting.py`      | The six figures. |
| `analysis/summarize_beta.py`| CLI entry point tying it together. |
| `analysis/tests/`           | pytest suite (fixtures generated in-code). |

## Tests

```bash
python -m pytest analysis/tests/ -q
```
