# FPUT_Symplectic_Integrator

## ✨ Intro
* **Symplectic Integration**: Energy-preserving algorithm crucial for long-term nonlinear dynamics.
* **Dual Solvers**: 
  * `FPUT_1024_solver.py`: CPU-optimized solver using Numba JIT (ideal for $N<1024$ and extracting spatial shapes).
  * `FPUT_4096_solver.py`: GPU-accelerated solver (ideal for massive $N$).
  * `simulations_cpu/yoshida/FPUT_yoshida_solver.cpp`: 4th-order Yoshida symplectic integrator (CPU). Compile with: `g++ -O3 -march=native -o fput_yoshida FPUT_yoshida_solver.cpp`
* ** Plot **: The solvers automatically save physical parameters ($\alpha, \beta, N, dt$, etc.) within the generated `.csv` headers. Run the plotter to get visualize modal energy evolution, spectral energy heatmaps, spatial wave displacement.

## 📦 Requirements
```bash
pip install -r requirements.txt
```

## 📊 FPUT-β analysis pipeline

A reproducible pipeline in `analysis/` summarizes the **existing** FPUT-β
Yoshida4 trajectories (energy density vs. spectral entropy, energy-drift
diagnostics, finite-size-collapse metrics). It only reads raw CSVs — it never
modifies them.

```bash
python -m analysis.summarize_beta summarize-beta \
    --input-dir data/yoshida_threshold_v2 \
    --output-dir results
```

This writes `beta_runs.csv`, `beta_rejected_files.csv`, `beta_duplicates.csv`,
`beta_data_quality.csv`, `beta_collapse_metrics.csv`, and six figures (PNG + PDF)
under `results/figures/`. Raw data live in `data/yoshida_threshold_v2` (canonical
long runs; the other `data/` subdirs are older/short and excluded by default).
Generated CSVs and `results/` are gitignored by design.

Run the tests with:

```bash
python -m pytest analysis/tests/ -q
```

See [`docs/analysis_pipeline.md`](docs/analysis_pipeline.md) for full definitions
(`epsilon`, tail statistics, energy drift, analytic `H0`), the exclusion rules
for legacy/short runs, duplicate handling, and how to reproduce the plots.

## 🧭 Dynamical diagnostics (Toda integral J + Lyapunov exponent)

The canonical Yoshida-4 solver has opt-in diagnostics: the intensive fourth-order
**Toda integral J** and the finite-time **maximum Lyapunov exponent** (Benettin).
They are disabled by default — trajectories and the CSV schema are unchanged
unless a diagnostic flag is passed.

```bash
cd simulations_cpu/yoshida && make            # build solver + self-test driver
./fput_yoshida 1024 alpha 1.0 3.0 out.csv \
    --dt 0.1 --stride 200000 --nseg 500 --entropy --toda --lyapunov

python -m pytest simulations_cpu/yoshida/tests/ -q          # diagnostics tests
python -m analysis.plot_diagnostics --input out.csv --output-dir figs
```

This campaign fixes **alpha = 1** (see the convention and the exact
alpha-rescaling with η-invariance). J is an adiabatic invariant for FPUT-alpha
but only a **control observable** for FPUT-beta. Full details, formulas, indexing
(`N_paper = N_solver − 1`), CLI/CSV reference, and the production-run manifest are
in [`docs/diagnostics.md`](docs/diagnostics.md).
