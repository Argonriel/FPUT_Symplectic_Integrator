# FPUT_Symplectic_Integrator

## ✨ Intro
* **Symplectic Integration**: Energy-preserving algorithm crucial for long-term nonlinear dynamics.
* **Dual Solvers**: 
  * `FPUT_1024_solver.py`: CPU-optimized solver using Numba JIT (ideal for $N<1024$ and extracting spatial shapes).
  * `FPUT_4096_solver.py`: GPU-accelerated solver (ideal for massive $N$ thermodynamic limit studies).
* ** Plot **: The solvers automatically save physical parameters ($\alpha, \beta, N, dt$, etc.) within the generated `.csv` headers. Run the plotter to get visualize modal energy evolution, spectral energy heatmaps, spatial wave displacement.

## 📦 Requirements
To run this code, make sure you have the following Python libraries installed:
```bash
pip install numpy scipy pandas matplotlib seaborn numba
