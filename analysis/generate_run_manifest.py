"""Generate a PROPOSED production-run manifest (never executed here).

Two groups:

  * Main group  -- FPUT-alpha pilot at alpha = 1. Energy-density targets are
    seeded by the existing alpha = 0.25 mode-1 entropy runs after the exact
    lossless rescaling A_1 = 0.25*A_0.25, epsilon_1 = epsilon_0.25/16 (eta is
    invariant). Amplitudes are chosen to MATCH energy density across N (never
    chosen independently per N).

  * Control group -- a few missing low-amplitude N = 4096 FPUT-beta points that
    fill the current low-entropy gap. Amplitudes are chosen so their exact
    epsilon (from the beta initial-energy formula) spans ~1e-4 up to the current
    lowest sampled N=4096 beta epsilon. These are controls, NOT a new beta study.

Outputs (machine-readable) under ``analysis/manifests/``:
  run_manifest.json    -- full manifest + selection rationale (tracked; not *.csv)
  run_manifest.csv     -- flat table for the run manager (gitignored via *.csv)

Usage:
  python -m analysis.generate_run_manifest \
      [--data-dir data/yoshida_threshold_v2] [--output-dir analysis/manifests]

Nothing is run. Review the manifest before executing anything.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the frozen beta analytic energy helpers and the single metadata parser.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.statistics import analytic_h0 as beta_h0  # noqa: E402
_VIS = os.path.join(os.path.dirname(__file__), "..", "visualization")
if _VIS not in sys.path:
    sys.path.insert(0, _VIS)
from plot_utils import get_metadata  # noqa: E402

# Machine-measured per-step cost used only for ROUGH cost estimates (this
# machine, -O3). baseline ~1.5e-6 s/step at N=512; cost ~ O(N); Lyapunov ~2.3x.
_C_PER_STEP_AT_512 = 1.5e-6
_LYAP_FACTOR = 2.3

# Pilot integration parameters (shorter than the 1.4e8 production runs).
_PILOT_DT = 0.1
_PILOT_STRIDE = 200_000
_PILOT_NSEG = 500  # t_max(last saved) = 499*stride*dt = 9.98e6

# Beta control parameters: match existing N=4096 beta convention for comparability.
_BETA_DT = 0.1
_BETA_STRIDE = 2_800_000
_BETA_NSEG = 500


# --------------------------------------------------------------------------- #
# Energy <-> amplitude
# --------------------------------------------------------------------------- #
def _sine_ic_bonds(N: int) -> np.ndarray:
    """Bond displacements r_i (i=0..N-1) for the unit-amplitude mode-1 sine IC."""
    M = N - 1
    q = np.zeros(N + 1)
    q[1:M + 1] = np.sin(np.pi * np.arange(1, M + 1) / N)  # q_1..q_M, q_0=q_N=0
    return np.diff(q)  # length N


def alpha_h0(amp: float, N: int, alpha: float = 1.0) -> float:
    """Exact H(0) of the alpha mode-1 IC (zero velocity): sum r^2/2 + (alpha/3) r^3."""
    r = amp * _sine_ic_bonds(N)
    return float(np.sum(0.5 * r**2 + (alpha / 3.0) * r**3))


def amplitude_for_epsilon_alpha(epsilon: float, N: int, alpha: float = 1.0) -> float:
    """Exact amplitude for the alpha=1 mode-1 (first normal mode) IC.

    For a pure sine profile the cubic bond term sums to EXACTLY zero
    (sum_i r_i^3 = 0 by the cos^3 symmetry of the equally-spaced bond phases), so
    H(0) is purely harmonic:

        z   = A^2 sin^2(pi/(2N))
        H0  = N z           (no cubic term for the sine IC)
        eps = H0/(N-1)
        =>  A = sqrt( eps (N-1)/N ) / sin(pi/(2N))

    This is independent of alpha (the cubic term vanishes regardless).
    """
    return math.sqrt(epsilon * (N - 1) / N) / math.sin(math.pi / (2.0 * N))


def most_negative_initial_strain_alpha(amp: float, N: int) -> float:
    """Most negative initial bond strain of the alpha mode-1 IC.

    r_i = 2A sin(pi/2N) cos(pi(2i+1)/2N); the most negative is at the far edge,
    -A sin(pi/N). For alpha=1 the compression barrier is at r=-1 (V(-1)=1/6), so
    this magnitude must stay well below 1.
    """
    return -amp * math.sin(math.pi / N)


def amplitude_for_epsilon_beta(epsilon: float, N: int, beta: float = 1.0) -> float:
    """Invert epsilon = H0/(N-1) for the beta mode-1 IC.

    H0 = N (z + 1.5 beta z^2), z = A^2 sin^2(pi/2N). Solve the quadratic in z,
    then A = sqrt(z)/sin(pi/2N).
    """
    s = math.sin(math.pi / (2.0 * N))
    rhs = epsilon * (N - 1) / N  # = z + 1.5 beta z^2
    # 1.5 beta z^2 + z - rhs = 0
    disc = 1.0 + 6.0 * beta * rhs
    z = (-1.0 + math.sqrt(disc)) / (3.0 * beta)
    return math.sqrt(z) / s


# --------------------------------------------------------------------------- #
# Seed regimes from existing alpha=0.25 data (rescaled to alpha=1)
# --------------------------------------------------------------------------- #
def load_alpha025_rescaled(data_dir: str, N: int):
    """Return sorted arrays (epsilon_1, eta_tail) from existing alpha=0.25 runs."""
    eps1, eta = [], []
    for f in sorted(glob.glob(os.path.join(data_dir, f"alpha_N{N}_*.csv"))):
        meta = get_metadata(f)
        if meta.get("Model") != "alpha":
            continue
        df = pd.read_csv(f, comment="#")
        df.columns = df.columns.str.strip()
        if "Eta" not in df.columns:
            continue
        h0 = float(df["TotalEnergy"].iloc[0])
        eps1.append(h0 / (N - 1) / 16.0)  # rescale alpha=0.25 -> alpha=1
        eta.append(float(df["Eta"].iloc[int(0.8 * len(df)):].mean()))
    order = np.argsort(eps1)
    return np.array(eps1)[order], np.array(eta)[order]


def eta_at(epsilon1: float, eps_grid: np.ndarray, eta_grid: np.ndarray):
    """Interpolate eta at epsilon1 within the data range; None if extrapolating."""
    if eps_grid.size == 0 or epsilon1 < eps_grid[0] or epsilon1 > eps_grid[-1]:
        return None
    return float(np.interp(epsilon1, eps_grid, eta_grid))


def est_cost_seconds(N: int, stride: int, nseg: int, lyapunov: bool) -> float:
    steps = stride * nseg
    return steps * _C_PER_STEP_AT_512 * (N / 512.0) * (_LYAP_FACTOR if lyapunov else 1.0)


# --------------------------------------------------------------------------- #
# Manifest construction
# --------------------------------------------------------------------------- #
def build_alpha_pilot(data_dir: str):
    """Alpha=1 pilot rows + rationale."""
    sizes = [512, 1024, 2048]
    seed = {N: load_alpha025_rescaled(data_dir, N) for N in sizes}

    # RECALIBRATED matched energy-density targets (same epsilon across N), chosen
    # to span three regimes VISIBLE within the pilot's nominal duration (1e7).
    # The earlier draft grid (~3e-6..2e-4) was all far below thermalization at
    # 1e7 (paper alpha=1 laws: Td ~ 0.04 eps^-2.33, Teq ~ 0.54 eps^-2.75, so even
    # eps=2e-4 gives Teq ~ 8e9), which would show a flat J everywhere. The bands
    # below are the paper's random-IC reference values; the mode-1 IC used here
    # may shift them, which is exactly what the smoke test calibrates.
    targets = [3.0e-5, 1.0e-4, 8.0e-4, 3.0e-3]
    target_role = {
        3.0e-5: "flat / recurrence (J ~ constant; entropy weak, J/lambda matter)",
        1.0e-4: "low / weak-chaos (second low anchor)",
        8.0e-4: "onset (J rising, not yet equilibrated at 1e7)",
        3.0e-3: "thermalized (J approaches ~2 eps within 1e7)",
    }
    barrier_strain = -1.0  # alpha=1 compression barrier r=-1, V(-1)=1/6

    rows = []
    for N in sizes:
        eps_grid, eta_grid = seed[N]
        A_tested_max = (amplitude_for_epsilon_alpha(eps_grid[-1], N, 1.0)
                        if eps_grid.size else None)
        for eps in targets:
            A = amplitude_for_epsilon_alpha(eps, N, alpha=1.0)
            r_min0 = most_negative_initial_strain_alpha(A, N)
            # Separation from the escape barrier at r=-1 (fraction of the way there).
            escape_fraction = r_min0 / barrier_strain  # in (0,1); <1 means safe
            eta_guess = eta_at(eps, eps_grid, eta_grid)  # seed grid is already in eps_1
            in_range = eta_guess is not None

            # Build a SINGLE notes list so no note is silently dropped: seed-eta
            # context, an amplitude-vs-tested-range warning, and the escape status.
            note = []
            if in_range:
                note.append(f"seed eta_1~{eta_guess:.2f}")
            else:
                note.append("epsilon outside existing alpha=0.25 seed range (extrapolated)")
            if A_tested_max is not None and A > 1.2 * A_tested_max:
                note.append("amplitude above tested range; check alpha well stability")
            note.append("escape-safe" if escape_fraction < 0.5 else "CHECK escape strain")

            rows.append({
                "group": "alpha_pilot",
                "N": N,
                "model": "alpha",
                "value": 1.0,
                "target_epsilon": eps,
                "amplitude": round(A, 8),
                "dt": _PILOT_DT,
                "stride": _PILOT_STRIDE,
                "nseg": _PILOT_NSEG,
                "last_saved_time": (_PILOT_NSEG - 1) * _PILOT_STRIDE * _PILOT_DT,
                "nominal_duration": _PILOT_NSEG * _PILOT_STRIDE * _PILOT_DT,
                "most_negative_initial_strain": round(r_min0, 6),
                "escape_barrier_strain": barrier_strain,
                "escape_fraction": round(escape_fraction, 4),
                "diagnostics": "entropy,toda,lyapunov",
                "expected_output": f"alpha_pilot/alpha_N{N}_eps{eps:.2e}.csv",
                "est_cost_seconds": round(est_cost_seconds(N, _PILOT_STRIDE, _PILOT_NSEG, True), 1),
                "expected_regime": target_role[eps],
                "notes": "; ".join(note),
            })

    rationale = {
        "convention": "alpha fixed to 1; mode-1 (first normal mode) sine IC. Cubic "
                      "term contributes 0 to H(0) for a pure sine, so "
                      "A = sqrt(eps (N-1)/N)/sin(pi/2N) (exact).",
        "sizes": sizes,
        "matched_epsilon_targets": targets,
        "recalibration": "targets raised to span flat(3e-5) -> onset(8e-4) -> "
                         "thermalized(3e-3) within nominal 1e7; the prior grid was "
                         "all sub-thermalization at 1e7.",
        "paper_reference_laws_alpha1": "Td ~ 0.04 eps^-2.33, Teq ~ 0.54 eps^-2.75 "
                                       "(random IC); mode-1 IC may shift these.",
        "escape_note": "alpha=1 compression barrier at r=-1 (V=1/6); most negative "
                       "INITIAL strain ~ -A sin(pi/N) ~ -2 sqrt(eps); the run-time "
                       "min_bond_strain telemetry reports the dynamic extreme.",
        "seed_ranges_epsilon1": {
            str(N): [float(seed[N][0][0]), float(seed[N][0][-1])] if seed[N][0].size else []
            for N in sizes
        },
        "N4096_excluded_from_pilot": "excluded per instruction unless cost/need justifies.",
        "cost_model": f"steps * {_C_PER_STEP_AT_512:g} s * (N/512) * {_LYAP_FACTOR} (lyapunov); "
                      "machine-dependent rough estimate.",
    }
    return rows, rationale


def build_beta_controls(data_dir: str):
    """Missing low-amplitude N=4096 beta control rows + rationale."""
    N = 4096
    # Current lowest sampled epsilon for N=4096 beta (from existing data).
    existing = sorted(glob.glob(os.path.join(data_dir, f"beta_N{N}_*.csv")))
    eps_min_existing = None
    for f in existing:
        df = pd.read_csv(f, comment="#")
        df.columns = df.columns.str.strip()
        eps = float(df["TotalEnergy"].iloc[0]) / (N - 1)
        eps_min_existing = eps if eps_min_existing is None else min(eps_min_existing, eps)

    # Span ~1e-4 up to the current lowest sampled epsilon (fills the low gap).
    hi = eps_min_existing if eps_min_existing else 3.68e-4
    targets = list(np.geomspace(1.0e-4, hi, 5))

    rows = []
    for eps in targets:
        A = amplitude_for_epsilon_beta(eps, N, beta=1.0)
        # Cross-check epsilon with the frozen beta analytic H0 helper.
        h0 = beta_h0(A, N, 1.0)
        eps_check = h0 / (N - 1)
        rows.append({
            "group": "beta_control",
            "N": N,
            "model": "beta",
            "value": 1.0,
            "target_epsilon": float(eps),
            "epsilon_check": float(eps_check),
            "amplitude": round(A, 6),
            "dt": _BETA_DT,
            "stride": _BETA_STRIDE,
            "nseg": _BETA_NSEG,
            "last_saved_time": (_BETA_NSEG - 1) * _BETA_STRIDE * _BETA_DT,
            "nominal_duration": _BETA_NSEG * _BETA_STRIDE * _BETA_DT,
            # Toda is ~free; Lyapunov at N=4096 is expensive -> propose toda only,
            # lyapunov optional (flagged).
            "diagnostics": "entropy,toda",
            "expected_output": f"beta_control/beta_N{N}_eps{eps:.2e}.csv",
            "est_cost_seconds": round(est_cost_seconds(N, _BETA_STRIDE, _BETA_NSEG, False), 1),
            "expected_regime": "control: fills low-entropy gap (eta not assumed)",
            "notes": "control point, not a new beta study; lyapunov optional "
                     f"(+{int((_LYAP_FACTOR-1)*100)}% cost at N=4096)",
        })

    rationale = {
        "purpose": "Fill the current N=4096 beta low-entropy gap (eta only reaches "
                   "~0.75 at the lowest sampled epsilon).",
        "current_lowest_sampled_epsilon": eps_min_existing,
        "epsilon_span": [1.0e-4, float(hi)],
        "empirical_crossover_note": "the completed beta analysis found an empirical "
            "fixed-observation-time entropy crossover ~ N^-0.33 (NOT a critical energy "
            "density); these controls probe below the current N=4096 sampling.",
        "amplitude_formula": "H0 = N (z + 1.5 beta z^2), z = A^2 sin^2(pi/2N); "
                             "epsilon = H0/(N-1); inverted exactly for A.",
        "eta_not_assumed": "the realized eta for each control is not predicted in advance.",
        "diagnostics_affordability": "Toda J is ~free per snapshot; the Lyapunov "
            "tangent roughly doubles per-step cost, which is expensive at N=4096, so "
            "it is proposed as optional.",
    }
    return rows, rationale


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m analysis.generate_run_manifest")
    ap.add_argument("--data-dir", default="data/yoshida_threshold_v2")
    ap.add_argument("--output-dir", default="analysis/manifests")
    args = ap.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha_rows, alpha_rationale = build_alpha_pilot(args.data_dir)
    beta_rows, beta_rationale = build_beta_controls(args.data_dir)
    all_rows = alpha_rows + beta_rows

    manifest = {
        "description": "PROPOSED production-run manifest for the Toda-J / Lyapunov "
                       "diagnostics campaign. NOT executed automatically.",
        "solver": "simulations_cpu/yoshida/fput_yoshida",
        "generated_from": args.data_dir,
        "alpha_pilot": {"rationale": alpha_rationale, "runs": alpha_rows},
        "beta_controls": {"rationale": beta_rationale, "runs": beta_rows},
        "total_runs": len(all_rows),
        "total_est_cost_seconds": round(sum(r["est_cost_seconds"] for r in all_rows), 1),
    }

    json_path = out_dir / "run_manifest.json"
    json_path.write_text(json.dumps(manifest, indent=2))
    csv_path = out_dir / "run_manifest.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}  ({len(all_rows)} proposed runs; NOT executed)")
    print(f"total estimated cost: {manifest['total_est_cost_seconds']/3600:.1f} core-hours "
          "(rough, machine-dependent)")
    print("\nalpha pilot (alpha=1), matched energy densities:")
    print(pd.DataFrame(alpha_rows)[["N", "target_epsilon", "amplitude",
                                    "est_cost_seconds", "expected_regime"]].to_string(index=False))
    print("\nbeta controls (N=4096):")
    print(pd.DataFrame(beta_rows)[["target_epsilon", "epsilon_check", "amplitude",
                                   "est_cost_seconds"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
