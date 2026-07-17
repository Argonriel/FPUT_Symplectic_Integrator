"""Short numerical validation harness for the upgraded Yoshida solver.

Runs only SHORT simulations (no production T=1.4e8 jobs). Reports:
  - total-energy drift;
  - J(t) behaviour and exact-vs-quadratic J at low energy;
  - cumulative/local Lyapunov behaviour;
  - runtime overhead of each diagnostic;
  - the alpha=0.25 -> alpha=1 rescaling regression (eta invariance).

Usage:
  python run_validation.py --binary ./fput_yoshida [--existing-alpha025 <csv>]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


def run(binary, N, model, value, amp, out, *flags):
    cmd = [str(binary), str(N), model, str(value), str(amp), str(out), *[str(f) for f in flags]]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"solver failed: {proc.stderr}")
    df = pd.read_csv(out, comment="#")
    df.columns = df.columns.str.strip()
    return df, dt


def rel_energy_drift(df):
    e = df["TotalEnergy"].to_numpy()
    return float(np.max(np.abs(e - e[0]) / abs(e[0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="./fput_yoshida")
    ap.add_argument("--existing-alpha025", default=None,
                    help="Path to an existing alpha=0.25 CSV for the rescaling check.")
    args = ap.parse_args()
    binary = Path(args.binary).resolve()
    tmp = Path(tempfile.mkdtemp(prefix="fput_valid_"))
    print(f"binary : {binary}")
    print(f"scratch: {tmp}\n")

    # 1. Energy drift (moderate alpha=1) -------------------------------------
    df, _ = run(binary, 256, "alpha", 1.0, 4.0, tmp / "drift.csv",
                "--dt", "0.1", "--stride", "2000", "--nseg", "50", "--toda")
    print("[1] Energy drift (alpha=1, N=256, A=4):")
    print(f"    max |dE/E0| = {rel_energy_drift(df):.3e}  (Yoshida-4 expected ~1e-6..1e-5)")

    # 2. Exact vs quadratic J at low energy ----------------------------------
    print("\n[2] Exact vs quadratic J (alpha=1, N=128), amplitude scan:")
    for A in (0.5, 1.0, 2.0):
        d, _ = run(binary, 128, "alpha", 1.0, A, tmp / f"jq_{A}.csv",
                   "--dt", "0.1", "--stride", "500", "--nseg", "20", "--toda-debug")
        j, jq = d["TodaJ"].to_numpy(), d["TodaJ_quad"].to_numpy()
        eps = d["TotalEnergy"].iloc[0] / (128 - 1)
        rel = np.max(np.abs(j - jq)) / abs(j[0])
        Jdrift = (j.max() - j.min()) / abs(j[0])
        print(f"    A={A:>4}: eps={eps:.3e}  max|J-Jquad|/|J0|={rel:.2e}  "
              f"J(0)={j[0]:.4e}  J range/|J0|={Jdrift:.2e}")

    # 3. Lyapunov convergence + local fluctuations ---------------------------
    d, _ = run(binary, 128, "beta", 1.0, 20.0, tmp / "lyap.csv",
               "--dt", "0.1", "--stride", "1000", "--nseg", "40",
               "--lyapunov", "--lyap-renorm-steps", "20", "--entropy", "--toda")
    ftle = d["LyapunovFTLE"].to_numpy()
    loc = d["LyapunovLocal"].to_numpy()[1:]
    print("\n[3] Lyapunov (beta=1, N=128, A=20):")
    print(f"    FTLE: early={ftle[1]:.4e} -> late={ftle[-1]:.4e}  (cumulative)")
    print(f"    Local exponent: mean={loc.mean():.4e} std={loc.std():.4e} "
          f"(fluctuates around the cumulative value)")
    print(f"    energy drift with tangent on: {rel_energy_drift(d):.3e}")

    # 4. Runtime overhead ----------------------------------------------------
    print("\n[4] Runtime overhead (alpha=1, N=512, 200 snapshots x 2000 steps):")
    base = dict(N=512, stride="2000", nseg="200", dt="0.1")
    _, t_base = run(binary, 512, "alpha", 1.0, 4.0, tmp / "o0.csv",
                    "--dt", "0.1", "--stride", "2000", "--nseg", "200")
    _, t_toda = run(binary, 512, "alpha", 1.0, 4.0, tmp / "o1.csv",
                    "--dt", "0.1", "--stride", "2000", "--nseg", "200", "--toda")
    _, t_lyap = run(binary, 512, "alpha", 1.0, 4.0, tmp / "o2.csv",
                    "--dt", "0.1", "--stride", "2000", "--nseg", "200", "--lyapunov")
    _, t_both = run(binary, 512, "alpha", 1.0, 4.0, tmp / "o3.csv",
                    "--dt", "0.1", "--stride", "2000", "--nseg", "200", "--toda", "--lyapunov")
    print(f"    baseline           : {t_base:.3f}s")
    print(f"    --toda             : {t_toda:.3f}s  (+{100*(t_toda/t_base-1):.0f}%)")
    print(f"    --lyapunov         : {t_lyap:.3f}s  (+{100*(t_lyap/t_base-1):.0f}%)")
    print(f"    --toda --lyapunov  : {t_both:.3f}s  (+{100*(t_both/t_base-1):.0f}%)")

    # 5. Rescaling regression: eta invariance --------------------------------
    print("\n[5] Rescaling regression (eta invariance under y=alpha*q):")
    # alpha=0.25 at A vs alpha=1 at 0.25A, identical dt/stride/nseg.
    A = 1.0
    stride, nseg = "200000", "8"
    d025, _ = run(binary, 512, "alpha", 0.25, A, tmp / "r025.csv",
                  "--dt", "0.1", "--stride", stride, "--nseg", nseg, "--entropy")
    d1, _ = run(binary, 512, "alpha", 1.0, 0.25 * A, tmp / "r1.csv",
                "--dt", "0.1", "--stride", stride, "--nseg", nseg, "--entropy")
    eta_diff = np.max(np.abs(d025["Eta"].to_numpy() - d1["Eta"].to_numpy()))
    print(f"    alpha=0.25 A={A} vs alpha=1 A={0.25*A}:  max|eta_diff| = {eta_diff:.3e} "
          f"(should be ~FP noise)")

    if args.existing_alpha025:
        ex = pd.read_csv(args.existing_alpha025, comment="#")
        ex.columns = ex.columns.str.strip()
        # Reproduce the existing run's first few snapshots with the new solver.
        d_new, _ = run(binary, 512, "alpha", 0.25, 1.0, tmp / "repro.csv",
                       "--dt", "0.1", "--stride", "2800000", "--nseg", "3", "--entropy")
        n = min(len(d_new), len(ex))
        eta_repro = np.max(np.abs(d_new["Eta"].to_numpy()[:n] - ex["Eta"].to_numpy()[:n]))
        print(f"    new alpha=0.25 reproduces existing file eta (first {n} rows): "
              f"max|diff| = {eta_repro:.3e}")

    print("\nValidation complete.")


if __name__ == "__main__":
    sys.exit(main())
