"""Single-trajectory diagnostic plots: Toda integral J and Lyapunov exponent.

Usage:
    python -m analysis.plot_diagnostics --input <run.csv> --output-dir <dir>

Produces (only for the diagnostics present in the CSV):
    1. TodaJ vs time
    2. TodaJ / J(0) vs time
    3. TodaJ - TodaJ(0) vs log time
    4. LyapunovFTLE vs time
    5. LyapunovLocal vs time
    6. aligned (separate-panel) spectral entropy, TodaJ, and Lyapunov diagnostics

Design rules honoured:
  * matplotlib only (no seaborn); the global style is never modified.
  * quantities with incompatible scales are never stacked on one unlabeled axis
    (figure 6 uses separate stacked panels that only share the time axis).
  * the alpha equilibrium estimate J_eq ~ 2*epsilon is drawn ONLY for the alpha
    model; for beta it is neither computed nor drawn (J is a control observable).
  * Eta is plotted as stored (already normalized spectral entropy).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Reuse the single metadata parser.
_VIS = os.path.join(os.path.dirname(__file__), "..", "visualization")
if _VIS not in sys.path:
    sys.path.insert(0, _VIS)
from plot_utils import get_metadata  # noqa: E402


def _save(fig, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def _title(meta: dict) -> str:
    model = meta.get("Model", "?")
    N = meta.get("N", "?")
    coeff_key = "Alpha" if model == "alpha" else "Beta"
    coeff = meta.get(coeff_key, "?")
    return f"FPUT-{model} (Yoshida4)  N={N}  {coeff_key.lower()}={coeff}"


def plot_toda(df, meta, epsilon, out_dir) -> list[Path]:
    written = []
    t = df["Time"].to_numpy()
    J = df["TodaJ"].to_numpy()
    J0 = J[0]
    model = meta.get("Model", "?")

    # 1. TodaJ vs time
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, J, lw=1.3, color="C0", label="TodaJ")
    if model == "alpha" and epsilon is not None:
        # Alpha-only equilibrium estimate J_eq ~ 2*epsilon.
        ax.axhline(2.0 * epsilon, ls="--", color="0.4", lw=1.0,
                   label=r"$J_{\rm eq}\approx 2\epsilon$ (alpha)")
    ax.set_xlabel("time  t")
    ax.set_ylabel("Toda integral  J")
    ax.set_title(_title(meta) + "\nToda integral vs time")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    ax.ticklabel_format(axis="both", style="sci", scilimits=(-3, 4), useMathText=True)
    written += _save(fig, out_dir, "diag_toda_J_vs_time")

    # 2. TodaJ / J(0) vs time
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, J / J0, lw=1.3, color="C0")
    ax.axhline(1.0, ls=":", color="0.6", lw=1.0)
    ax.set_xlabel("time  t")
    ax.set_ylabel(r"$J(t)/J(0)$")
    ax.set_title(_title(meta) + "\nnormalized Toda integral vs time")
    ax.grid(True, alpha=0.3)
    written += _save(fig, out_dir, "diag_toda_J_over_J0_vs_time")

    # 3. TodaJ - TodaJ(0) vs log time
    fig, ax = plt.subplots(figsize=(7, 4.5))
    mask = t > 0
    ax.plot(t[mask], (J - J0)[mask], lw=1.3, color="C0")
    ax.set_xscale("log")
    ax.set_xlabel("time  t (log scale)")
    ax.set_ylabel(r"$J(t) - J(0)$")
    ax.set_title(_title(meta) + "\nToda integral drift vs log time")
    ax.grid(True, which="both", alpha=0.3)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
    written += _save(fig, out_dir, "diag_toda_Jdrift_vs_logtime")
    return written


def plot_lyapunov(df, meta, out_dir) -> list[Path]:
    written = []
    t = df["Time"].to_numpy()

    # 4. LyapunovFTLE vs time
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, df["LyapunovFTLE"].to_numpy(), lw=1.3, color="C3")
    ax.set_xlabel("time  t")
    ax.set_ylabel(r"cumulative FTLE  $\lambda_{\max}(t)$")
    ax.set_title(_title(meta) + "\nfinite-time max Lyapunov exponent vs time")
    ax.grid(True, alpha=0.3)
    written += _save(fig, out_dir, "diag_lyapunov_ftle_vs_time")

    # 5. LyapunovLocal vs time
    if "LyapunovLocal" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(t, df["LyapunovLocal"].to_numpy(), lw=1.0, color="C1")
        ax.set_xlabel("time  t")
        ax.set_ylabel(r"local exponent  $\lambda_{\rm loc}(t)$")
        ax.set_title(_title(meta) + "\nlocal (per-interval) Lyapunov exponent vs time")
        ax.grid(True, alpha=0.3)
        written += _save(fig, out_dir, "diag_lyapunov_local_vs_time")
    return written


def plot_aligned(df, meta, out_dir) -> list[Path]:
    """Separate stacked panels sharing only the time axis (no mixed scales)."""
    panels = []
    if "Eta" in df.columns:
        panels.append(("Eta", r"spectral entropy $\eta$", "C2"))
    if "TodaJ" in df.columns:
        panels.append(("TodaJ", "Toda integral  J", "C0"))
    if "LyapunovFTLE" in df.columns:
        panels.append(("LyapunovFTLE", r"FTLE $\lambda_{\max}$", "C3"))
    if len(panels) < 2:
        return []

    t = df["Time"].to_numpy()
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.5, 2.4 * len(panels)),
                             sharex=True)
    for ax, (col, label, color) in zip(axes, panels):
        ax.plot(t, df[col].to_numpy(), lw=1.2, color=color)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time  t")
    axes[0].set_title(_title(meta) + "\naligned diagnostics (separate axes)")
    fig.align_ylabels(axes)
    return _save(fig, out_dir, "diag_aligned_panels")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m analysis.plot_diagnostics",
        description="Plot single-trajectory Toda-J and Lyapunov diagnostics.",
    )
    ap.add_argument("--input", required=True, help="Path to a diagnostics CSV.")
    ap.add_argument("--output-dir", required=True, help="Directory for figures.")
    args = ap.parse_args(argv)

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    meta = get_metadata(str(in_path))
    df = pd.read_csv(in_path, comment="#")
    df.columns = df.columns.str.strip()

    if "Time" not in df.columns:
        print("ERROR: CSV has no Time column.", file=sys.stderr)
        return 2

    # epsilon = H(0)/(N-1), only used for the alpha J_eq reference line.
    epsilon = None
    try:
        N = int(meta["N"])
        epsilon = float(df["TotalEnergy"].iloc[0]) / (N - 1)
    except (KeyError, ValueError, ZeroDivisionError):
        pass

    written = []
    if "TodaJ" in df.columns:
        written += plot_toda(df, meta, epsilon, out_dir)
    else:
        print("note: no TodaJ column; skipping Toda plots.")
    if "LyapunovFTLE" in df.columns:
        written += plot_lyapunov(df, meta, out_dir)
    else:
        print("note: no LyapunovFTLE column; skipping Lyapunov plots.")
    written += plot_aligned(df, meta, out_dir)

    if not written:
        print("No diagnostic columns found; nothing plotted.")
        return 1
    for p in written:
        print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
