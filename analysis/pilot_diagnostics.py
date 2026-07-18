"""Analysis + plotting for the completed FPUT-alpha=1 diagnostics pilot.

Read-only: consumes the frozen pilot trajectories in ``data/alpha_pilot_v1/`` and
writes figures (PNG+PDF) and a summary table into a gitignored directory
(default ``figures/alpha_pilot_v1/``). Never runs a simulation, never edits raw
CSVs or the manifest.

Run identity comes from the manifest (``group == "alpha_pilot"``) matched to each
CSV by *metadata* (model, N, amplitude), not by filename.

Usage:
    python -m analysis.pilot_diagnostics \
        --data-dir data/alpha_pilot_v1 \
        --manifest analysis/manifests/run_manifest.json \
        --output-dir figures/alpha_pilot_v1

Conventions (stated in the generated report):
  * epsilon_actual = TotalEnergy[0]/(N-1) is used for every normalization/ratio.
  * Log-time plots omit only the Time==0 sample from the drawn curve; the t=0 row
    is retained for J(0), initial values, and all summary statistics.
  * Tail statistics use the final 20% of rows = the last 100 of 500. Reported
    standard deviations are TEMPORAL variation over the tail window, not
    statistical uncertainty across realizations.
  * nominal integrated duration = 1.0e7 ; last saved time = 9.98e6 (kept distinct).
  * J = 2*epsilon is the alpha *equilibrium estimate* (not an exact theorem, not a
    beta result). The mode-1 IC has J(0) ~ 3*epsilon > 2*epsilon.
  * matplotlib only; no seaborn; global styles are not modified. N is encoded by
    both color and line/marker style (never color alone).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend (not a style change)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Reuse the single metadata parser.
_VIS = os.path.join(os.path.dirname(__file__), "..", "visualization")
if _VIS not in sys.path:
    sys.path.insert(0, _VIS)
from plot_utils import get_metadata  # noqa: E402

# Deterministic ordering.
EPS_ORDER = [3e-5, 1e-4, 8e-4, 3e-3]
N_ORDER = [512, 1024, 2048]
REQUIRED_COLUMNS = ["Time", "TotalEnergy", "TodaJ", "Eta", "LyapunovFTLE", "LyapunovLocal"]
EXPECT_COMMIT = "4a66fec"
TAIL_FRACTION = 0.20
NROWS = 500

# Per-N visual style: (color, linestyle, marker) so N is never color-only.
N_STYLE = {
    512:  ("C0", "-", "o"),
    1024: ("C1", "--", "s"),
    2048: ("C3", "-.", "^"),
}
SUPTITLE = ("FPUT-$\\alpha$  ($\\alpha=1$, mode-1 initial condition)  —  "
            "nominal duration $10^{7}$ (last saved $9.98\\times10^{6}$)")


# --------------------------------------------------------------------------- #
# Discovery & validation
# --------------------------------------------------------------------------- #
@dataclass
class Run:
    N: int
    eps_target: float
    amplitude: float
    csv: Path
    meta: dict
    df: pd.DataFrame
    eps_actual: float = 0.0
    eps_rel_disc: float = 0.0
    min_bond_strain: float = float("nan")
    tail_stats: dict = field(default_factory=dict)


class ValidationError(RuntimeError):
    pass


def _match_csv_by_metadata(row, csv_metas, amp_tol=1e-4):
    """Find the single CSV whose metadata matches a manifest row (model/N/amp)."""
    N = int(row["N"]); amp = float(row["amplitude"])
    hits = []
    for path, meta in csv_metas.items():
        if meta.get("Model", "").lower() != "alpha":
            continue
        if int(float(meta.get("N", -1))) != N:
            continue
        if "Amplitude" not in meta:
            continue
        if abs(float(meta["Amplitude"]) - amp) <= amp_tol:
            hits.append(path)
    return hits


def _read_min_bond_from_log(csv: Path) -> float:
    """Most-negative bond strain from the run's batch log (Done! line), if present."""
    log = csv.with_suffix(".log")
    if not log.exists():
        return float("nan")
    val = float("nan")
    for line in log.read_text().splitlines():
        if "min_bond_strain=" in line:
            try:
                val = float(line.split("min_bond_strain=")[1].split()[0].rstrip("|").strip())
            except ValueError:
                pass
    return val


def discover_and_validate(data_dir: Path, manifest_path: Path) -> list[Run]:
    manifest = json.load(open(manifest_path))
    rows = [r for r in manifest["alpha_pilot"]["runs"] if r.get("group") == "alpha_pilot"]
    if len(rows) != 12:
        raise ValidationError(f"expected 12 alpha_pilot manifest rows, got {len(rows)}")

    csvs = sorted(data_dir.glob("*.csv"))
    csv_metas = {p: get_metadata(str(p)) for p in csvs}

    runs: list[Run] = []
    used: set[Path] = set()
    for row in rows:
        N = int(row["N"]); eps_t = float(row["target_epsilon"])
        if row.get("model") != "alpha" or float(row.get("value")) != 1.0:
            raise ValidationError(f"manifest row not alpha/value=1: N={N} eps={eps_t}")
        if N not in N_ORDER:
            raise ValidationError(f"unexpected N={N}")
        if not any(abs(eps_t - e) < 1e-12 for e in EPS_ORDER):
            raise ValidationError(f"unexpected target epsilon={eps_t}")
        hits = _match_csv_by_metadata(row, csv_metas)
        hits = [h for h in hits if h not in used]
        if len(hits) != 1:
            raise ValidationError(
                f"manifest row N={N} A={row['amplitude']} matched {len(hits)} CSVs (need exactly 1)")
        csv = hits[0]; used.add(csv)
        meta = csv_metas[csv]
        df = pd.read_csv(csv, comment="#"); df.columns = df.columns.str.strip()

        # --- assertions ---
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValidationError(f"{csv.name}: missing columns {missing}")
        if len(df) != NROWS:
            raise ValidationError(f"{csv.name}: {len(df)} rows (expected {NROWS})")
        for c in REQUIRED_COLUMNS:
            if not np.all(np.isfinite(df[c].to_numpy())):
                raise ValidationError(f"{csv.name}: non-finite in column {c}")
        t = df["Time"].to_numpy()
        if abs(t[0]) > 1e-9:
            raise ValidationError(f"{csv.name}: first saved time != 0 ({t[0]})")
        if abs(t[-1] - 9.98e6) > 1.0:
            raise ValidationError(f"{csv.name}: last saved time != 9.98e6 ({t[-1]})")
        if meta.get("SolverGitCommit") != EXPECT_COMMIT:
            raise ValidationError(f"{csv.name}: SolverGitCommit={meta.get('SolverGitCommit')} != {EXPECT_COMMIT}")
        if meta.get("SolverGitDirty") != "0":
            raise ValidationError(f"{csv.name}: SolverGitDirty={meta.get('SolverGitDirty')} != 0")

        eps_actual = float(df["TotalEnergy"].iloc[0]) / (N - 1)
        run = Run(N=N, eps_target=eps_t, amplitude=float(row["amplitude"]),
                  csv=csv, meta=meta, df=df, eps_actual=eps_actual,
                  eps_rel_disc=(eps_actual - eps_t) / eps_t,
                  min_bond_strain=_read_min_bond_from_log(csv))
        runs.append(run)

    # deterministic order
    runs.sort(key=lambda r: (EPS_ORDER.index(_closest_eps(r.eps_target)), N_ORDER.index(r.N)))
    return runs


def _closest_eps(eps: float) -> float:
    return min(EPS_ORDER, key=lambda e: abs(e - eps))


# --------------------------------------------------------------------------- #
# Derived quantities
# --------------------------------------------------------------------------- #
def phi_j(df: pd.DataFrame, eps_actual: float) -> tuple[np.ndarray, float]:
    """Normalized Toda progress Phi_J(t) = (J(t)-J0)/(2 eps - J0).

    For the mode-1 IC J0 > 2 eps, so denominator < 0 and Phi_J rises 0 -> 1 as J
    decreases toward 2 eps. Returns (phi, denominator).
    """
    J = df["TodaJ"].to_numpy()
    denom = 2.0 * eps_actual - J[0]
    return (J - J[0]) / denom, denom


def j_label(j0_2e: float, jtail_2e: float) -> str:
    """Finite-time category at nominal 1e7 (same criteria as the pilot report)."""
    if abs(jtail_2e - 1.0) <= 0.05:
        return "J-near-equilibrium"
    if abs(jtail_2e - j0_2e) <= 0.02 * abs(j0_2e):
        return "J-flat"
    return "J-evolving"


def compute_tail_stats(run: Run) -> dict:
    df = run.df
    k = int((1.0 - TAIL_FRACTION) * len(df))  # 400 -> last 100 rows
    twoeps = 2.0 * run.eps_actual
    J = df["TodaJ"].to_numpy()
    ftle = df["LyapunovFTLE"].to_numpy()
    loc = df["LyapunovLocal"].to_numpy()
    eta = df["Eta"].to_numpy()
    E = df["TotalEnergy"].to_numpy()
    phi, denom = phi_j(df, run.eps_actual)
    j_ratio = J / twoeps

    def ms(a):
        return float(np.mean(a)), float(np.std(a, ddof=1))

    jt_m, jt_s = ms(j_ratio[k:])
    st = {
        "tail_rows": len(df) - k,
        "J0_over_2eps": float(j_ratio[0]),
        "J_tail_mean_over_2eps": jt_m, "J_tail_std_over_2eps": jt_s,
        "PhiJ_tail_mean": ms(phi[k:])[0], "PhiJ_tail_std": ms(phi[k:])[1],
        "PhiJ_denominator": float(denom),
        "FTLE_final": float(ftle[-1]),
        "FTLE_tail_mean": ms(ftle[k:])[0], "FTLE_tail_std": ms(ftle[k:])[1],
        "LyapLocal_tail_mean": ms(loc[k:])[0], "LyapLocal_tail_std": ms(loc[k:])[1],
        "Eta_tail_mean": ms(eta[k:])[0], "Eta_tail_std": ms(eta[k:])[1],
        "max_abs_rel_energy_drift": float(np.max(np.abs(E - E[0]) / abs(E[0]))),
        "min_bond_strain": run.min_bond_strain,
        "J_label": j_label(float(j_ratio[0]), jt_m),
    }
    return st


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def _save(fig, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def _runs_by_eps(runs, eps):
    return sorted([r for r in runs if abs(_closest_eps(r.eps_target) - eps) < 1e-12],
                  key=lambda r: N_ORDER.index(r.N))


def _panel_title(eps, runs):
    labels = {r.tail_stats["J_label"] for r in _runs_by_eps(runs, eps)}
    lbl = labels.pop() if len(labels) == 1 else "/".join(sorted(labels))
    return f"$\\epsilon_{{\\rm target}}={eps:.0e}$   [{lbl}]"


def _logtime_curve(ax, t, y, N):
    """Plot y vs t on a log-time axis, omitting only the t==0 sample."""
    color, ls, mk = N_STYLE[N]
    m = t > 0
    ax.plot(t[m], y[m], color=color, linestyle=ls, marker=mk, markersize=3,
            markevery=8, linewidth=1.2, label=f"N = {N}")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_toda_ratio(runs, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, eps in zip(axes.ravel(), EPS_ORDER):
        for r in _runs_by_eps(runs, eps):
            t = r.df["Time"].to_numpy()
            y = r.df["TodaJ"].to_numpy() / (2.0 * r.eps_actual)
            _logtime_curve(ax, t, y, r.N)
        ax.axhline(1.0, color="0.35", ls=":", lw=1.0,
                   label=r"$J/2\epsilon=1$ (α equilib. estimate)")
        ax.axhline(1.5, color="0.6", ls="--", lw=0.9,
                   label=r"$J/2\epsilon=1.5$ (mode-1 $J(0)$ ref)")
        ax.set_xscale("log"); ax.grid(True, which="both", alpha=0.3)
        ax.set_title(_panel_title(eps, runs))
        ax.set_ylabel(r"$J(t)\,/\,(2\,\epsilon_{\rm act})$")
    for ax in axes[-1]:
        ax.set_xlabel(r"time $t$ (log scale; $t=0$ omitted)")
    axes[0, 0].legend(fontsize=8, frameon=True, loc="best")
    fig.suptitle(SUPTITLE + "\nexact Toda observable $J(t)/(2\\epsilon_{\\rm act})$", y=1.0)
    fig.tight_layout()
    return _save(fig, out_dir, "fig3_toda_ratio_vs_logtime")


def fig_phi_j(runs, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, eps in zip(axes.ravel(), EPS_ORDER):
        for r in _runs_by_eps(runs, eps):
            t = r.df["Time"].to_numpy()
            phi, _ = phi_j(r.df, r.eps_actual)
            _logtime_curve(ax, t, phi, r.N)
        ax.axhline(0.0, color="0.6", ls="--", lw=0.9, label=r"$\Phi_J=0$ (initial)")
        ax.axhline(1.0, color="0.35", ls=":", lw=1.0, label=r"$\Phi_J=1$ ($J=2\epsilon$)")
        ax.set_xscale("log"); ax.grid(True, which="both", alpha=0.3)
        ax.set_title(_panel_title(eps, runs))
        ax.set_ylabel(r"$\Phi_J(t)=\dfrac{J(t)-J(0)}{2\epsilon_{\rm act}-J(0)}$")
    for ax in axes[-1]:
        ax.set_xlabel(r"time $t$ (log scale; $t=0$ omitted)")
    axes[0, 0].legend(fontsize=8, frameon=True, loc="best")
    fig.suptitle(SUPTITLE + "\nnormalized Toda progress  (sign: $J(0)>2\\epsilon$, so "
                 "num. & denom. both $<0$; $\\Phi_J$ rises $0\\to1$)", y=1.0)
    fig.tight_layout()
    return _save(fig, out_dir, "fig4_phiJ_vs_logtime")


def fig_lyapunov(runs, out_dir):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    nonpos_note = []
    for col, eps in enumerate(EPS_ORDER):
        ax_top, ax_bot = axes[0, col], axes[1, col]
        for r in _runs_by_eps(runs, eps):
            t = r.df["Time"].to_numpy()
            ftle = r.df["LyapunovFTLE"].to_numpy()
            loc = r.df["LyapunovLocal"].to_numpy()
            color, ls, mk = N_STYLE[r.N]
            m = t > 0
            # top: cumulative FTLE, log-y (omit nonpositive from log render)
            pos = m & (ftle > 0)
            if np.any(m & (ftle <= 0)):
                nonpos_note.append(f"eps={eps:.0e} N={r.N}: {int(np.sum(m & (ftle <= 0)))} nonpos FTLE omitted from log")
            ax_top.plot(t[pos], ftle[pos], color=color, linestyle=ls, marker=mk,
                        markersize=3, markevery=8, linewidth=1.2, label=f"N = {r.N}")
            # bottom: LyapunovLocal, linear-y (keep negatives, no log)
            ax_bot.plot(t[m], loc[m], color=color, linestyle=ls, marker=mk,
                        markersize=2.5, markevery=8, linewidth=1.0, label=f"N = {r.N}")
        # 1/t visual guide on the top panel (labeled, not a fit)
        tt = np.logspace(np.log10(2e4), np.log10(9.98e6), 50)
        g = _runs_by_eps(runs, eps)[0]
        anchor = g.df["LyapunovFTLE"].to_numpy()[1] * g.df["Time"].to_numpy()[1]
        ax_top.plot(tt, anchor / tt, color="0.6", ls=":", lw=0.9, label=r"$\propto 1/t$ guide")
        ax_top.set_xscale("log"); ax_top.set_yscale("log")
        ax_top.grid(True, which="both", alpha=0.3)
        ax_top.set_title(_panel_title(eps, runs), fontsize=9)
        ax_bot.set_xscale("log"); ax_bot.grid(True, which="both", alpha=0.3)
        ax_bot.axhline(0.0, color="0.4", lw=1.0)
        ax_bot.set_xlabel(r"time $t$ (log; $t=0$ omitted)")
    axes[0, 0].set_ylabel(r"cumulative FTLE  $\lambda_{\max}(t)$  (log)")
    axes[1, 0].set_ylabel(r"local exponent  $\lambda_{\rm loc}(t)$  (linear)")
    axes[0, 0].legend(fontsize=7, frameon=True)
    axes[1, 0].legend(fontsize=7, frameon=True)
    fig.suptitle(SUPTITLE + "\nLyapunov: cumulative FTLE (top, log-y) and local exponent "
                 "(bottom, linear-y with zero line)", y=1.0)
    fig.tight_layout()
    paths = _save(fig, out_dir, "fig5_lyapunov_vs_logtime")
    return paths, nonpos_note


def fig_eta(runs, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, eps in zip(axes.ravel(), EPS_ORDER):
        for r in _runs_by_eps(runs, eps):
            t = r.df["Time"].to_numpy()
            _logtime_curve(ax, t, r.df["Eta"].to_numpy(), r.N)
        ax.set_xscale("log"); ax.grid(True, which="both", alpha=0.3)
        ax.set_title(_panel_title(eps, runs))
        ax.set_ylabel(r"spectral entropy  $\eta(t)$")
        ax.set_ylim(-0.02, 1.02)
    for ax in axes[-1]:
        ax.set_xlabel(r"time $t$ (log scale; $t=0$ omitted)")
    axes[0, 0].legend(fontsize=8, frameon=True, loc="best")
    fig.suptitle(SUPTITLE + "\nnormalized spectral entropy $\\eta(t)$ (as stored; not renormalized)", y=1.0)
    fig.tight_layout()
    return _save(fig, out_dir, "fig6_eta_vs_logtime")


def fig_summary(runs, out_dir):
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(17, 5.2))
    for N in N_ORDER:
        color, ls, mk = N_STYLE[N]
        sub = sorted([r for r in runs if r.N == N],
                     key=lambda r: _closest_eps(r.eps_target))
        eps = [r.eps_actual for r in sub]
        # Panel A: J tail mean / 2eps with temporal-std error bars
        jm = [r.tail_stats["J_tail_mean_over_2eps"] for r in sub]
        js = [r.tail_stats["J_tail_std_over_2eps"] for r in sub]
        axA.errorbar(eps, jm, yerr=js, color=color, ls=ls, marker=mk, capsize=3,
                     lw=1.3, label=f"N = {N}")
        # Panel B: FTLE tail mean (log-y; all positive)
        fm = [r.tail_stats["FTLE_tail_mean"] for r in sub]
        fs = [r.tail_stats["FTLE_tail_std"] for r in sub]
        axB.errorbar(eps, fm, yerr=fs, color=color, ls=ls, marker=mk, capsize=3,
                     lw=1.3, label=f"N = {N}")
        # Panel C: Eta tail mean
        em = [r.tail_stats["Eta_tail_mean"] for r in sub]
        es = [r.tail_stats["Eta_tail_std"] for r in sub]
        axC.errorbar(eps, em, yerr=es, color=color, ls=ls, marker=mk, capsize=3,
                     lw=1.3, label=f"N = {N}")
    for ax in (axA, axB, axC):
        ax.set_xscale("log"); ax.grid(True, which="both", alpha=0.3)
        ax.set_xlabel(r"energy density  $\epsilon_{\rm act}$")
    axA.axhline(1.0, color="0.35", ls=":", lw=1.0)
    axA.axhline(1.5, color="0.6", ls="--", lw=0.9)
    axA.set_ylabel(r"$\langle J/2\epsilon_{\rm act}\rangle_{\rm tail}$")
    axA.set_title("A. Toda action (tail)")
    axB.set_yscale("log")
    axB.set_ylabel(r"$\langle\lambda_{\max}\rangle_{\rm tail}$  (finite-time est.)")
    axB.set_title("B. finite-time max Lyapunov (tail)")
    axC.set_ylabel(r"$\langle\eta\rangle_{\rm tail}$")
    axC.set_title("C. spectral entropy (tail)")
    axA.legend(fontsize=8, frameon=True)
    # annotate the finite-time J-category at each epsilon (descriptive, sparse)
    for eps in EPS_ORDER:
        g = _runs_by_eps(runs, eps)
        lbl = g[0].tail_stats["J_label"]
        y = max(r.tail_stats["J_tail_mean_over_2eps"] for r in g)
        axA.annotate(lbl.replace("J-", ""), (g[0].eps_actual, y),
                     textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7, color="0.3")
    fig.suptitle(SUPTITLE + "\nfinite-time tail diagnostics vs energy density "
                 "(error bars = temporal std over tail, NOT realization uncertainty)", y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir, "fig8_summary_vs_epsilon")


def fig_finite_size(runs, out_dir):
    fig, (axJ, axF, axE) = plt.subplots(1, 3, figsize=(17, 5.0))
    eps_styles = {3e-5: ("C0", "o"), 1e-4: ("C1", "s"), 8e-4: ("C2", "^"), 3e-3: ("C3", "D")}
    for eps in EPS_ORDER:
        g = _runs_by_eps(runs, eps)
        Ns = [r.N for r in g]
        color, mk = eps_styles[eps]
        axJ.plot(Ns, [r.tail_stats["J_tail_mean_over_2eps"] for r in g], color=color,
                 marker=mk, lw=1.2, label=f"$\\epsilon={eps:.0e}$")
        axF.plot(Ns, [r.tail_stats["FTLE_tail_mean"] for r in g], color=color,
                 marker=mk, lw=1.2, label=f"$\\epsilon={eps:.0e}$")
        axE.plot(Ns, [r.tail_stats["Eta_tail_mean"] for r in g], color=color,
                 marker=mk, lw=1.2, label=f"$\\epsilon={eps:.0e}$")
    for ax in (axJ, axF, axE):
        ax.set_xscale("log", base=2); ax.set_xticks(N_ORDER); ax.set_xticklabels(N_ORDER)
        ax.grid(True, which="both", alpha=0.3); ax.set_xlabel("system size $N$")
    axJ.axhline(1.0, color="0.35", ls=":", lw=1.0)
    axJ.set_ylabel(r"$\langle J/2\epsilon_{\rm act}\rangle_{\rm tail}$"); axJ.set_title("Toda action")
    axF.set_yscale("log"); axF.set_ylabel(r"$\langle\lambda_{\max}\rangle_{\rm tail}$"); axF.set_title("finite-time FTLE")
    axE.set_ylabel(r"$\langle\eta\rangle_{\rm tail}$"); axE.set_title("spectral entropy")
    axJ.legend(fontsize=8, frameon=True)
    fig.suptitle(SUPTITLE + "\nfinite-size trends at fixed $\\epsilon$ (per-observable; "
                 "do not read as a single 'more thermalized' axis)", y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir, "fig9_finite_size_vs_N")


# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #
def write_summary(runs, out_dir) -> Path:
    recs = []
    for r in runs:
        recs.append({
            "N": r.N, "eps_target": r.eps_target, "eps_actual": r.eps_actual,
            "eps_rel_discrepancy": r.eps_rel_disc, "amplitude": r.amplitude,
            **r.tail_stats,
            "source_csv": str(r.csv.name),
        })
    df = pd.DataFrame(recs)
    path = out_dir / "alpha_pilot_summary.csv"
    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m analysis.pilot_diagnostics")
    ap.add_argument("--data-dir", default="data/alpha_pilot_v1")
    ap.add_argument("--manifest", default="analysis/manifests/run_manifest.json")
    ap.add_argument("--output-dir", default="figures/alpha_pilot_v1")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)

    print(f"Discovering + validating pilot in {data_dir} against {args.manifest} ...")
    runs = discover_and_validate(data_dir, Path(args.manifest))
    print(f"  validated {len(runs)} trajectories (all checks pass)")

    # epsilon discrepancy report + Phi_J denominator safety
    print("  epsilon_actual vs target, and Phi_J denominator safety:")
    for r in runs:
        r.tail_stats = compute_tail_stats(r)
        denom = r.tail_stats["PhiJ_denominator"]
        rel = abs(denom) / (2.0 * r.eps_actual)
        if rel < 1e-6:
            raise ValidationError(
                f"Phi_J denominator too small for N={r.N} eps={r.eps_target:.0e}: |denom|/2eps={rel:.2e}")
    max_disc = max(abs(r.eps_rel_disc) for r in runs)
    print(f"    max |eps_actual-eps_target|/eps_target = {max_disc:.2e}")
    print(f"    min |Phi_J denominator|/(2 eps) = {min(abs(r.tail_stats['PhiJ_denominator'])/(2*r.eps_actual) for r in runs):.3f} (safe)")

    written = []
    written += fig_toda_ratio(runs, out_dir)
    written += fig_phi_j(runs, out_dir)
    lyap_paths, nonpos = fig_lyapunov(runs, out_dir)
    written += lyap_paths
    written += fig_eta(runs, out_dir)
    written += fig_summary(runs, out_dir)
    written += fig_finite_size(runs, out_dir)
    summary_path = write_summary(runs, out_dir)

    print(f"\n  wrote {len(written)} figure files + summary table:")
    for p in written:
        print(f"    {p}")
    print(f"    {summary_path}")
    if nonpos:
        print("  NOTE nonpositive cumulative FTLE omitted from log render:")
        for n in nonpos:
            print(f"    {n}")
    else:
        print("  (no nonpositive cumulative FTLE encountered)")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
