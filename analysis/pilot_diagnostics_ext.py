"""Analysis + plotting for the FPUT-alpha=1 pilot EXTENDED to nominal 1e8.

Read-only companion to ``analysis/pilot_diagnostics.py``. It rebuilds the full
0 -> 1e8 trajectories for the five key alpha-pilot points by concatenating the
frozen pilot rows in ``data/alpha_pilot_v1/`` (segments 0..499, times 0 ->
9.98e6) with the continuation rows in ``data/alpha_pilot_v1_ext/`` (segments
500..4999, times 1e7 -> 9.998e7). It NEVER runs a simulation and NEVER edits any
raw CSV, checkpoint, or manifest.

Extended points (continuation present on this machine):
    eps=1e-4  N=512, 1024, 2048        (all three N extended)
    eps=8e-4  N=2048                    (only this N extended)
The eps=8e-4 N=512 continuation named in the task is ABSENT here, so that point
cannot be built to 1e8; it is reported as a seam-validation failure and left at
its 1e7 pilot length. The other pilot eps values (3e-5, 3e-3) were not extended
and are shown at 1e7 only, clearly marked as shorter-duration.

Conventions (inherited from pilot_diagnostics.py, restated where they change):
  * epsilon_actual = TotalEnergy[0]/(N-1), read from the PILOT half, used for
    every normalization/ratio.
  * Seam continuity is asserted: cont first Time == pilot last Time + one stride
    (9.98e6 + 2e4 = 1e7), no duplicated/skipped snapshot; 5000 total rows;
    provenance SolverGitCommit=4a66fec / SolverGitDirty=0 in BOTH halves;
    epsilon_actual consistent between halves.
  * Log-time plots omit only the Time==0 sample from the drawn curve.
  * A light vertical line marks t=1e7 (pilot/continuation boundary) in each panel.
  * Tail statistics use the final 20% of the available rows. For the extended
    points that is the last 1000 of 5000 rows (t in [8e7, 9.998e7]); for the
    shorter-duration (1e7) points it is the last 100 of 500 rows. Reported std
    is TEMPORAL variation over the tail window, not realization uncertainty.
  * J = 2*epsilon is the alpha *equilibrium estimate* (not a theorem). The mode-1
    IC has J(0) ~ 3*epsilon, so J approaches 2*eps from above.
  * FTLE reference/guide lines are visual guides, not fits. Low-eps FTLE
    behaviour is described as "consistent with decay toward zero over the
    observed interval" where applicable.
  * matplotlib only; global styles not modified. N encoded by color AND
    line/marker style (never color alone).

Usage:
    python -m analysis.pilot_diagnostics_ext \
        --pilot-dir data/alpha_pilot_v1 \
        --ext-dir   data/alpha_pilot_v1_ext \
        --output-dir figures/alpha_pilot_v1_ext
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend (not a style change)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

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

STRIDE = 2.0e4              # dt(0.1) * Stride(200000)
PILOT_ROWS = 500           # segments 0..499
EXT_ROWS = 4500            # segments 500..4999
FULL_ROWS = 5000
PILOT_LAST_T = 9.98e6
SEAM_T = 1.0e7             # first continuation time (= pilot last + stride)
FULL_LAST_T = 9.998e7
NOM_1E7 = 1.0e7
NOM_1E8 = 1.0e8

# Per-N visual style: (color, linestyle, marker) so N is never color-only.
N_STYLE = {
    512:  ("C0", "-", "o"),
    1024: ("C1", "--", "s"),
    2048: ("C3", "-.", "^"),
}
SUPTITLE = ("FPUT-$\\alpha$  ($\\alpha=1$, mode-1 initial condition)  —  "
            "pilot extended to nominal $10^{8}$ (last saved $9.998\\times10^{7}$)")


# --------------------------------------------------------------------------- #
# Discovery, concatenation, seam validation
# --------------------------------------------------------------------------- #
@dataclass
class Run:
    N: int
    eps_target: float
    amplitude: float
    pilot_csv: Path
    ext_csv: Path | None
    df: pd.DataFrame                       # concatenated (or pilot-only)
    extended: bool
    nominal_duration: float
    last_time: float
    eps_actual: float
    eps_actual_ext: float = float("nan")   # from continuation half (NaN if none)
    seam_report: str = ""
    tail_stats: dict = field(default_factory=dict)


class ValidationError(RuntimeError):
    pass


_FN = re.compile(r"alpha_N(\d+)_eps([0-9.]+e[+-]\d+)_A([0-9]+\.[0-9]+)")


def _parse_pilot_name(p: Path):
    m = _FN.search(p.name)
    if not m:
        return None
    return int(m.group(1)), float(m.group(2)), float(m.group(3))


def _read_clean(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv, comment="#")
    df.columns = df.columns.str.strip()
    return df


def _basic_checks(df: pd.DataFrame, name: str, nrows: int):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValidationError(f"{name}: missing columns {missing}")
    if len(df) != nrows:
        raise ValidationError(f"{name}: {len(df)} rows (expected {nrows})")
    for c in REQUIRED_COLUMNS:
        if not np.all(np.isfinite(df[c].to_numpy())):
            raise ValidationError(f"{name}: non-finite in column {c}")


def _closest_eps(eps: float) -> float:
    return min(EPS_ORDER, key=lambda e: abs(e - eps))


def _match_ext(N: int, amp: float, ext_metas: dict) -> Path | None:
    """Find the continuation CSV matching a pilot (N, amplitude) by metadata."""
    for path, meta in ext_metas.items():
        if meta.get("Model", "").lower() != "alpha":
            continue
        if int(float(meta.get("N", -1))) != N:
            continue
        if "Amplitude" not in meta:
            continue
        if abs(float(meta["Amplitude"]) - amp) <= 1e-2:
            return path
    return None


def discover(pilot_dir: Path, ext_dir: Path) -> tuple[list[Run], list[str]]:
    pilots = sorted(pilot_dir.glob("alpha_N*.csv"))
    ext_csvs = sorted(ext_dir.glob("alpha_N*_cont_*.csv"))
    ext_metas = {p: get_metadata(str(p)) for p in ext_csvs}
    matched_ext: set[Path] = set()

    runs: list[Run] = []
    failures: list[str] = []
    for pc in pilots:
        parsed = _parse_pilot_name(pc)
        if parsed is None:
            continue
        N, eps_t, amp = parsed
        pmeta = get_metadata(str(pc))
        pilot_df = _read_clean(pc)

        try:
            _basic_checks(pilot_df, pc.name, PILOT_ROWS)
            tp = pilot_df["Time"].to_numpy()
            if abs(tp[0]) > 1e-9:
                raise ValidationError(f"{pc.name}: first saved time != 0 ({tp[0]})")
            if abs(tp[-1] - PILOT_LAST_T) > 1.0:
                raise ValidationError(f"{pc.name}: last saved time != 9.98e6 ({tp[-1]})")
            if pmeta.get("SolverGitCommit") != EXPECT_COMMIT:
                raise ValidationError(f"{pc.name}: pilot commit {pmeta.get('SolverGitCommit')} != {EXPECT_COMMIT}")
            if pmeta.get("SolverGitDirty") != "0":
                raise ValidationError(f"{pc.name}: pilot dirty {pmeta.get('SolverGitDirty')} != 0")
        except ValidationError as e:
            failures.append(str(e))
            continue

        eps_actual = float(pilot_df["TotalEnergy"].iloc[0]) / (N - 1)
        ec = _match_ext(N, amp, ext_metas)

        if ec is None:
            # No continuation: keep pilot-only, 1e7 duration.
            run = Run(N=N, eps_target=eps_t, amplitude=amp, pilot_csv=pc, ext_csv=None,
                      df=pilot_df.reset_index(drop=True), extended=False,
                      nominal_duration=NOM_1E7, last_time=float(tp[-1]),
                      eps_actual=eps_actual,
                      seam_report="no continuation CSV found -> pilot-only (1e7)")
            runs.append(run)
            continue

        # Continuation present: validate seam and concatenate.
        matched_ext.add(ec)
        cmeta = ext_metas[ec]
        cont_df = _read_clean(ec)
        seam_ok = True
        try:
            _basic_checks(cont_df, ec.name, EXT_ROWS)
            tc = cont_df["Time"].to_numpy()
            if abs(tc[0] - (tp[-1] + STRIDE)) > 1.0:
                raise ValidationError(
                    f"{ec.name}: seam gap cont_first({tc[0]:.6e}) != pilot_last+stride({tp[-1] + STRIDE:.6e})")
            if abs(tc[0] - SEAM_T) > 1.0:
                raise ValidationError(f"{ec.name}: cont first time {tc[0]:.6e} != 1e7")
            if abs(tc[-1] - FULL_LAST_T) > 1.0:
                raise ValidationError(f"{ec.name}: cont last time {tc[-1]:.6e} != 9.998e7")
            dt = np.diff(tc)
            if np.max(np.abs(dt - STRIDE)) > 1.0:
                raise ValidationError(f"{ec.name}: non-uniform continuation stride")
            if cmeta.get("SolverGitCommit") != EXPECT_COMMIT:
                raise ValidationError(f"{ec.name}: cont commit {cmeta.get('SolverGitCommit')} != {EXPECT_COMMIT}")
            if cmeta.get("SolverGitDirty") != "0":
                raise ValidationError(f"{ec.name}: cont dirty {cmeta.get('SolverGitDirty')} != 0")
            eps_actual_ext = float(cont_df["TotalEnergy"].iloc[0]) / (N - 1)
            rel = abs(eps_actual_ext - eps_actual) / eps_actual
            if rel > 1e-3:
                raise ValidationError(
                    f"{ec.name}: eps_actual mismatch pilot({eps_actual:.6e}) vs cont({eps_actual_ext:.6e}) rel={rel:.2e}")
        except ValidationError as e:
            failures.append(str(e))
            seam_ok = False

        if not seam_ok:
            continue

        full = pd.concat([pilot_df, cont_df], ignore_index=True)
        _basic_checks(full, f"{pc.name}+cont", FULL_ROWS)
        run = Run(N=N, eps_target=eps_t, amplitude=amp, pilot_csv=pc, ext_csv=ec,
                  df=full, extended=True, nominal_duration=NOM_1E8,
                  last_time=float(full["Time"].to_numpy()[-1]),
                  eps_actual=eps_actual, eps_actual_ext=eps_actual_ext,
                  seam_report=(f"OK: 500+4500=5000 rows; seam {SEAM_T:.3e}=pilot_last+stride; "
                               f"commit={EXPECT_COMMIT}/dirty=0 both halves; "
                               f"eps rel-diff={rel:.2e}"))
        runs.append(run)

    # eps=8e-4 N=512 is expected by the task but has no continuation -> explicit note
    have_ext = {(r.N, _closest_eps(r.eps_target)) for r in runs if r.extended}
    for want in [(512, 8e-4)]:
        if want not in have_ext:
            failures.append(
                f"expected continuation for N={want[0]} eps={want[1]:.0e} NOT FOUND in ext dir "
                f"-> point cannot be extended to 1e8 (left at 1e7 pilot length)")

    runs.sort(key=lambda r: (EPS_ORDER.index(_closest_eps(r.eps_target)), N_ORDER.index(r.N)))
    return runs, failures


# --------------------------------------------------------------------------- #
# Derived quantities & tail stats
# --------------------------------------------------------------------------- #
def phi_j(df: pd.DataFrame, eps_actual: float) -> tuple[np.ndarray, float]:
    J = df["TodaJ"].to_numpy()
    denom = 2.0 * eps_actual - J[0]
    return (J - J[0]) / denom, denom


def compute_tail_stats(run: Run) -> dict:
    df = run.df
    n = len(df)
    k = int((1.0 - TAIL_FRACTION) * n)  # extended: 4000 -> last 1000; pilot-only: 400 -> last 100
    twoeps = 2.0 * run.eps_actual
    J = df["TodaJ"].to_numpy()
    ftle = df["LyapunovFTLE"].to_numpy()
    loc = df["LyapunovLocal"].to_numpy()
    eta = df["Eta"].to_numpy()
    E = df["TotalEnergy"].to_numpy()
    t = df["Time"].to_numpy()
    phi, denom = phi_j(df, run.eps_actual)
    j_ratio = J / twoeps

    def ms(a):
        return float(np.mean(a)), float(np.std(a, ddof=1))

    jt_m, jt_s = ms(j_ratio[k:])
    ll_m, ll_s = ms(loc[k:])
    ft_m, ft_s = ms(ftle[k:])
    # finite-time positive-plateau test (all three, per wording checklist B):
    ftle_flat = abs(ftle[-1] - ft_m) <= 0.25 * abs(ft_m) if ft_m != 0 else False
    loc_pos = ll_m > 0
    loc_above_noise = ll_m > ll_s          # fluctuations sit around a positive value
    return {
        "n_rows": n, "tail_rows": n - k,
        "tail_t_start": float(t[k]), "tail_t_end": float(t[-1]),
        "J0_over_2eps": float(j_ratio[0]),
        "J_at_1e7_over_2eps": float(j_ratio[min(PILOT_ROWS, n - 1)]),
        "J_tail_mean_over_2eps": jt_m, "J_tail_std_over_2eps": jt_s,
        "J_final_over_2eps": float(j_ratio[-1]),
        "PhiJ_tail_mean": ms(phi[k:])[0], "PhiJ_tail_std": ms(phi[k:])[1],
        "PhiJ_denominator": float(denom),
        "FTLE_final": float(ftle[-1]),
        "FTLE_tail_mean": ft_m, "FTLE_tail_std": ft_s,
        "LyapLocal_tail_mean": ll_m, "LyapLocal_tail_std": ll_s,
        "Eta_tail_mean": ms(eta[k:])[0], "Eta_tail_std": ms(eta[k:])[1],
        "max_abs_rel_energy_drift": float(np.max(np.abs(E - E[0]) / abs(E[0]))),
        "pos_plateau_ftle_flat": bool(ftle_flat),
        "pos_plateau_loc_positive": bool(loc_pos),
        "pos_plateau_loc_above_noise": bool(loc_above_noise),
        "positive_plateau_all3": bool(ftle_flat and loc_pos and loc_above_noise),
    }


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
    g = _runs_by_eps(runs, eps)
    ext_ns = [r.N for r in g if r.extended]
    if ext_ns:
        tag = f"to $10^8$: N={','.join(str(n) for n in ext_ns)}"
        if len(ext_ns) < len(g):
            tag += "; others $10^7$ only"
    else:
        tag = "$10^7$ only (not extended)"
    return f"$\\epsilon_{{\\rm target}}={eps:.0e}$   [{tag}]"


def _seam_vline(ax):
    ax.axvline(SEAM_T, color="0.7", ls="-", lw=0.8, zorder=0)


def _curve(ax, t, y, N, extended, **kw):
    color, ls, mk = N_STYLE[N]
    m = t > 0
    lbl = f"N = {N}" + ("" if extended else "  ($10^7$)")
    # non-extended (1e7-only) curves drawn thinner + lower alpha to read as shorter-duration
    lw = kw.pop("lw", 1.2 if extended else 0.9)
    alpha = kw.pop("alpha", 1.0 if extended else 0.6)
    ax.plot(t[m], y[m], color=color, linestyle=ls, marker=mk, markersize=3,
            markevery=max(1, int(np.sum(m) / 60)), linewidth=lw, alpha=alpha, label=lbl, **kw)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_toda_ratio(runs, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, eps in zip(axes.ravel(), EPS_ORDER):
        for r in _runs_by_eps(runs, eps):
            t = r.df["Time"].to_numpy()
            y = r.df["TodaJ"].to_numpy() / (2.0 * r.eps_actual)
            _curve(ax, t, y, r.N, r.extended)
        _seam_vline(ax)
        ax.axhline(1.0, color="0.35", ls=":", lw=1.0,
                   label=r"$J/2\epsilon=1$ ($\alpha$ equilib. estimate)")
        ax.axhline(1.5, color="0.6", ls="--", lw=0.9,
                   label=r"$J/2\epsilon=1.5$ (mode-1 $J(0)$ ref)")
        ax.set_xscale("log"); ax.grid(True, which="both", alpha=0.3)
        ax.set_title(_panel_title(eps, runs), fontsize=9)
        ax.set_ylabel(r"$J(t)\,/\,(2\,\epsilon_{\rm act})$")
    for ax in axes[-1]:
        ax.set_xlabel(r"time $t$ (log scale; $t=0$ omitted; grey line $=10^7$ seam)")
    axes[0, 0].legend(fontsize=7, frameon=True, loc="best")
    fig.suptitle(SUPTITLE + "\nexact Toda observable $J(t)/(2\\epsilon_{\\rm act})$; "
                 "lines at 1.0/1.5 are guides, not fits", y=1.0)
    fig.tight_layout()
    return _save(fig, out_dir, "fig3_toda_ratio_vs_logtime_1e8")


def fig_phi_j(runs, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, eps in zip(axes.ravel(), EPS_ORDER):
        for r in _runs_by_eps(runs, eps):
            t = r.df["Time"].to_numpy()
            phi, _ = phi_j(r.df, r.eps_actual)
            _curve(ax, t, phi, r.N, r.extended)
        _seam_vline(ax)
        ax.axhline(0.0, color="0.6", ls="--", lw=0.9, label=r"$\Phi_J=0$ (initial)")
        ax.axhline(1.0, color="0.35", ls=":", lw=1.0, label=r"$\Phi_J=1$ ($J=2\epsilon$ est.)")
        ax.set_xscale("log"); ax.grid(True, which="both", alpha=0.3)
        ax.set_title(_panel_title(eps, runs), fontsize=9)
        ax.set_ylabel(r"$\Phi_J(t)=\dfrac{J(t)-J(0)}{2\epsilon_{\rm act}-J(0)}$")
    for ax in axes[-1]:
        ax.set_xlabel(r"time $t$ (log scale; $t=0$ omitted; grey line $=10^7$ seam)")
    axes[0, 0].legend(fontsize=7, frameon=True, loc="best")
    fig.suptitle(SUPTITLE + "\nnormalized Toda progress ($J(0)>2\\epsilon$, so $\\Phi_J$ rises "
                 "$0\\to1$ as $J\\to2\\epsilon$)", y=1.0)
    fig.tight_layout()
    return _save(fig, out_dir, "fig4_phiJ_vs_logtime_1e8")


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
            lw = 1.2 if r.extended else 0.9
            alpha = 1.0 if r.extended else 0.6
            me = max(1, int(np.sum(m) / 60))
            lbl = f"N = {r.N}" + ("" if r.extended else "  ($10^7$)")
            pos = m & (ftle > 0)
            if np.any(m & (ftle <= 0)):
                nonpos_note.append(
                    f"eps={eps:.0e} N={r.N}: {int(np.sum(m & (ftle <= 0)))} nonpos FTLE omitted from log")
            ax_top.plot(t[pos], ftle[pos], color=color, linestyle=ls, marker=mk,
                        markersize=3, markevery=me, linewidth=lw, alpha=alpha, label=lbl)
            ax_bot.plot(t[m], loc[m], color=color, linestyle=ls, marker=mk,
                        markersize=2.5, markevery=me, linewidth=lw * 0.85, alpha=alpha, label=lbl)
        # 1/t visual guide (labeled, not a fit), anchored on the earliest positive sample
        g = _runs_by_eps(runs, eps)[0]
        gt = g.df["Time"].to_numpy(); gf = g.df["LyapunovFTLE"].to_numpy()
        anchor = gf[1] * gt[1]
        tt = np.logspace(np.log10(2e4), np.log10(g.last_time), 60)
        ax_top.plot(tt, anchor / tt, color="0.6", ls=":", lw=0.9, label=r"$\propto 1/t$ guide (not a fit)")
        _seam_vline(ax_top); _seam_vline(ax_bot)
        ax_top.set_xscale("log"); ax_top.set_yscale("log")
        ax_top.grid(True, which="both", alpha=0.3)
        ax_top.set_title(_panel_title(eps, runs), fontsize=8)
        ax_bot.set_xscale("log"); ax_bot.grid(True, which="both", alpha=0.3)
        ax_bot.axhline(0.0, color="0.4", lw=1.0)
        ax_bot.set_xlabel(r"time $t$ (log; $t=0$ omitted; grey $=10^7$)")
    axes[0, 0].set_ylabel(r"cumulative FTLE  $\lambda_{\max}(t)$  (log)")
    axes[1, 0].set_ylabel(r"local exponent  $\lambda_{\rm loc}(t)$  (linear)")
    axes[0, 0].legend(fontsize=7, frameon=True)
    axes[1, 0].legend(fontsize=7, frameon=True)
    fig.suptitle(SUPTITLE + "\nLyapunov: cumulative FTLE (top, log-y; $1/t$ line is a guide) and "
                 "local exponent (bottom, linear-y, zero line)", y=1.0)
    fig.tight_layout()
    paths = _save(fig, out_dir, "fig5_lyapunov_vs_logtime_1e8")
    return paths, nonpos_note


def fig_eta(runs, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, eps in zip(axes.ravel(), EPS_ORDER):
        for r in _runs_by_eps(runs, eps):
            t = r.df["Time"].to_numpy()
            _curve(ax, t, r.df["Eta"].to_numpy(), r.N, r.extended)
        _seam_vline(ax)
        ax.set_xscale("log"); ax.grid(True, which="both", alpha=0.3)
        ax.set_title(_panel_title(eps, runs), fontsize=9)
        ax.set_ylabel(r"spectral entropy  $\eta(t)$")
        ax.set_ylim(-0.02, 1.02)
    for ax in axes[-1]:
        ax.set_xlabel(r"time $t$ (log scale; $t=0$ omitted; grey line $=10^7$ seam)")
    axes[0, 0].legend(fontsize=7, frameon=True, loc="best")
    fig.suptitle(SUPTITLE + "\nnormalized spectral entropy $\\eta(t)$ (harmonic-mode spreading; "
                 "not evidence of chaos or equilibration)", y=1.0)
    fig.tight_layout()
    return _save(fig, out_dir, "fig6_eta_vs_logtime_1e8")


def fig_summary(runs, out_dir):
    """Summary vs epsilon using 1e8 tail for extended points, 1e7 tail otherwise."""
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(17, 5.2))
    for N in N_ORDER:
        color, ls, mk = N_STYLE[N]
        sub = sorted([r for r in runs if r.N == N], key=lambda r: _closest_eps(r.eps_target))
        eps = [r.eps_actual for r in sub]
        # open markers for shorter-duration (1e7) points, filled for 1e8
        for ax, key, kerr in ((axA, "J_tail_mean_over_2eps", "J_tail_std_over_2eps"),
                              (axB, "FTLE_tail_mean", "FTLE_tail_std"),
                              (axC, "Eta_tail_mean", "Eta_tail_std")):
            ym = [r.tail_stats[key] for r in sub]
            ys = [r.tail_stats[kerr] for r in sub]
            ax.errorbar(eps, ym, yerr=ys, color=color, ls=ls, lw=1.3, capsize=3, label=f"N = {N}")
            for r, x, y in zip(sub, eps, ym):
                ax.plot([x], [y], color=color, marker=mk, markersize=7,
                        markerfacecolor=(color if r.extended else "white"),
                        markeredgecolor=color, zorder=5)
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
    axA.legend(fontsize=8, frameon=True, title="filled=$10^8$ tail, open=$10^7$ tail")
    fig.suptitle(SUPTITLE + "\ntail diagnostics vs energy density  (filled marker = $10^8$ tail; "
                 "open marker = $10^7$ tail, shorter duration; "
                 "error bars = temporal std, NOT realization uncertainty)", y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir, "fig8_summary_vs_epsilon_1e8")


# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #
def write_summary(runs, out_dir) -> Path:
    recs = []
    for r in runs:
        recs.append({
            "N": r.N, "eps_target": r.eps_target, "eps_actual": r.eps_actual,
            "eps_actual_ext": r.eps_actual_ext, "amplitude": r.amplitude,
            "extended_to_1e8": r.extended, "nominal_duration": r.nominal_duration,
            "last_time": r.last_time, **r.tail_stats,
            "pilot_csv": r.pilot_csv.name,
            "ext_csv": (r.ext_csv.name if r.ext_csv else ""),
        })
    df = pd.DataFrame(recs)
    path = out_dir / "alpha_pilot_ext_summary.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m analysis.pilot_diagnostics_ext")
    ap.add_argument("--pilot-dir", default="data/alpha_pilot_v1")
    ap.add_argument("--ext-dir", default="data/alpha_pilot_v1_ext")
    ap.add_argument("--output-dir", default="figures/alpha_pilot_v1_ext")
    args = ap.parse_args(argv)

    pilot_dir = Path(args.pilot_dir)
    ext_dir = Path(args.ext_dir)
    out_dir = Path(args.output_dir)

    print(f"Discovering pilot={pilot_dir}  ext={ext_dir}")
    runs, failures = discover(pilot_dir, ext_dir)

    print("\n=== seam / provenance validation ===")
    for r in runs:
        tag = f"eps={_closest_eps(r.eps_target):.0e} N={r.N}"
        print(f"  [{'EXT ' if r.extended else '1e7 '}] {tag}: {r.seam_report}")
    if failures:
        print("\n  FAILURES / MISSING (reported, not fatal to the rest):")
        for f in failures:
            print(f"    - {f}")

    for r in runs:
        r.tail_stats = compute_tail_stats(r)

    written = []
    written += fig_toda_ratio(runs, out_dir)
    written += fig_phi_j(runs, out_dir)
    lyap_paths, nonpos = fig_lyapunov(runs, out_dir)
    written += lyap_paths
    written += fig_eta(runs, out_dir)
    written += fig_summary(runs, out_dir)
    summary_path = write_summary(runs, out_dir)

    print(f"\n=== wrote {len(written)} figure files + summary table ===")
    for p in written:
        print(f"  {p}")
    print(f"  {summary_path}")
    if nonpos:
        print("  NOTE nonpositive cumulative FTLE omitted from log render:")
        for n in nonpos:
            print(f"    {n}")

    # Console tail table (extended points highlighted)
    print("\n=== TAIL TABLE (tail = final 20% of available rows) ===")
    hdr = ("eps_t    N     dur    J0/2e  J@1e7  Jtail/2e   Jfinal  FTLEtail   locTail       "
           "eta   posPlateau(all3)")
    print(hdr)
    for r in runs:
        s = r.tail_stats
        print(f"  {_closest_eps(r.eps_target):.0e} {r.N:5d} "
              f"{'1e8' if r.extended else '1e7':>5} "
              f"{s['J0_over_2eps']:6.3f} {s['J_at_1e7_over_2eps']:6.3f} "
              f"{s['J_tail_mean_over_2eps']:6.3f}±{s['J_tail_std_over_2eps']:.3f} "
              f"{s['J_final_over_2eps']:6.3f} {s['FTLE_tail_mean']:.2e} "
              f"{s['LyapLocal_tail_mean']:+.2e}±{s['LyapLocal_tail_std']:.1e} "
              f"{s['Eta_tail_mean']:5.3f}  {s['positive_plateau_all3']}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
