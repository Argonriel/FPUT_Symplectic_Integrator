"""Diagnostic figures for the FPUT-beta summary (matplotlib only, no seaborn).

Every figure is saved as both PNG and PDF via
``plt.savefig(path, dpi=300, bbox_inches="tight")`` (the repository convention).
The module never modifies the global matplotlib style. ``Eta`` is plotted as
stored in the CSV — it is already normalized spectral entropy and is never
re-normalized here.

Ordering is deterministic: ``N`` ascending, epsilon ascending within each curve.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; not a style change
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.collapse import CollapseResult, Curve  # noqa: E402

_TITLE_PREFIX = "FPUT-β (Yoshida4)"


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    """Save a figure as PNG and PDF; return the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def _by_n(df: pd.DataFrame):
    """Yield ``(N, sub_df)`` groups sorted by N, each sorted by epsilon ascending."""
    for n in sorted(df["N"].unique()):
        sub = df[df["N"] == n].sort_values("epsilon")
        yield int(n), sub


def _scatter_line_by_n(
    df: pd.DataFrame,
    ycol: str,
    *,
    ylabel: str,
    title: str,
    out_dir: Path,
    stem: str,
    logx: bool = True,
    logy: bool = False,
) -> list[Path]:
    """Shared helper: one line+marker series per N against epsilon."""
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for n, sub in _by_n(df):
        ax.plot(sub["epsilon"], sub[ycol], marker="o", markersize=4, linewidth=1.2, label=f"N = {n}")
    ax.set_xlabel(r"energy density  $\epsilon = H(0)/(N-1)$")
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="system size", frameon=True)
    # Scientific notation only applies to linear axes (log axes use LogFormatter).
    if not logy:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
    return _save(fig, out_dir, stem)


def plot_entropy_tail_mean(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Figure 1: tail-mean spectral entropy vs epsilon, one curve per N."""
    return _scatter_line_by_n(
        df,
        "eta_tail_mean",
        ylabel=r"tail-mean spectral entropy  $\langle\eta\rangle_{\rm tail}$",
        title=f"{_TITLE_PREFIX}: tail-mean spectral entropy vs energy density",
        out_dir=out_dir,
        stem="fig1_entropy_tailmean_vs_epsilon",
    )


def plot_entropy_tail_std(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Figure 2: tail std of spectral entropy vs epsilon."""
    return _scatter_line_by_n(
        df,
        "eta_tail_std",
        ylabel=r"tail std of spectral entropy  $\sigma(\eta)_{\rm tail}$",
        title=f"{_TITLE_PREFIX}: tail-std spectral entropy vs energy density",
        out_dir=out_dir,
        stem="fig2_entropy_tailstd_vs_epsilon",
    )


def plot_mode1_tail_mean(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Figure 3: tail-mean Mode1/TotalEnergy vs epsilon."""
    df = df.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["mode1_frac"] = df["mode1_tail_mean"] / df["totalenergy_tail_mean"]
    return _scatter_line_by_n(
        df,
        "mode1_frac",
        ylabel=r"tail-mean  $\langle E_1/E_{\rm tot}\rangle_{\rm tail}$",
        title=f"{_TITLE_PREFIX}: tail-mean mode-1 energy fraction vs energy density",
        out_dir=out_dir,
        stem="fig3_mode1_fraction_vs_epsilon",
    )


def plot_max_energy_error(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Figure 4: max relative energy error vs epsilon (log-log)."""
    return _scatter_line_by_n(
        df,
        "max_abs_rel_energy_error",
        ylabel=r"max relative energy error  $\max_t |E(t)-E_0|/|E_0|$",
        title=f"{_TITLE_PREFIX}: max relative energy error vs energy density",
        out_dir=out_dir,
        stem="fig4_max_energy_error_vs_epsilon",
        logy=True,
    )


def plot_amplitude_vs_epsilon(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Figure 5: amplitude vs epsilon (auxiliary diagnostic)."""
    return _scatter_line_by_n(
        df,
        "amplitude",
        ylabel="initial amplitude  A",
        title=f"{_TITLE_PREFIX}: amplitude vs energy density (diagnostic)",
        out_dir=out_dir,
        stem="fig5_amplitude_vs_epsilon",
        logy=True,
    )


def plot_collapse(
    df: pd.DataFrame,
    curves: list[Curve],
    result: CollapseResult,
    out_dir: Path,
) -> list[Path]:
    """Figure 6: finite-size-collapse comparison over the common epsilon range.

    Faint full curves for context, bold segments over the common overlap
    interval, and a shaded band marking the interval used for the spread metric.
    """
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    lo, hi = result.common_lo, result.common_hi
    have_interval = np.isfinite(lo) and np.isfinite(hi) and hi > lo

    for n, sub in _by_n(df):
        eps = sub["epsilon"].to_numpy()
        eta = sub["eta_tail_mean"].to_numpy()
        (line,) = ax.plot(eps, eta, marker="o", markersize=3, linewidth=0.8, alpha=0.35)
        if have_interval:
            m = (eps >= lo) & (eps <= hi)
            ax.plot(
                eps[m], eta[m], marker="o", markersize=5, linewidth=1.8,
                color=line.get_color(), label=f"N = {n}",
            )
        else:
            line.set_label(f"N = {n}")
            line.set_alpha(0.9)

    if have_interval:
        ax.axvspan(lo, hi, color="grey", alpha=0.12, label="common $\\epsilon$ range")
        subtitle = (
            f"common $\\epsilon\\in[{lo:.2e},\\,{hi:.2e}]$, "
            f"RMS spread $={result.rms_spread:.2e}$"
        )
    else:
        subtitle = "no common epsilon overlap across N"

    ax.set_xscale("log")
    ax.set_xlabel(r"energy density  $\epsilon = H(0)/(N-1)$")
    ax.set_ylabel(r"tail-mean spectral entropy  $\langle\eta\rangle_{\rm tail}$")
    ax.set_title(f"{_TITLE_PREFIX}: finite-size collapse\n{subtitle}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="system size", frameon=True, fontsize=9)
    return _save(fig, out_dir, "fig6_finite_size_collapse")


def make_all_figures(
    df: pd.DataFrame,
    curves: list[Curve],
    collapse_result: CollapseResult,
    out_dir: str | Path,
) -> list[Path]:
    """Produce all six figures. Returns every written path (PNG and PDF)."""
    out = Path(out_dir)
    written: list[Path] = []
    if df.empty:
        return written
    written += plot_entropy_tail_mean(df, out)
    written += plot_entropy_tail_std(df, out)
    written += plot_mode1_tail_mean(df, out)
    written += plot_max_energy_error(df, out)
    written += plot_amplitude_vs_epsilon(df, out)
    written += plot_collapse(df, curves, collapse_result, out)
    return written
