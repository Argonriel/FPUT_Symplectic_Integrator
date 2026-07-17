"""Descriptive finite-size-collapse metrics for tail-mean spectral entropy.

This module *quantifies* how close the ``Eta_tail_mean`` vs ``epsilon`` curves
for different ``N`` lie to one another over their common epsilon range. It is
strictly descriptive:

* No stochasticity threshold is fitted.
* No curve is extrapolated: each ``N`` is interpolated only inside its own
  measured epsilon range.
* No universal collapse is auto-declared; an aggregate RMS spread is reported so
  the reader can judge.

The common overlapping interval is expected to be narrow because, e.g., the
``N=4096`` beta runs start at high epsilon and may barely overlap the low-epsilon
end of the smaller-``N`` runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Curve:
    """One ``N``'s (epsilon, eta) curve, cleaned and sorted."""

    n: int
    epsilon: np.ndarray  # strictly increasing, > 0
    eta: np.ndarray

    @property
    def eps_min(self) -> float:
        return float(self.epsilon[0])

    @property
    def eps_max(self) -> float:
        return float(self.epsilon[-1])


@dataclass
class CollapseResult:
    """Outputs of the collapse computation."""

    metrics: pd.DataFrame  # per grid-point statistics
    common_lo: float
    common_hi: float
    n_values: list[int]
    rms_spread: float
    note: str = ""
    extras: dict = field(default_factory=dict)


def build_curve(n: int, epsilon: np.ndarray, eta: np.ndarray) -> Curve:
    """Clean a raw (epsilon, eta) pair into a monotone curve.

    Drops non-finite and non-positive epsilon (log axis), sorts by epsilon, and
    averages eta across any exactly-duplicated epsilon values.
    """
    eps = np.asarray(epsilon, dtype=float)
    y = np.asarray(eta, dtype=float)
    mask = np.isfinite(eps) & np.isfinite(y) & (eps > 0)
    eps, y = eps[mask], y[mask]
    order = np.argsort(eps, kind="stable")
    eps, y = eps[order], y[order]

    # Average duplicate epsilon values so interpolation stays well-defined.
    uniq, inv = np.unique(eps, return_inverse=True)
    if uniq.size != eps.size:
        summed = np.zeros(uniq.size)
        counts = np.zeros(uniq.size)
        np.add.at(summed, inv, y)
        np.add.at(counts, inv, 1.0)
        y = summed / counts
        eps = uniq
    return Curve(n=n, epsilon=eps, eta=y)


def interpolate_on_grid(curve: Curve, grid: np.ndarray) -> np.ndarray:
    """Interpolate ``curve`` onto a shared log-epsilon ``grid`` without extrapolating.

    Grid points outside ``[eps_min, eps_max]`` of this curve return ``nan``.
    Interpolation is linear in ``log(epsilon)``.
    """
    out = np.full(grid.shape, np.nan, dtype=float)
    in_range = (grid >= curve.eps_min) & (grid <= curve.eps_max)
    if curve.epsilon.size >= 2:
        out[in_range] = np.interp(
            np.log(grid[in_range]),
            np.log(curve.epsilon),
            curve.eta,
        )
    elif curve.epsilon.size == 1:
        # A single point can only contribute exactly at its own epsilon.
        exact = in_range & np.isclose(grid, curve.eps_min)
        out[exact] = curve.eta[0]
    return out


def compute_collapse(curves: list[Curve], *, grid_points: int = 50) -> CollapseResult:
    """Compute per-grid-point spread statistics over the common epsilon interval.

    Steps: find the common overlap ``[max(min_i), min(max_i)]``; build a shared
    log-spaced grid inside it; interpolate every curve (no extrapolation); then
    at each grid point compute mean/std/max-min/CV across the ``N`` that cover it.
    An aggregate RMS spread (root-mean-square of the across-N std over the grid)
    is returned.
    """
    usable = [c for c in curves if c.epsilon.size >= 1]
    n_values = sorted(c.n for c in usable)

    empty_metrics = pd.DataFrame(
        columns=["epsilon", "n_curves", "eta_mean", "eta_std", "eta_maxmin", "eta_cv"]
    )

    if len(usable) < 2:
        return CollapseResult(
            metrics=empty_metrics,
            common_lo=math.nan,
            common_hi=math.nan,
            n_values=n_values,
            rms_spread=math.nan,
            note="fewer than 2 curves with data; no collapse computed",
        )

    common_lo = max(c.eps_min for c in usable)
    common_hi = min(c.eps_max for c in usable)

    if not (common_hi > common_lo):
        return CollapseResult(
            metrics=empty_metrics,
            common_lo=common_lo,
            common_hi=common_hi,
            n_values=n_values,
            rms_spread=math.nan,
            note=(
                f"no overlapping epsilon interval across N={n_values} "
                f"(max-min={common_lo:.3e} >= min-max={common_hi:.3e})"
            ),
        )

    grid = np.logspace(math.log10(common_lo), math.log10(common_hi), grid_points)

    # Rows = grid points, cols = curves (sorted by N for determinism).
    usable_sorted = sorted(usable, key=lambda c: c.n)
    interp = np.vstack([interpolate_on_grid(c, grid) for c in usable_sorted]).T

    rows = []
    for i, eps in enumerate(grid):
        vals = interp[i, :]
        finite = vals[np.isfinite(vals)]
        count = int(finite.size)
        if count == 0:
            rows.append((eps, 0, math.nan, math.nan, math.nan, math.nan))
            continue
        mean = float(np.mean(finite))
        std = float(np.std(finite, ddof=1)) if count > 1 else math.nan
        maxmin = float(np.max(finite) - np.min(finite))
        cv = float(std / mean) if (count > 1 and mean != 0) else math.nan
        rows.append((eps, count, mean, std, maxmin, cv))

    metrics = pd.DataFrame(
        rows, columns=["epsilon", "n_curves", "eta_mean", "eta_std", "eta_maxmin", "eta_cv"]
    )

    # Aggregate spread: RMS of the across-N std over grid points where >=2 curves
    # overlap (std is defined there).
    valid_std = metrics.loc[metrics["n_curves"] >= 2, "eta_std"].to_numpy(dtype=float)
    valid_std = valid_std[np.isfinite(valid_std)]
    rms_spread = float(np.sqrt(np.mean(valid_std**2))) if valid_std.size else math.nan

    return CollapseResult(
        metrics=metrics,
        common_lo=common_lo,
        common_hi=common_hi,
        n_values=n_values,
        rms_spread=rms_spread,
        note="ok",
    )
