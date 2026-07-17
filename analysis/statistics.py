"""Per-run scientific quantities for FPUT-beta trajectories.

All definitions here are documented in ``docs/analysis_pipeline.md``. Key ones:

Energy density (the main control variable)::

    epsilon = H(0) / (N - 1)

where ``H(0)`` is the *initial* ``TotalEnergy`` value stored in the CSV (the
primary source of truth) and ``N - 1`` is the number of moving interior
particles in this solver's fixed-boundary convention.

Analytic cross-check for the pure mode-1 beta initial condition::

    z           = A**2 * sin**2(pi / (2 N))
    H0_analytic = N * (z + (3/2) * beta * z**2)

Note the deliberate asymmetry: ``H0_analytic`` sums over ``N`` bonds while
``epsilon`` divides by ``N - 1`` moving particles. Both are intentional and
documented. Also note this ``epsilon`` differs from the ``E/N`` convention used
in some papers by a factor ``N/(N-1)``.

Tail statistics use the last ``tail_fraction`` of saved samples. At the default
``tail_fraction = 0.20`` the tail-mean of ``Eta`` reduces to the existing
``aggregate_eta.py`` logic ``df["Eta"].iloc[int(0.8*len(df)):].mean()``; that
logic is refactored here into :func:`tail_start_index` / :func:`tail_stats` so
there is a single implementation.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Columns every accepted beta trajectory must contain. Modes beyond Mode1 are
# not all required for the summary, but the header always writes Mode1..Mode20.
REQUIRED_COLUMNS = ("Time", "Mode1", "TotalEnergy", "Eta")


def tail_start_index(n: int, tail_fraction: float) -> int:
    """Index at which the tail window begins for ``n`` samples.

    The tail is the last ``tail_fraction`` of the samples. At
    ``tail_fraction=0.20`` this equals ``int(0.8 * n)``, matching the existing
    ``aggregate_eta.py`` convention.

    Raises
    ------
    ValueError
        If ``tail_fraction`` is not in the half-open interval ``(0, 1]``.
    """
    if not (0.0 < tail_fraction <= 1.0):
        raise ValueError(f"tail_fraction must be in (0, 1], got {tail_fraction}")
    if n <= 0:
        return 0
    start = int((1.0 - tail_fraction) * n)
    # Guarantee at least one sample in the tail.
    return min(start, n - 1)


def tail_slice(series: pd.Series, tail_fraction: float) -> pd.Series:
    """Return the tail-window slice of a Series (last ``tail_fraction`` of samples)."""
    start = tail_start_index(len(series), tail_fraction)
    return series.iloc[start:]


@dataclass(frozen=True)
class TailStats:
    """Summary statistics of a quantity over the tail window."""

    mean: float
    std: float
    min: float
    max: float
    count: int


def tail_stats(series: pd.Series, tail_fraction: float) -> TailStats:
    """Compute mean/std/min/max/count of ``series`` over its tail window.

    ``std`` uses the sample standard deviation (ddof=1) to match pandas defaults;
    it is ``nan`` for a single-sample tail.
    """
    tail = tail_slice(series, tail_fraction).to_numpy(dtype=float)
    if tail.size == 0:
        return TailStats(math.nan, math.nan, math.nan, math.nan, 0)
    std = float(np.std(tail, ddof=1)) if tail.size > 1 else math.nan
    return TailStats(
        mean=float(np.mean(tail)),
        std=std,
        min=float(np.min(tail)),
        max=float(np.max(tail)),
        count=int(tail.size),
    )


def analytic_h0(amplitude: float, n: int, beta: float) -> float:
    """Analytic ``H(0)`` for the pure mode-1 beta initial condition.

    ``z = A**2 * sin**2(pi/(2N))``; ``H0 = N * (z + 1.5 * beta * z**2)``. Sums
    over ``N`` bonds (intentionally different from the ``N-1`` used for epsilon).
    """
    z = amplitude**2 * math.sin(math.pi / (2.0 * n)) ** 2
    return n * (z + 1.5 * beta * z**2)


def epsilon_from_h0(h0: float, n: int) -> float:
    """Energy density ``epsilon = H(0) / (N - 1)`` (interior moving particles)."""
    if n <= 1:
        raise ValueError(f"N must be > 1 to define epsilon, got {n}")
    return h0 / (n - 1)


def relative_discrepancy(analytic: float, measured: float) -> float:
    """Signed relative discrepancy ``(analytic - measured) / measured``.

    Returns ``nan`` if ``measured`` is zero.
    """
    if measured == 0 or not math.isfinite(measured):
        return math.nan
    return (analytic - measured) / measured


def energy_errors(total_energy: pd.Series) -> tuple[float, float]:
    """Relative total-energy error series reduced to two scalars.

    Returns ``(max_abs_rel_error, final_rel_error)`` where the reference is the
    first sample ``E(0)``. For a good Yoshida-4 run both are ~1e-6.
    """
    e = total_energy.to_numpy(dtype=float)
    if e.size == 0 or e[0] == 0 or not math.isfinite(e[0]):
        return math.nan, math.nan
    rel = np.abs(e - e[0]) / abs(e[0])
    return float(np.max(rel)), float(rel[-1])


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file's raw bytes (never loads the whole file)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class RunStats:
    """All per-run quantities for one accepted beta trajectory.

    Field names are the column names emitted to ``beta_runs.csv`` (with
    ``source_path`` and ``sha256`` added by the caller from file identity).
    """

    # timing (actual, read from the Time column)
    first_saved_time: float
    last_saved_time: float
    n_saved_rows: int
    # timing (nominal, derived from metadata)
    nominal_duration: float
    metadata_last_saved_time: float
    # energy
    H0: float
    epsilon: float
    H0_analytic: float
    rel_analytic_energy_discrepancy: float
    max_abs_rel_energy_error: float
    final_rel_energy_error: float
    # entropy (primary observable)
    final_eta: float
    eta_tail_mean: float
    eta_tail_std: float
    eta_tail_min: float
    eta_tail_max: float
    # auxiliary tail statistics
    mode1_tail_mean: float
    mode1_tail_std: float
    totalenergy_tail_mean: float
    totalenergy_tail_std: float
    # quality signals surfaced for the data-quality report
    has_nonfinite: bool = False
    extra: dict = field(default_factory=dict)


def compute_run_stats(
    df: pd.DataFrame,
    *,
    n: int,
    beta: float,
    amplitude: float,
    nominal_duration: float,
    metadata_last_saved_time: float,
    tail_fraction: float,
) -> RunStats:
    """Compute every per-run quantity from a trajectory DataFrame + its metadata.

    ``df`` must already have stripped column names and contain
    :data:`REQUIRED_COLUMNS`. ``Eta`` is used verbatim (the CSV already stores
    normalized spectral entropy — it is never re-normalized here).
    """
    time = df["Time"]
    total_energy = df["TotalEnergy"]
    eta = df["Eta"]
    mode1 = df["Mode1"]

    h0 = float(total_energy.iloc[0])
    eps = epsilon_from_h0(h0, n)
    h0_an = analytic_h0(amplitude, n, beta)
    max_rel, final_rel = energy_errors(total_energy)

    eta_t = tail_stats(eta, tail_fraction)
    mode1_t = tail_stats(mode1, tail_fraction)
    energy_t = tail_stats(total_energy, tail_fraction)

    # Non-finite anywhere in the physically meaningful columns.
    finite_cols = [c for c in ("Time", "Mode1", "TotalEnergy", "Eta") if c in df.columns]
    has_nonfinite = bool(
        not np.isfinite(df[finite_cols].to_numpy(dtype=float)).all()
    )

    return RunStats(
        first_saved_time=float(time.iloc[0]),
        last_saved_time=float(time.iloc[-1]),
        n_saved_rows=int(len(df)),
        nominal_duration=float(nominal_duration),
        metadata_last_saved_time=float(metadata_last_saved_time),
        H0=h0,
        epsilon=eps,
        H0_analytic=h0_an,
        rel_analytic_energy_discrepancy=relative_discrepancy(h0_an, h0),
        max_abs_rel_energy_error=max_rel,
        final_rel_energy_error=final_rel,
        final_eta=float(eta.iloc[-1]),
        eta_tail_mean=eta_t.mean,
        eta_tail_std=eta_t.std,
        eta_tail_min=eta_t.min,
        eta_tail_max=eta_t.max,
        mode1_tail_mean=mode1_t.mean,
        mode1_tail_std=mode1_t.std,
        totalenergy_tail_mean=energy_t.mean,
        totalenergy_tail_std=energy_t.std,
        has_nonfinite=has_nonfinite,
    )
