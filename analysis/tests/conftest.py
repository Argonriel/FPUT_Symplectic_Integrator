"""Shared test helpers.

Fixture CSVs are generated in-code (into ``tmp_path``) rather than checked in,
because ``.gitignore`` ignores ``*.csv`` globally — a committed fixture would be
invisible to a fresh clone.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

# 20 mode columns, matching the real solver output.
_MODE_COLS = [f"Mode{i}" for i in range(1, 21)]
_HEADER_COLS = ["Time"] + _MODE_COLS + ["TotalEnergy", "Eta"]


def analytic_h0(amplitude: float, n: int, beta: float) -> float:
    """Reference analytic H0 (duplicated in the test to guard the implementation)."""
    z = amplitude**2 * math.sin(math.pi / (2.0 * n)) ** 2
    return n * (z + 1.5 * beta * z**2)


def write_run_csv(
    path: Path,
    *,
    integrator: str = "Yoshida4",
    model: str = "beta",
    n: int = 512,
    beta: float = 1.0,
    amplitude: float = 8.0,
    dt: float = 0.1,
    stride: int = 2_800_000,
    num_segments: int = 500,
    n_rows: int | None = None,
    energy_drift: float = 3e-6,
    eta_tail_value: float = 0.5,
    nonfinite: bool = False,
) -> Path:
    """Write a synthetic run CSV with a realistic commented header.

    ``H(0)`` (first TotalEnergy) is set to the analytic value so epsilon and the
    analytic cross-check are exercised end to end. ``energy_drift`` controls the
    max relative deviation of TotalEnergy from its initial value.
    """
    rows = n_rows if n_rows is not None else num_segments + 1
    h0 = analytic_h0(amplitude, n, beta)
    times = np.arange(rows, dtype=float) * (stride * dt)

    # TotalEnergy: flat then a single spike of relative size ``energy_drift``.
    energy = np.full(rows, h0, dtype=float)
    if rows > 2:
        energy[rows // 2] = h0 * (1.0 + energy_drift)

    # Eta: ramps early, settles to ``eta_tail_value`` over the tail.
    eta = np.full(rows, eta_tail_value, dtype=float)
    if rows > 1:
        eta[0] = 0.0

    # Mode1 carries a small fraction of the energy in the tail.
    mode1 = np.full(rows, 0.05 * h0, dtype=float)
    mode1[0] = h0  # all energy in mode 1 initially

    if nonfinite and rows > 3:
        eta[3] = math.nan

    header = (
        f"# Integrator: {integrator}\n"
        f"# Model: {model}\n"
        f"# N: {n}\n"
        f"# Beta: {beta}\n"
        f"# Amplitude: {amplitude}\n"
        f"# dt: {dt}\n"
        f"# Stride: {stride}\n"
        f"# NumSegments: {num_segments}\n"
        f"# Shape: 0\n"
        f"# Entropy: 1\n"
    )

    with open(path, "w") as f:
        f.write(header)
        f.write(",".join(_HEADER_COLS) + "\n")
        for i in range(rows):
            vals = [f"{times[i]:.15e}"]
            vals.append(f"{mode1[i]:.15e}")
            vals.extend(f"{1e-6:.15e}" for _ in range(19))  # Mode2..Mode20
            vals.append(f"{energy[i]:.15e}")
            vals.append(f"{eta[i]:.15e}")
            f.write(",".join(vals) + "\n")
    return path


@pytest.fixture
def make_csv(tmp_path):
    """Factory fixture: ``make_csv(name, **kwargs) -> Path``."""
    def _make(name: str, **kwargs) -> Path:
        return write_run_csv(tmp_path / name, **kwargs)
    return _make
