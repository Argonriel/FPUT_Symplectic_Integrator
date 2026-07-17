"""Tests for descriptive collapse metrics (no extrapolation)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from analysis.collapse import build_curve, compute_collapse, interpolate_on_grid


def test_interpolation_never_extrapolates():
    curve = build_curve(512, epsilon=np.array([1e-3, 1e-2]), eta=np.array([0.2, 0.8]))
    grid = np.array([1e-4, 1e-3, 3e-3, 1e-2, 1e-1])
    out = interpolate_on_grid(curve, grid)
    # Outside [1e-3, 1e-2] -> nan; inside -> finite.
    assert math.isnan(out[0]) and math.isnan(out[4])
    assert np.isfinite(out[1]) and np.isfinite(out[2]) and np.isfinite(out[3])
    # Linear in log-epsilon: midpoint (in log) of [1e-3,1e-2] ~ 3.16e-3.
    mid = interpolate_on_grid(curve, np.array([math.sqrt(1e-3 * 1e-2)]))[0]
    assert mid == pytest.approx(0.5)  # halfway in log space between 0.2 and 0.8


def test_common_interval_is_the_overlap():
    c1 = build_curve(512, np.array([1e-3, 1e-2]), np.array([0.3, 0.5]))
    c2 = build_curve(1024, np.array([3e-3, 5e-2]), np.array([0.35, 0.55]))
    res = compute_collapse([c1, c2], grid_points=20)
    assert res.common_lo == 3e-3   # max of the mins
    assert res.common_hi == 1e-2   # min of the maxes
    assert res.n_values == [512, 1024]
    assert math.isfinite(res.rms_spread)
    # Every grid point lies within the overlap and both curves cover it.
    assert (res.metrics["n_curves"] == 2).all()


def test_no_overlap_reports_note_not_crash():
    c1 = build_curve(512, np.array([1e-3, 2e-3]), np.array([0.3, 0.4]))
    c2 = build_curve(1024, np.array([1e-1, 2e-1]), np.array([0.5, 0.6]))
    res = compute_collapse([c1, c2], grid_points=10)
    assert res.metrics.empty
    assert math.isnan(res.rms_spread)
    assert "no overlapping epsilon interval" in res.note


def test_identical_curves_have_zero_spread():
    eps = np.array([1e-3, 5e-3, 1e-2])
    eta = np.array([0.2, 0.5, 0.8])
    res = compute_collapse([build_curve(512, eps, eta), build_curve(1024, eps, eta)],
                           grid_points=15)
    assert res.rms_spread == 0.0 or res.rms_spread < 1e-12


def test_single_curve_yields_no_metrics():
    res = compute_collapse([build_curve(512, np.array([1e-3, 1e-2]), np.array([0.2, 0.8]))])
    assert res.metrics.empty
    assert "fewer than 2 curves" in res.note
