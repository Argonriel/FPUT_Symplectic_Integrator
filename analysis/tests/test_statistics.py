"""Tests for per-run scientific quantities."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from analysis.statistics import (
    analytic_h0,
    energy_errors,
    epsilon_from_h0,
    relative_discrepancy,
    tail_slice,
    tail_start_index,
    tail_stats,
)


def test_epsilon_divides_by_n_minus_one():
    # epsilon = H(0) / (N - 1), NOT H(0)/N.
    assert epsilon_from_h0(100.0, 101) == pytest.approx(1.0)
    assert epsilon_from_h0(10.0, 6) == pytest.approx(2.0)


def test_epsilon_requires_n_gt_one():
    with pytest.raises(ValueError):
        epsilon_from_h0(1.0, 1)


def test_analytic_h0_matches_closed_form():
    n, amp, beta = 512, 8.0, 1.0
    z = amp**2 * math.sin(math.pi / (2 * n)) ** 2
    expected = n * (z + 1.5 * beta * z**2)
    assert analytic_h0(amp, n, beta) == pytest.approx(expected, rel=1e-12)


def test_analytic_h0_beta_term_scales_with_beta():
    n, amp = 256, 10.0
    linear = analytic_h0(amp, n, beta=0.0)
    with_beta = analytic_h0(amp, n, beta=1.0)
    assert with_beta > linear  # positive quartic contribution


def test_tail_start_index_matches_aggregate_eta_convention():
    # At tail_fraction=0.20 the tail begins at int(0.8*n) (aggregate_eta.py logic).
    for n in (5, 10, 100, 501):
        assert tail_start_index(n, 0.20) == int(0.8 * n)


def test_tail_start_index_bounds():
    assert tail_start_index(0, 0.2) == 0
    assert tail_start_index(1, 0.2) == 0  # at least one tail sample
    with pytest.raises(ValueError):
        tail_start_index(10, 0.0)
    with pytest.raises(ValueError):
        tail_start_index(10, 1.5)


def test_tail_slice_selects_trailing_samples():
    s = pd.Series(range(10))
    assert list(tail_slice(s, 0.20)) == [8, 9]


def test_tail_stats_values():
    s = pd.Series([0.0] * 8 + [2.0, 4.0])  # tail (0.2) = [2, 4]
    st = tail_stats(s, 0.20)
    assert st.mean == pytest.approx(3.0)
    assert st.min == pytest.approx(2.0)
    assert st.max == pytest.approx(4.0)
    assert st.count == 2
    assert st.std == pytest.approx(np.std([2.0, 4.0], ddof=1))


def test_relative_discrepancy():
    assert relative_discrepancy(1.1, 1.0) == pytest.approx(0.1)
    assert math.isnan(relative_discrepancy(1.0, 0.0))


def test_energy_errors_reference_is_initial_value():
    e = pd.Series([100.0, 100.0, 100.5, 100.0])  # spike +0.5%
    max_rel, final_rel = energy_errors(e)
    assert max_rel == pytest.approx(0.005)
    assert final_rel == pytest.approx(0.0)
