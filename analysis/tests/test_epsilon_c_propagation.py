"""Tests for the canonical beta epsilon_c error propagation AND the report layer.

The original bug lived DOWNSTREAM of the math, in the Markdown fit-table
transcription (sigma_epsilon_c had been copied from epsilon_c). These tests guard
both the propagation formula and the real rendering path
(``analysis.beta_gap_v1_analysis.render_fit_table_md``).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from analysis.beta_gap_v1_analysis import (
    epsilon_c_and_sigma,
    eps_of_A,
    deps_dA,
    make_fit_record,
    render_fit_table_md,
    FIT_TABLE_COLUMNS,
)

# (N, Ac, sigma_Ac) -> expected (eps_c, sigma_eps_c). Checked at rtol=1e-3.
# NOTE: the expected literals are quoted to 4 significant figures (not 3). Three
# sig figs only bound the relative error to ~5e-3, so a 3-sig-fig literal (e.g.
# eps_c=1.85e-4 vs the true 1.8473e-4) would false-fail at rtol=1e-3. 4 sig figs
# makes rtol=1e-3 both meaningful and non-false-failing; the EXACT invariant is
# covered separately by test_self_consistency_identity (np.isclose rtol=1e-12).
KNOWN = [
    (512, 7.325954652336504, 0.1697610129362528, 5.065e-4, 2.349e-5),
    (1024, 12.46679576052764, 0.49694018693016195, 3.663e-4, 2.922e-5),
    (2048, 21.84554369286018, 0.18743576888935293, 2.810e-4, 4.824e-6),
    (4096, 35.432117545008076, 2.2558481988988834, 1.847e-4, 2.353e-5),
]


@pytest.mark.parametrize("N,Ac,sAc,exp_eps,exp_sig", KNOWN)
def test_known_points(N, Ac, sAc, exp_eps, exp_sig):
    eps_c, sig = epsilon_c_and_sigma(N, Ac, sAc)
    assert eps_c == pytest.approx(exp_eps, rel=1e-3)
    assert sig == pytest.approx(exp_sig, rel=1e-3)


@pytest.mark.parametrize("N,Ac,sAc,exp_eps,exp_sig", KNOWN)
def test_self_consistency_identity(N, Ac, sAc, exp_eps, exp_sig):
    """The real invariant (no literals): sigma_eps_c == |deps/dA(Ac)| * sigma_Ac."""
    _, sig = epsilon_c_and_sigma(N, Ac, sAc)
    assert np.isclose(sig, abs(deps_dA(N, Ac) * sAc), rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("N,Ac,sAc,exp_eps,exp_sig", KNOWN)
def test_finite_difference_derivative(N, Ac, sAc, exp_eps, exp_sig):
    """Numerically differentiate eps(A) and compare to the analytic deps/dA
    (catches a future formula error such as a dropped (1+3z) factor)."""
    h = Ac * 1e-6
    fd = (eps_of_A(N, Ac + h) - eps_of_A(N, Ac - h)) / (2.0 * h)
    assert deps_dA(N, Ac) == pytest.approx(fd, rel=1e-6)


@pytest.mark.parametrize("N,Ac,sAc,exp_eps,exp_sig", KNOWN)
def test_current_dataset_sanity_sigma_vs_eps(N, Ac, sAc, exp_eps, exp_sig):
    """CURRENT-DATASET SANITY CHECK ONLY — not a mathematical invariant.

    For this fitted dataset every sigma_eps_c is well below eps_c (max ratio ~0.13
    at N=4096). A poorly-constrained fit could legitimately violate this, so it is a
    dataset-specific smoke test, NOT an assertion of correctness. Its only job is to
    catch a gross regression to the old sigma_eps_c==eps_c column-swap (ratio ~1.0).
    """
    eps_c, sig = epsilon_c_and_sigma(N, Ac, sAc)
    assert sig < 0.5 * eps_c, f"N={N}: sigma_eps_c={sig:.3e} not << eps_c={eps_c:.3e}"


def test_eps_of_A_matches_frozen_analytic_map():
    from analysis.statistics import analytic_h0, epsilon_from_h0
    for N, Ac, *_ in KNOWN:
        assert eps_of_A(N, Ac) == pytest.approx(
            epsilon_from_h0(analytic_h0(Ac, N, 1.0), N), rel=1e-12)


def test_nan_sigma_when_ac_se_nonfinite():
    eps_c, sig = epsilon_c_and_sigma(512, 7.326, float("nan"))
    assert math.isfinite(eps_c) and math.isnan(sig)


# --------------------------------------------------------------------------
# Report-layer regression tests: the bug was in the RENDERED Markdown table.
# render_fit_table_md is the only producer of the table in beta_report.md.
# --------------------------------------------------------------------------
def _render_rows_for(*points):
    recs = [make_fit_record(N, Ac, sAc, label="fit", converged=True, n_points=20,
                            residual_RMS=0.01, Ac_in_range=True)
            for (N, Ac, sAc) in points]
    md = render_fit_table_md(recs)
    lines = [ln for ln in md.splitlines() if ln.strip().startswith("|")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    idx = {c: i for i, c in enumerate(header)}
    rows = {}
    for ln in lines[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        rows[int(cells[idx["N"]])] = cells
    return header, idx, rows


def test_rendered_table_column_slots():
    """Rendered header must expose Ac, sigma_Ac, epsilon_c, sigma_epsilon_c in order."""
    header, idx, _ = _render_rows_for((512, 7.326, 0.170))
    for col in ("Ac", "sigma_Ac", "epsilon_c", "sigma_epsilon_c"):
        assert col in idx, f"missing column {col} in rendered table {header}"
    assert idx["Ac"] < idx["sigma_Ac"] < idx["epsilon_c"] < idx["sigma_epsilon_c"]
    assert header == FIT_TABLE_COLUMNS


@pytest.mark.parametrize("N,Ac,sAc", [
    (512, 7.325954652336504, 0.1697610129362528),
    (4096, 35.432117545008076, 2.2558481988988834),
])
def test_rendered_sigma_matches_derivative_and_not_epsilon(N, Ac, sAc):
    """Against the REAL render path: emitted sigma_epsilon_c equals |deps/dA|*sigma_Ac
    (rtol 1e-3), is NOT the column-swapped epsilon_c, and lives in the right slot."""
    _, idx, rows = _render_rows_for((512, 7.325954652336504, 0.1697610129362528),
                                    (4096, 35.432117545008076, 2.2558481988988834))
    cells = rows[N]
    emitted_eps = float(cells[idx["epsilon_c"]])
    emitted_sig = float(cells[idx["sigma_epsilon_c"]])
    emitted_Ac = float(cells[idx["Ac"]])
    emitted_sAc = float(cells[idx["sigma_Ac"]])
    # right slots hold the inputs we passed
    assert emitted_Ac == pytest.approx(Ac, rel=1e-3)
    assert emitted_sAc == pytest.approx(sAc, rel=1e-3)
    # sigma equals the analytic propagation (rendered at 3-4 sig figs)
    assert emitted_sig == pytest.approx(abs(deps_dA(N, Ac) * sAc), rel=1e-3)
    # and is NOT the exact column-swap bug (sigma copied from eps_c)
    assert not np.isclose(emitted_sig, emitted_eps, rtol=0.1)
