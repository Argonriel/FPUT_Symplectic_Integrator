"""Toda integral J: correctness, quadratic limit, N-scaling, edges, conservation.

Covers required tests 3–8.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import toda_reference as ref  # noqa: E402


# --- 3. C++ exact J agrees with the independent Python reference -------------
@pytest.mark.parametrize("N", [4, 8, 13, 32])
def test_cpp_J_matches_python(selftest, N):
    rng = np.random.default_rng(N)
    M = N - 1
    for _ in range(5):
        x = rng.normal(0, 0.4, M)
        v = rng.normal(0, 0.4, M)
        j_cpp = selftest("toda_J", N, arrays=[x, v])[0]
        j_py = ref.exact_toda_J(x, v)
        assert j_cpp == pytest.approx(j_py, rel=1e-12, abs=1e-12)


# --- 4. exact and quadratic J approach each other as amplitude -> 0 ----------
def test_exact_approaches_quadratic_at_low_amplitude(selftest):
    N = 16
    M = N - 1
    rng = np.random.default_rng(1)
    x0 = rng.normal(0, 1.0, M)
    v0 = rng.normal(0, 1.0, M)
    prev = None
    for s in (1e-1, 1e-2, 1e-3, 1e-4):
        x, v = s * x0, s * v0
        j_exact = ref.exact_toda_J(x, v)
        j_quad = ref.quadratic_toda_J(x, v, 1.0, "alpha")
        # Both -> 0; compare the deviation from the common leading behaviour.
        diff = abs(j_exact - j_quad)
        if prev is not None:
            # Deviation shrinks faster than the scale factor (it is higher order).
            assert diff < prev
        prev = diff
    assert diff < 1e-10


# --- 5. J is finite and correctly normalized for multiple N ------------------
@pytest.mark.parametrize("N", [4, 16, 64, 128])
def test_J_finite_and_ground_state_zero(selftest, N):
    M = N - 1
    zeros = np.zeros(M)
    # Ground state (x=v=0): every per-bond term is 3/8 - 3/8 = 0 -> J = 0 exactly.
    j0 = selftest("toda_J", N, arrays=[zeros, zeros])[0]
    assert j0 == pytest.approx(0.0, abs=1e-13)
    # A finite state gives a finite, intensive (O(1)) value.
    rng = np.random.default_rng(N)
    x = rng.normal(0, 0.2, M)
    v = rng.normal(0, 0.2, M)
    j = selftest("toda_J", N, arrays=[x, v])[0]
    assert np.isfinite(j)


def test_intensive_normalization_uses_M_not_N(selftest):
    """J must use 1/(2M)=1/(2(N-1)), never 1/(2N).

    Build a state where every bond term equals a known constant, so the sum is
    (M+1)*const and J = (M+1)*const/(2M). Verify against 1/(2M), which is
    numerically distinct from 1/(2N).
    """
    N = 10
    M = N - 1
    x = np.zeros(M)  # all dq: dq_0=0.., all zero -> b_n = 1
    v = np.zeros(M)
    # With x=v=0 the per-bond term is exactly 0, so use a uniform momentum tweak.
    v = np.full(M, 0.1)
    j = selftest("toda_J", N, arrays=[x, v])[0]
    j_py = ref.exact_toda_J(x, v)
    assert j == pytest.approx(j_py, rel=1e-12)
    # Recompute the reference sum and confirm the denominator is 2M.
    dq = ref.bond_displacements(x)
    b = np.exp(2 * dq)
    p = ref.momenta_padded(v)
    b_lo = np.concatenate(([b[0]], b[:-1]))
    b_hi = np.concatenate((b[1:], [b[-1]]))
    p_n, p_n1 = p[:M + 1], p[1:M + 2]
    s = np.sum(p_n**4 + b * (p_n**2 + p_n * p_n1 + p_n1**2)
               + (b / 8) * (b_lo + b + b_hi) - 3 / 8)
    assert j == pytest.approx(s / (2 * M), rel=1e-12)
    assert j != pytest.approx(s / (2 * N), rel=1e-6)


# --- 6. Fixed-boundary edge cases explicitly tested -------------------------
def test_boundary_terms_use_ghost_continuation(selftest):
    """Only the first/last bond nonzero; check n=0 and n=M contributions.

    Put a single nonzero displacement at the left edge so only dq_0, dq_1 differ
    from zero; the ghost rule b_{-1}=b_0 must be used at n=0.
    """
    N = 6
    M = N - 1
    x = np.zeros(M)
    x[0] = 0.15  # affects dq_0 (=x0) and dq_1 (=x1-x0)
    v = np.zeros(M)
    j_cpp = selftest("toda_J", N, arrays=[x, v])[0]
    j_py = ref.exact_toda_J(x, v)
    assert j_cpp == pytest.approx(j_py, rel=1e-12)

    # Symmetric case at the right edge should give the same J by mirror symmetry
    # (the functional is symmetric under bond reversal for a symmetric state).
    xr = np.zeros(M)
    xr[-1] = -0.15  # mirror of the left-edge state
    j_r = selftest("toda_J", N, arrays=[xr, v])[0]
    assert j_r == pytest.approx(j_cpp, rel=1e-10)


def test_edge_momentum_terms(selftest):
    """p_0 = p_{M+1} = 0: a single nonzero edge momentum exercises p_n p_{n+1}."""
    N = 5
    M = N - 1
    x = np.zeros(M)
    v = np.zeros(M)
    v[0] = 0.3
    j_cpp = selftest("toda_J", N, arrays=[x, v])[0]
    j_py = ref.exact_toda_J(x, v)
    assert j_cpp == pytest.approx(j_py, rel=1e-12)


# --- 7. alpha and beta use the SAME J observable ----------------------------
def test_J_is_model_independent_observable(selftest):
    """J depends only on (x, v); the selftest toda_J takes no model argument."""
    N = 12
    M = N - 1
    rng = np.random.default_rng(7)
    x = rng.normal(0, 0.3, M)
    v = rng.normal(0, 0.3, M)
    j = selftest("toda_J", N, arrays=[x, v])[0]
    # Python reference likewise has no model parameter for exact J.
    assert j == pytest.approx(ref.exact_toda_J(x, v), rel=1e-12)


# --- 8. J conserved to integrator precision along genuine Toda dynamics ------
def test_J_conserved_along_toda_dynamics():
    N = 16
    M = N - 1
    # Mode-1-like smooth IC, moderate amplitude.
    j0_x = 0.5 * np.sin(np.pi * np.arange(1, M + 1) / N)
    v0 = np.zeros(M)
    dt = 0.02
    traj = ref.integrate_toda(j0_x, v0, dt, n_steps=2000)
    Js = np.array([ref.exact_toda_J(x, v) for (x, v) in traj])
    rel_var = (Js.max() - Js.min()) / abs(Js[0])
    # Yoshida-4 with dt=0.02 conserves the exact Toda integral to ~O(dt^4).
    assert rel_var < 1e-6, f"J not conserved along Toda dynamics: rel_var={rel_var:.2e}"
