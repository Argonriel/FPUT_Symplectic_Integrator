"""Tangent dynamics: FD Jacobian, full-step FD, boundaries.

Covers required tests 9–11.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import toda_reference as ref  # noqa: E402


# --- 9. analytic tangent force == centered finite-difference JVP -------------
@pytest.mark.parametrize("model,value", [("alpha", 1.0), ("beta", 1.0), ("alpha", 0.25)])
def test_tangent_force_matches_finite_difference(selftest, model, value):
    N = 20
    M = N - 1
    rng = np.random.default_rng(hash((model, value)) % (2**32))
    x = rng.normal(0, 0.3, M)
    dxv = rng.normal(0, 1.0, M)
    h = 1e-6

    analytic = selftest("tangent_force", N, model, value, arrays=[x, dxv])
    f_plus = selftest("forces", N, model, value, arrays=[x + h * dxv])
    f_minus = selftest("forces", N, model, value, arrays=[x - h * dxv])
    fd = (f_plus - f_minus) / (2 * h)
    assert np.allclose(analytic, fd, rtol=1e-6, atol=1e-7)


# --- 10. one full tangent Yoshida step == FD of the full physical step -------
@pytest.mark.parametrize("model,value", [("alpha", 1.0), ("beta", 0.8)])
def test_full_tangent_step_matches_finite_difference(selftest, model, value):
    N = 16
    M = N - 1
    dt = 0.1
    rng = np.random.default_rng(hash((model, value, "step")) % (2**32))
    x = rng.normal(0, 0.3, M)
    v = rng.normal(0, 0.3, M)
    dx = rng.normal(0, 1.0, M)
    dv = rng.normal(0, 1.0, M)
    h = 1e-6

    out = selftest("tangent_step", N, model, value, dt, arrays=[x, v, dx, dv])
    dx_out = out[2 * M:3 * M]
    dv_out = out[3 * M:4 * M]

    plus = selftest("step", N, model, value, dt, arrays=[x + h * dx, v + h * dv])
    minus = selftest("step", N, model, value, dt, arrays=[x - h * dx, v - h * dv])
    fd = (plus - minus) / (2 * h)
    fd_dx, fd_dv = fd[:M], fd[M:2 * M]

    assert np.allclose(dx_out, fd_dx, rtol=1e-6, atol=1e-7)
    assert np.allclose(dv_out, fd_dv, rtol=1e-6, atol=1e-7)


def test_tangent_step_leaves_physical_trajectory_identical(selftest):
    """The (x,v) produced by the tangent step equals the plain physical step."""
    N = 24
    M = N - 1
    dt = 0.1
    rng = np.random.default_rng(99)
    x = rng.normal(0, 0.3, M)
    v = rng.normal(0, 0.3, M)
    dx = rng.normal(0, 1.0, M)
    dv = rng.normal(0, 1.0, M)
    out = selftest("tangent_step", N, "alpha", 1.0, dt, arrays=[x, v, dx, dv])
    x_t, v_t = out[:M], out[M:2 * M]
    phys = selftest("step", N, "alpha", 1.0, dt, arrays=[x, v])
    assert np.array_equal(x_t, phys[:M])
    assert np.array_equal(v_t, phys[M:2 * M])


# --- 11. tangent respects fixed boundaries ----------------------------------
def test_tangent_force_respects_fixed_boundaries(selftest):
    """The tangent force is a JVP of a fixed-boundary force: interior only, and
    a boundary-localized tangent perturbation propagates correctly (no ghost dof).
    """
    N = 10
    M = N - 1
    x = np.zeros(M)
    # Unit tangent on the first interior particle only.
    dxv = np.zeros(M)
    dxv[0] = 1.0
    tf = selftest("tangent_force", N, "alpha", 1.0, arrays=[x, dxv])
    # At x=0 stiffness is 1 everywhere; DF is the discrete Laplacian.
    # (Laplacian . e_0)_0 = -2, (.)_1 = +1, rest 0.
    expected = np.zeros(M)
    expected[0] = -2.0
    expected[1] = 1.0
    assert np.allclose(tf, expected, atol=1e-12)


def test_tangent_step_python_cpp_agree(selftest):
    """Cross-check the full tangent step against the Python reference."""
    N = 14
    M = N - 1
    dt = 0.1
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.3, M)
    v = rng.normal(0, 0.3, M)
    dx = rng.normal(0, 1.0, M)
    dv = rng.normal(0, 1.0, M)
    out = selftest("tangent_step", N, "beta", 1.0, dt, arrays=[x, v, dx, dv])
    xr, vr, dxr, dvr = ref.yoshida_step_tangent(x, v, dx, dv, dt, 1.0, "beta")
    assert np.allclose(out[:M], xr, rtol=1e-12, atol=1e-14)
    assert np.allclose(out[M:2 * M], vr, rtol=1e-12, atol=1e-14)
    assert np.allclose(out[2 * M:3 * M], dxr, rtol=1e-12, atol=1e-14)
    assert np.allclose(out[3 * M:4 * M], dvr, rtol=1e-12, atol=1e-14)
