"""Independent Python reference implementations for the Yoshida-solver diagnostics.

These are written from the mathematical definitions (NOT translated line-by-line
from the C++), so that the pytest suite cross-checks the production kernels
against a genuinely separate implementation. They are for tests/validation only
— never for production simulation.

Conventions (match the C++ solver, ``M = N - 1`` moving interior particles):

    q_0 = q_{M+1} = 0 ,  q_n = x[n-1]  for n = 1..M
    p_0 = p_{M+1} = 0 ,  p_n = v[n-1]  for n = 1..M
    dq_n = q_{n+1} - q_n ,  n = 0..M                 (M+1 = N bonds)

Toda integral J (alpha = 1 form; Christodoulidi & Flach, Chaos 35, 113127
(2025), Eq. 5), with fixed-boundary ghost continuation b_{-1}=b_0, b_{M+1}=b_M.
"""

from __future__ import annotations

import numpy as np

MODEL_ALPHA = "alpha"
MODEL_BETA = "beta"


# --------------------------------------------------------------------------- #
# Bond helpers
# --------------------------------------------------------------------------- #
def bond_displacements(x: np.ndarray) -> np.ndarray:
    """Return ``dq_n`` for ``n = 0..M`` (length ``M+1 = N`` bonds).

    ``dq_0 = x[0] - 0``, ``dq_n = x[n]-x[n-1]``, ``dq_M = 0 - x[M-1]``.
    """
    x = np.asarray(x, dtype=float)
    q = np.concatenate(([0.0], x, [0.0]))  # q_0 .. q_{M+1}
    return np.diff(q)                       # length M+1


def momenta_padded(v: np.ndarray) -> np.ndarray:
    """Return ``p_n`` for ``n = 0..M+1`` with ``p_0 = p_{M+1} = 0``."""
    v = np.asarray(v, dtype=float)
    return np.concatenate(([0.0], v, [0.0]))


# --------------------------------------------------------------------------- #
# Toda integral
# --------------------------------------------------------------------------- #
def exact_toda_J(x: np.ndarray, v: np.ndarray) -> float:
    """Exact intensive 4th-order Toda integral J (alpha = 1 form).

    J = 1/(2M) * sum_{n=0}^{M} [ p_n^4
                                 + b_n (p_n^2 + p_n p_{n+1} + p_{n+1}^2)
                                 + (b_n/8)(b_{n-1} + b_n + b_{n+1})
                                 - 3/8 ]
    with b_n = exp(2 dq_n) and ghost bonds b_{-1}=b_0, b_{M+1}=b_M.
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    M = x.size
    dq = bond_displacements(x)                 # n = 0..M
    b = np.exp(2.0 * dq)                        # n = 0..M
    p = momenta_padded(v)                       # p_0..p_{M+1}

    # b with ghosts: index shift so b_ghost[n+1] == b_n, b_ghost[0]==b_{-1}=b_0,
    # b_ghost[M+2-1]... simpler: build neighbor arrays explicitly.
    b_lo = np.empty_like(b)   # b_{n-1}
    b_hi = np.empty_like(b)   # b_{n+1}
    b_lo[0] = b[0]            # ghost b_{-1} = b_0
    b_lo[1:] = b[:-1]
    b_hi[-1] = b[-1]          # ghost b_{M+1} = b_M
    b_hi[:-1] = b[1:]

    p_n = p[0:M + 1]          # p_0..p_M
    p_n1 = p[1:M + 2]         # p_1..p_{M+1}

    terms = (
        p_n**4
        + b * (p_n**2 + p_n * p_n1 + p_n1**2)
        + (b / 8.0) * (b_lo + b + b_hi)
        - 3.0 / 8.0
    )
    return float(terms.sum() / (2.0 * M))


def quadratic_toda_J(x: np.ndarray, v: np.ndarray, value: float, model: str) -> float:
    """Low-energy quadratic approximation to J (Eq. 6), for validation only.

    J_quad = 2*epsilon + 1/(2M) sum_{n=0}^{M} [ p_n p_{n+1} + dq_n dq_{n+1} ]
    with epsilon = H_FPUT/M and dq_{M+1} = dq_M.
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    M = x.size
    H = total_energy(x, v, value, model)
    epsilon = H / M

    dq = bond_displacements(x)                 # n = 0..M
    dq_ext = np.concatenate((dq, [dq[-1]]))    # append dq_{M+1} = dq_M
    p = momenta_padded(v)                       # p_0..p_{M+1}

    p_n = p[0:M + 1]
    p_n1 = p[1:M + 2]
    dq_n = dq_ext[0:M + 1]
    dq_n1 = dq_ext[1:M + 2]

    corr = np.sum(p_n * p_n1 + dq_n * dq_n1)
    return float(2.0 * epsilon + corr / (2.0 * M))


# --------------------------------------------------------------------------- #
# Energy and forces
# --------------------------------------------------------------------------- #
def total_energy(x: np.ndarray, v: np.ndarray, value: float, model: str) -> float:
    """FPUT total Hamiltonian with fixed ends (matches the C++ solver)."""
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    kin = 0.5 * np.sum(v * v)
    dq = bond_displacements(x)  # all N bonds
    pot = 0.5 * np.sum(dq * dq)
    if model == MODEL_ALPHA:
        pot += (value / 3.0) * np.sum(dq**3)
    else:
        pot += (value / 4.0) * np.sum(dq**4)
    return float(kin + pot)


def forces(x: np.ndarray, value: float, model: str) -> np.ndarray:
    """Force on each interior particle (fixed ends). Matches C++ compute_forces."""
    x = np.asarray(x, dtype=float)
    M = x.size
    prev = np.concatenate(([0.0], x[:-1]))
    nxt = np.concatenate((x[1:], [0.0]))
    dx_f = nxt - x
    dx_b = x - prev
    F = dx_f - dx_b
    if model == MODEL_ALPHA:
        F = F + value * (dx_f**2 - dx_b**2)
    else:
        F = F + value * (dx_f**3 - dx_b**3)
    return F


def tangent_force(x: np.ndarray, dx: np.ndarray, value: float, model: str) -> np.ndarray:
    """Linearized force (DF(x) . dx) with fixed-boundary tangent displacements.

    (DF dx)_j = K_f (dx_{j+1}-dx_j) - K_b (dx_j - dx_{j-1}),
    K(r) = 1 + 2*alpha*r (alpha) or 1 + 3*beta*r^2 (beta).
    """
    x = np.asarray(x, dtype=float)
    dx = np.asarray(dx, dtype=float)
    prev = np.concatenate(([0.0], x[:-1]))
    nxt = np.concatenate((x[1:], [0.0]))
    r_f = nxt - x
    r_b = x - prev
    if model == MODEL_ALPHA:
        K_f = 1.0 + 2.0 * value * r_f
        K_b = 1.0 + 2.0 * value * r_b
    else:
        K_f = 1.0 + 3.0 * value * r_f**2
        K_b = 1.0 + 3.0 * value * r_b**2
    dprev = np.concatenate(([0.0], dx[:-1]))
    dnext = np.concatenate((dx[1:], [0.0]))
    return K_f * (dnext - dx) - K_b * (dx - dprev)


# --------------------------------------------------------------------------- #
# Yoshida integrator (physical and tangent)
# --------------------------------------------------------------------------- #
_CBRT2 = 2.0 ** (1.0 / 3.0)
_W1 = 1.0 / (2.0 - _CBRT2)
_W0 = -_CBRT2 / (2.0 - _CBRT2)
C1 = _W1 * 0.5
C2 = (_W0 + _W1) * 0.5
D1 = _W1
D2 = _W0


def yoshida_step(x, v, dt, value, model, force_fn=None):
    """One physical Yoshida-4 step. Returns new (x, v).

    ``force_fn(x)`` overrides the FPUT force (used for the Toda reference test).
    """
    x = np.array(x, dtype=float)
    v = np.array(v, dtype=float)
    ff = force_fn if force_fn is not None else (lambda xx: forces(xx, value, model))
    x += C1 * dt * v
    v += D1 * dt * ff(x)
    x += C2 * dt * v
    v += D2 * dt * ff(x)
    x += C2 * dt * v
    v += D1 * dt * ff(x)
    x += C1 * dt * v
    return x, v


def yoshida_step_tangent(x, v, dx, dv, dt, value, model):
    """One Yoshida-4 step advancing physical (x,v) and tangent (dx,dv).

    Returns (x, v, dx, dv). Mirrors the C++ yoshida_step_tangent.
    """
    x = np.array(x, dtype=float)
    v = np.array(v, dtype=float)
    dx = np.array(dx, dtype=float)
    dv = np.array(dv, dtype=float)

    def kick(coeff):
        nonlocal v, dv
        v += coeff * forces(x, value, model)
        dv += coeff * tangent_force(x, dx, value, model)

    x += C1 * dt * v
    dx += C1 * dt * dv
    kick(D1 * dt)
    x += C2 * dt * v
    dx += C2 * dt * dv
    kick(D2 * dt)
    x += C2 * dt * v
    dx += C2 * dt * dv
    kick(D1 * dt)
    x += C1 * dt * v
    dx += C1 * dt * dv
    return x, v, dx, dv


# --------------------------------------------------------------------------- #
# Minimal Toda-potential reference dynamics (tests only)
# --------------------------------------------------------------------------- #
def toda_forces(x: np.ndarray) -> np.ndarray:
    """Force from the matched Toda potential V(r) = (exp(2r) - 1 - 2r)/4.

    V'(r) = (exp(2r) - 1)/2 ; F_j = V'(r_f) - V'(r_b) with fixed ends. This is
    the genuine integrable dynamics along which the exact Toda J is conserved.
    """
    x = np.asarray(x, dtype=float)
    prev = np.concatenate(([0.0], x[:-1]))
    nxt = np.concatenate((x[1:], [0.0]))
    r_f = nxt - x
    r_b = x - prev

    def vprime(r):
        return 0.5 * (np.exp(2.0 * r) - 1.0)

    return vprime(r_f) - vprime(r_b)


def integrate_toda(x0, v0, dt, n_steps):
    """Integrate the Toda reference dynamics; return list of (x, v) snapshots."""
    x = np.array(x0, dtype=float)
    v = np.array(v0, dtype=float)
    traj = [(x.copy(), v.copy())]
    for _ in range(n_steps):
        x, v = yoshida_step(x, v, dt, value=1.0, model=MODEL_ALPHA, force_fn=toda_forces)
        traj.append((x.copy(), v.copy()))
    return traj
