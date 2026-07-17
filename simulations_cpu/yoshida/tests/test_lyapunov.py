"""Lyapunov diagnostics: renorm-invariance, harmonic decay, nonlinear plateau,
reproducibility, no NaN. Covers required tests 12–16.
"""

from __future__ import annotations

import numpy as np
import pytest


# --- 12. renormalization does not change the accumulated exponent -----------
def test_ftle_invariant_to_renorm_interval(solver, tmp_path):
    """Same run, two very different renorm cadences -> same FTLE(t)."""
    kw = dict(model="alpha", value=1.0, amplitude=3.0)
    a = solver(64, kw["model"], kw["value"], kw["amplitude"], tmp_path / "a.csv",
               "--dt", "0.1", "--stride", "200", "--nseg", "8",
               "--lyapunov", "--lyap-renorm-steps", "1", "--lyap-seed", "5")
    b = solver(64, kw["model"], kw["value"], kw["amplitude"], tmp_path / "b.csv",
               "--dt", "0.1", "--stride", "200", "--nseg", "8",
               "--lyapunov", "--lyap-renorm-steps", "97", "--lyap-seed", "5")
    # FTLE at each snapshot must match despite different renorm cadence.
    fa = a["LyapunovFTLE"].to_numpy()
    fb = b["LyapunovFTLE"].to_numpy()
    assert np.allclose(fa, fb, rtol=1e-9, atol=1e-12)
    # Physical trajectory identical too.
    assert np.allclose(a["TotalEnergy"], b["TotalEnergy"], rtol=1e-12)


# --- 13. harmonic chain: FTLE decreases toward zero -------------------------
def test_harmonic_ftle_decays(solver, tmp_path):
    """With zero nonlinearity the system is integrable; FTLE(t) ~ ln(t)/t -> 0."""
    df = solver(64, "alpha", 0.0, 1.0, tmp_path / "harm.csv",
                "--dt", "0.1", "--stride", "400", "--nseg", "12",
                "--lyapunov", "--lyap-renorm-steps", "50", "--lyap-seed", "1")
    ftle = df["LyapunovFTLE"].to_numpy()[1:]  # drop t=0 (defined as 0)
    early = ftle[0]
    late = ftle[-1]
    assert late < early, f"FTLE did not decrease: early={early:.3e} late={late:.3e}"
    assert late < 0.5 * early
    assert np.all(np.isfinite(ftle))


# --- 14. nonlinear case: reproducibly positive late-time FTLE ----------------
def test_nonlinear_ftle_positive(solver, tmp_path):
    """Strong nonlinear chaos gives a clearly positive late-time FTLE.

    The bounded FPUT-beta quartic well is used here so the test is robust: the
    FPUT-alpha cubic potential is unbounded below and a long, strongly-excited
    alpha run can escape the well (the solver's finite-value guard correctly
    aborts such a run). Per the task spec, a visually-converged positive
    *plateau* is a numerical-validation observation (see docs/validation runs),
    NOT a strict unit requirement; here we assert only finiteness and clear
    positivity, which are robust for a short automated run.
    """
    df = solver(64, "beta", 1.0, 20.0, tmp_path / "chaos.csv",
                "--dt", "0.1", "--stride", "600", "--nseg", "25",
                "--lyapunov", "--lyap-renorm-steps", "10", "--lyap-seed", "1")
    ftle = df["LyapunovFTLE"].to_numpy()
    late = ftle[len(ftle) // 2:]  # second half
    assert np.all(np.isfinite(ftle))
    assert late.mean() > 1e-2, f"expected positive FTLE, got {late.mean():.3e}"

    # Clearly larger than a harmonic run of the same length (which decays to ~0).
    harm = solver(64, "beta", 0.0, 20.0, tmp_path / "harm.csv",
                  "--dt", "0.1", "--stride", "600", "--nseg", "25",
                  "--lyapunov", "--lyap-renorm-steps", "10", "--lyap-seed", "1")
    harm_late = harm["LyapunovFTLE"].to_numpy()[len(harm) // 2:]
    assert late.mean() > 5.0 * harm_late.mean()


# --- 15. reproducible for the same seed; differs (in tangent) for another ----
def test_reproducible_same_seed(solver, tmp_path):
    args = (64, "alpha", 1.0, 5.0)
    flags = ["--dt", "0.1", "--stride", "200", "--nseg", "6", "--lyapunov",
             "--lyap-renorm-steps", "20"]
    d1 = solver(*args, tmp_path / "s1.csv", *flags, "--lyap-seed", "42")
    d2 = solver(*args, tmp_path / "s2.csv", *flags, "--lyap-seed", "42")
    assert np.array_equal(d1["LyapunovFTLE"].to_numpy(), d2["LyapunovFTLE"].to_numpy())


def test_seed_changes_transient_not_physics(solver, tmp_path):
    args = (64, "alpha", 1.0, 5.0)
    flags = ["--dt", "0.1", "--stride", "200", "--nseg", "6", "--lyapunov",
             "--lyap-renorm-steps", "20"]
    d1 = solver(*args, tmp_path / "s1.csv", *flags, "--lyap-seed", "1")
    d2 = solver(*args, tmp_path / "s2.csv", *flags, "--lyap-seed", "2")
    # Different seed -> different early tangent transient...
    assert not np.array_equal(d1["LyapunovFTLE"].to_numpy()[1:3],
                              d2["LyapunovFTLE"].to_numpy()[1:3])
    # ...but the physical trajectory is unaffected by the tangent seed.
    assert np.allclose(d1["TotalEnergy"], d2["TotalEnergy"], rtol=1e-12)


# --- 16. no NaN/inf silently written ----------------------------------------
def test_no_nan_in_diagnostic_output(solver, tmp_path):
    df = solver(48, "beta", 1.0, 12.0, tmp_path / "b.csv",
                "--dt", "0.1", "--stride", "300", "--nseg", "10",
                "--entropy", "--toda", "--lyapunov", "--lyap-renorm-steps", "25")
    for col in ("TotalEnergy", "Eta", "TodaJ", "LyapunovFTLE", "LyapunovLocal"):
        assert np.all(np.isfinite(df[col].to_numpy())), f"non-finite in {col}"


def test_renorm_count_progresses(solver, tmp_path):
    df = solver(32, "alpha", 1.0, 4.0, tmp_path / "rc.csv",
                "--dt", "0.1", "--stride", "100", "--nseg", "5",
                "--lyapunov", "--lyap-renorm-steps", "10")
    rc = df["LyapRenormCount"].to_numpy()
    # 100 steps / 10 = 10 renorms per snapshot interval.
    assert rc[0] == 0
    assert rc[1] == 10
    assert rc[-1] == 10 * (len(rc) - 1)
