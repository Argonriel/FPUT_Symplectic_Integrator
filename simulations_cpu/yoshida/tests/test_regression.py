"""Regression & compatibility: diagnostics-off reproduces the previous solver,
and existing CSV readers still parse the output. Covers required tests 1–2.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

YOSHIDA_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = YOSHIDA_DIR.parent.parent


# --- 1. diagnostics OFF reproduces the previous canonical solver -------------
def _pre_diagnostics_source():
    """Return the solver source from the newest ref that predates the diagnostics
    refactor (single-file, no fput_physics.hpp include). Tries origin/main first
    so the check keeps working after the diagnostics commit lands on a branch."""
    for ref in ("origin/main", "main", "HEAD"):
        try:
            src = subprocess.run(
                ["git", "show", f"{ref}:simulations_cpu/yoshida/FPUT_yoshida_solver.cpp"],
                cwd=REPO_ROOT, capture_output=True, text=True, check=True,
            ).stdout
        except Exception:  # noqa: BLE001
            continue
        if "fput_physics.hpp" not in src:
            return src
    return None


def test_diagnostics_off_matches_baseline(binaries, tmp_path):
    """Build the previous (pre-diagnostics) canonical solver and compare a short
    trajectory: diagnostics-off data rows must be bit-identical."""
    src = _pre_diagnostics_source()
    if src is None:
        pytest.skip("no pre-diagnostics baseline ref available")
    head_src = tmp_path / "head_solver.cpp"
    head_src.write_text(src)

    ref_bin = tmp_path / "fput_head"
    subprocess.run(["c++", "-O3", "-std=c++17", "-o", str(ref_bin), str(head_src)], check=True)

    common = ["--dt", "0.1", "--stride", "100", "--nseg", "6", "--entropy"]
    ref_csv = tmp_path / "ref.csv"
    new_csv = tmp_path / "new.csv"
    subprocess.run([str(ref_bin), "20", "alpha", "1.0", "0.7", str(ref_csv), *common], check=True)
    subprocess.run([str(binaries["solver"]), "20", "alpha", "1.0", "0.7", str(new_csv), *common], check=True)

    a = pd.read_csv(ref_csv, comment="#")
    b = pd.read_csv(new_csv, comment="#")
    a.columns = a.columns.str.strip()
    b.columns = b.columns.str.strip()
    assert list(a.columns) == list(b.columns)
    for col in a.columns:
        assert np.allclose(a[col].to_numpy(), b[col].to_numpy(), rtol=0, atol=0), \
            f"column {col} differs from the pre-diagnostics baseline"


# --- 2. existing CSV readers still parse the output --------------------------
def test_default_schema_unchanged(solver, tmp_path):
    df = solver(16, "beta", 1.0, 2.0, tmp_path / "d.csv",
                "--dt", "0.1", "--stride", "50", "--nseg", "4")
    expected = ["Time"] + [f"Mode{i}" for i in range(1, 21)] + ["TotalEnergy"]
    assert list(df.columns) == expected


def test_entropy_schema_unchanged(solver, tmp_path):
    df = solver(16, "beta", 1.0, 2.0, tmp_path / "d.csv",
                "--dt", "0.1", "--stride", "50", "--nseg", "4", "--entropy")
    expected = ["Time"] + [f"Mode{i}" for i in range(1, 21)] + ["TotalEnergy", "Eta"]
    assert list(df.columns) == expected


def test_get_metadata_parses_new_keys(solver, tmp_path):
    """visualization/plot_utils.get_metadata must read the new metadata lines."""
    sys.path.insert(0, str(REPO_ROOT / "visualization"))
    from plot_utils import get_metadata

    out = tmp_path / "meta.csv"
    solver(16, "alpha", 1.0, 2.0, out,
           "--dt", "0.1", "--stride", "50", "--nseg", "3",
           "--toda", "--lyapunov", "--lyap-renorm-steps", "10", "--lyap-seed", "77")
    meta = get_metadata(str(out))
    assert meta["Integrator"] == "Yoshida4"
    assert meta["TodaIntegral"] == "1"
    assert meta["Lyapunov"] == "1"
    assert meta["LyapRenormSteps"] == "10"
    assert meta["LyapSeed"] == "77"
    assert "SolverGitCommit" in meta
    assert meta["SolverGitDirty"] in ("0", "1")


def test_analysis_beta_pipeline_still_accepts_new_csv(solver, tmp_path):
    """A beta run with diagnostics enabled is still a valid beta trajectory for
    the (frozen) analysis pipeline's validator (name-based column selection)."""
    sys.path.insert(0, str(REPO_ROOT))
    from analysis.validation import validate_file

    out = tmp_path / "beta_diag.csv"
    solver(16, "beta", 1.0, 3.0, out,
           "--dt", "0.1", "--stride", "50", "--nseg", "5", "--entropy", "--toda")
    cand, rej = validate_file(out, min_saved_time=0.0)
    assert rej is None, f"unexpected rejection: {rej.reason if rej else ''}"
    assert cand.metadata.model == "beta"
