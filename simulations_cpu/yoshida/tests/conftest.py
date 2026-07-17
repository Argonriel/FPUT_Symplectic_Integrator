"""Shared pytest fixtures: build the C++ binaries once and expose thin wrappers.

The solver's physics header is compiled into two binaries in a temporary build
directory (session-scoped): the production ``fput_yoshida`` and the ``fput_selftest``
driver. Tests then drive the *exact production kernels* through the self-test
binary, or run short trajectories through the solver. No fixture CSVs are checked
in (``*.csv`` is gitignored globally).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

YOSHIDA_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = YOSHIDA_DIR.parent.parent


def _cxx() -> str:
    for cand in ("g++", "c++", "clang++"):
        if shutil.which(cand):
            return cand
    pytest.skip("no C++ compiler found (g++/c++/clang++)")


@pytest.fixture(scope="session")
def binaries(tmp_path_factory):
    """Compile fput_yoshida and fput_selftest once; return their paths."""
    build = tmp_path_factory.mktemp("fput_build")
    cxx = _cxx()
    solver = build / "fput_yoshida"
    selftest = build / "fput_selftest"
    subprocess.run(
        [cxx, "-O3", "-std=c++17", "-o", str(solver),
         str(YOSHIDA_DIR / "FPUT_yoshida_solver.cpp")],
        check=True, cwd=YOSHIDA_DIR,
    )
    subprocess.run(
        [cxx, "-O2", "-std=c++17", "-o", str(selftest),
         str(YOSHIDA_DIR / "fput_selftest.cpp")],
        check=True, cwd=YOSHIDA_DIR,
    )
    return {"solver": solver, "selftest": selftest}


@pytest.fixture(scope="session")
def selftest(binaries):
    """Return a callable ``run(cmd, N, *args, arrays=[...]) -> np.ndarray``."""
    exe = binaries["selftest"]

    def run(cmd, N, *args, arrays=()):
        inp = " ".join(repr(float(z)) for a in arrays for z in np.asarray(a).ravel())
        proc = subprocess.run(
            [str(exe), cmd, str(N), *[str(a) for a in args]],
            input=inp, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"selftest {cmd} failed: {proc.stderr}")
        return np.array([float(t) for t in proc.stdout.split()])

    return run


@pytest.fixture(scope="session")
def solver(binaries):
    """Return a callable that runs the solver and returns the parsed DataFrame."""
    import pandas as pd
    exe = binaries["solver"]

    def run(N, model, value, amplitude, out_path, *flags):
        cmd = [str(exe), str(N), model, str(value), str(amplitude), str(out_path), *flags]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"solver failed (rc={proc.returncode}): {proc.stderr}")
        df = pd.read_csv(out_path, comment="#")
        df.columns = df.columns.str.strip()
        return df

    return run
