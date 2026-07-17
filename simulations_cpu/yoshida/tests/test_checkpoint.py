"""Checkpoint/restart: split-run equivalence, extension, and rejection cases.

The binary checkpoint stores all doubles exactly, so an uninterrupted run and a
checkpoint/resume split must produce BITWISE-identical data rows (including
TodaJ and LyapunovFTLE).
"""

from __future__ import annotations

import subprocess

import numpy as np
import pandas as pd


def _read(path):
    df = pd.read_csv(path, comment="#")
    df.columns = df.columns.str.strip()
    return df


def _bitwise_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return list(a.columns) == list(b.columns) and bool((a.to_numpy() == b.to_numpy()).all())


DIAG = ["--entropy", "--toda", "--lyapunov", "--lyap-renorm-steps", "20", "--lyap-seed", "99"]


def test_restart_equivalence_full_diagnostics(solver, tmp_path):
    """A(nseg=6) == B1(nseg=3, checkpoint) + resume(nseg=6). Bitwise."""
    A = solver(32, "alpha", 1.0, 2.0, tmp_path / "A.csv",
               "--dt", "0.1", "--stride", "500", "--nseg", "6", *DIAG)
    solver(32, "alpha", 1.0, 2.0, tmp_path / "B1.csv",
           "--dt", "0.1", "--stride", "500", "--nseg", "3", *DIAG,
           "--checkpoint-file", str(tmp_path / "cp.bin"))
    B2 = solver(32, "alpha", 1.0, 2.0, tmp_path / "B2.csv",
                "--dt", "0.1", "--stride", "500", "--nseg", "6", *DIAG,
                "--resume", str(tmp_path / "cp.bin"))
    B1 = _read(tmp_path / "B1.csv")
    concat = pd.concat([B1, B2], ignore_index=True)
    assert _bitwise_equal(A, concat)
    # Explicitly assert the diagnostic columns match bitwise.
    for c in ("TodaJ", "LyapunovFTLE", "LyapunovLocal", "TotalEnergy"):
        assert np.array_equal(A[c].to_numpy(), concat[c].to_numpy())


def test_restart_equivalence_diagnostics_off(solver, tmp_path):
    """Physical-only checkpoint/resume is also bitwise-identical."""
    A = solver(24, "beta", 1.0, 3.0, tmp_path / "A.csv",
               "--dt", "0.1", "--stride", "400", "--nseg", "5")
    solver(24, "beta", 1.0, 3.0, tmp_path / "B1.csv",
           "--dt", "0.1", "--stride", "400", "--nseg", "2",
           "--checkpoint-file", str(tmp_path / "cp.bin"))
    B2 = solver(24, "beta", 1.0, 3.0, tmp_path / "B2.csv",
                "--dt", "0.1", "--stride", "400", "--nseg", "5",
                "--resume", str(tmp_path / "cp.bin"))
    concat = pd.concat([_read(tmp_path / "B1.csv"), B2], ignore_index=True)
    assert _bitwise_equal(A, concat)


def test_extend_beyond_original_length(solver, tmp_path):
    """The production use case: a short run is later EXTENDED to a longer nseg."""
    C = solver(32, "alpha", 1.0, 2.0, tmp_path / "C.csv",
               "--dt", "0.1", "--stride", "500", "--nseg", "8", *DIAG)
    solver(32, "alpha", 1.0, 2.0, tmp_path / "B1.csv",
           "--dt", "0.1", "--stride", "500", "--nseg", "3", *DIAG,
           "--checkpoint-file", str(tmp_path / "cp.bin"))
    B2 = solver(32, "alpha", 1.0, 2.0, tmp_path / "B2.csv",
                "--dt", "0.1", "--stride", "500", "--nseg", "8", *DIAG,
                "--resume", str(tmp_path / "cp.bin"))
    concat = pd.concat([_read(tmp_path / "B1.csv"), B2], ignore_index=True)
    assert _bitwise_equal(C, concat)


def test_periodic_checkpoint_resume(solver, tmp_path):
    """A mid-run periodic checkpoint (--checkpoint-every) also resumes correctly.

    Run B1 to nseg=6 with checkpoint-every=2 (the file ends holding the final
    next_seg=6 checkpoint after overwriting the periodic ones); resuming to a
    larger nseg reproduces the uninterrupted longer run.
    """
    C = solver(32, "alpha", 1.0, 2.0, tmp_path / "C.csv",
               "--dt", "0.1", "--stride", "300", "--nseg", "9", *DIAG)
    solver(32, "alpha", 1.0, 2.0, tmp_path / "B1.csv",
           "--dt", "0.1", "--stride", "300", "--nseg", "6", *DIAG,
           "--checkpoint-file", str(tmp_path / "cp.bin"), "--checkpoint-every", "2")
    B2 = solver(32, "alpha", 1.0, 2.0, tmp_path / "B2.csv",
                "--dt", "0.1", "--stride", "300", "--nseg", "9", *DIAG,
                "--resume", str(tmp_path / "cp.bin"))
    concat = pd.concat([_read(tmp_path / "B1.csv"), B2], ignore_index=True)
    assert _bitwise_equal(C, concat)


def test_checkpoint_is_atomic_no_tmp_left(solver, tmp_path):
    solver(16, "alpha", 1.0, 1.0, tmp_path / "o.csv",
           "--dt", "0.1", "--stride", "100", "--nseg", "4",
           "--checkpoint-file", str(tmp_path / "cp.bin"))
    assert (tmp_path / "cp.bin").exists()
    assert not (tmp_path / "cp.bin.tmp").exists()


def _raw(binaries, *args):
    return subprocess.run([str(binaries["solver"]), *[str(a) for a in args]],
                          capture_output=True, text=True)


def test_resume_rejects_parameter_mismatch(binaries, tmp_path):
    cp = tmp_path / "cp.bin"
    _raw(binaries, 16, "alpha", 1.0, 1.0, tmp_path / "a.csv",
         "--dt", "0.1", "--stride", "100", "--nseg", "3", "--toda",
         "--checkpoint-file", cp)
    # Wrong coefficient value on resume -> reject.
    r = _raw(binaries, 16, "alpha", 2.0, 1.0, tmp_path / "b.csv",
             "--dt", "0.1", "--stride", "100", "--nseg", "6", "--toda",
             "--resume", cp)
    assert r.returncode != 0
    assert "mismatch" in (r.stderr + r.stdout).lower()

    # Wrong N on resume -> reject.
    r2 = _raw(binaries, 32, "alpha", 1.0, 1.0, tmp_path / "c.csv",
              "--dt", "0.1", "--stride", "100", "--nseg", "6", "--toda",
              "--resume", cp)
    assert r2.returncode != 0

    # Mismatched diagnostics set (added --lyapunov) -> reject.
    r3 = _raw(binaries, 16, "alpha", 1.0, 1.0, tmp_path / "d.csv",
              "--dt", "0.1", "--stride", "100", "--nseg", "6", "--toda", "--lyapunov",
              "--resume", cp)
    assert r3.returncode != 0


def test_resume_requires_larger_nseg(binaries, tmp_path):
    cp = tmp_path / "cp.bin"
    _raw(binaries, 16, "alpha", 1.0, 1.0, tmp_path / "a.csv",
         "--dt", "0.1", "--stride", "100", "--nseg", "4",
         "--checkpoint-file", cp)  # final checkpoint next_seg=4
    r = _raw(binaries, 16, "alpha", 1.0, 1.0, tmp_path / "b.csv",
             "--dt", "0.1", "--stride", "100", "--nseg", "4", "--resume", cp)
    assert r.returncode != 0  # nseg (4) not > checkpoint seg (4)


def test_resume_from_corrupt_checkpoint_fails_cleanly(binaries, tmp_path):
    cp = tmp_path / "bad.bin"
    cp.write_bytes(b"not a checkpoint")
    r = _raw(binaries, 16, "alpha", 1.0, 1.0, tmp_path / "b.csv",
             "--dt", "0.1", "--stride", "100", "--nseg", "4", "--resume", cp)
    assert r.returncode != 0
    assert "checkpoint" in (r.stderr + r.stdout).lower()
