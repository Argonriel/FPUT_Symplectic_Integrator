"""Self-test for the resume runner analysis/extend_pilot.py.

Uses the FROZEN local binary (simulations_cpu/yoshida/fput_yoshida, built from
4a66fec with SolverGitDirty:0) so provenance assertions are meaningful. Skips if
that binary is absent or dirty. Runtime is seconds (tiny N, +5 segments); no long
simulation.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
FROZEN_BIN = REPO / "simulations_cpu/yoshida/fput_yoshida"
sys.path.insert(0, str(REPO))
from analysis.extend_pilot import (  # noqa: E402
    build_command, build_plan, format_progress_line, parse_checkpoint_header,
    validate_continuation,
)


def _frozen_binary_ok() -> bool:
    if not FROZEN_BIN.exists():
        return False
    out = REPO / "scratch/_extend_selftest_probe.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([str(FROZEN_BIN), "8", "alpha", "1.0", "0.3", str(out),
                        "--nseg", "2", "--stride", "5", "--toda"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return False
    meta = dict(l.lstrip("#").split(":", 1) for l in out.read_text().splitlines()
                if l.startswith("#") and ":" in l)
    return meta.get(" SolverGitDirty", meta.get("SolverGitDirty", "")).strip() == "0"


frozen = pytest.mark.skipif(not _frozen_binary_ok(),
                            reason="frozen 4a66fec/Dirty0 binary not available")


# --- pure functions --------------------------------------------------------
def test_format_progress_line():
    s = format_progress_line("mac", 1e-4, 512, 500, 5000, 65.0, 585.0)
    assert "[mac eps=1.0e-04 N=512]" in s
    assert "10.0%" in s and "(500/5000 segments)" in s
    assert "elapsed 1m05s" in s and "ETA ~9m45s" in s


def test_format_progress_line_nan_eps():
    s = format_progress_line("stay", float("nan"), 1024, 600, 5000, 10, 10)
    assert "eps=n/a" in s


@frozen
def test_parse_checkpoint_header(tmp_path):
    cp = tmp_path / "c.ckpt"
    subprocess.run([str(FROZEN_BIN), "32", "alpha", "1.0", "2.0", str(tmp_path / "a.csv"),
                    "--dt", "0.1", "--stride", "50", "--nseg", "3",
                    "--entropy", "--toda", "--lyapunov", "--checkpoint-file", str(cp)],
                   check=True, capture_output=True)
    h = parse_checkpoint_header(cp)
    assert h["magic"] == "FPUTCKPT" and h["version"] == 1
    assert h["N"] == 32 and h["model"] == "alpha" and h["amplitude"] == pytest.approx(2.0)
    assert h["dt"] == pytest.approx(0.1) and h["stride"] == 50
    assert h["entropy"] == 1 and h["toda"] == 1 and h["lyapunov"] == 1
    assert h["next_seg"] == 3  # final checkpoint after nseg=3
    assert h["origin_git_commit"] == "4a66fec" and h["origin_git_dirty"] == 0


@frozen
def test_tiny_resume_seam_provenance_and_progress(tmp_path):
    """Create a tiny checkpoint (nseg=3), resume +5 (nseg=8), and assert a
    continuous seam, preserved provenance, finite output, and that a progress line
    prints. Also check bitwise equivalence vs an uninterrupted nseg=8 run."""
    stride, dt = 50, 0.1
    flags = ["--dt", str(dt), "--stride", str(stride), "--entropy", "--toda",
             "--lyapunov", "--lyap-renorm-steps", "10", "--lyap-seed", "12345"]

    # base pilot-like run to nseg=3 with a checkpoint
    base_csv = tmp_path / "base.csv"; cp = tmp_path / "base.ckpt"
    subprocess.run([str(FROZEN_BIN), "48", "alpha", "1.0", "3.0", str(base_csv),
                    "--nseg", "3", *flags, "--checkpoint-file", str(cp)],
                   check=True, capture_output=True)

    # uninterrupted reference to nseg=8
    ref_csv = tmp_path / "ref.csv"
    subprocess.run([str(FROZEN_BIN), "48", "alpha", "1.0", "3.0", str(ref_csv),
                    "--nseg", "8", *flags], check=True, capture_output=True)

    # resume +5 (nseg=8) via the runner's plan/command
    out_root = tmp_path / "out"
    plan = build_plan(cp, out_root, nominal_duration=8 * stride * dt,
                      checkpoint_every=2, tag="selftest")
    assert plan.next_seg == 3 and plan.nseg_total == 8 and plan.new_segments == 5
    out_root.mkdir(parents=True, exist_ok=True)
    cmd = build_command(FROZEN_BIN, plan)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    # seam: continuation first time == base last time + spacing (no dup/skip)
    base = pd.read_csv(base_csv, comment="#"); base.columns = base.columns.str.strip()
    cont = pd.read_csv(plan.out_csv, comment="#"); cont.columns = cont.columns.str.strip()
    ref = pd.read_csv(ref_csv, comment="#"); ref.columns = ref.columns.str.strip()
    spacing = stride * dt
    assert cont["Time"].iloc[0] == pytest.approx(base["Time"].iloc[-1] + spacing)
    assert len(base) == 3 and len(cont) == 5

    # concatenation reproduces the uninterrupted run bitwise (same binary/arch)
    full = pd.concat([base, cont], ignore_index=True)
    assert list(full.columns) == list(ref.columns)
    assert np.array_equal(full.to_numpy(), ref.to_numpy())

    # provenance preserved + finite + checkpoint parses (Section 5)
    ok, msgs = validate_continuation(plan, pilot_csv=base_csv, expect_commit="4a66fec")
    assert ok, "\n".join(msgs)

    # progress line formatting on real counts
    line = format_progress_line("selftest", plan.eps_label, plan.N,
                                plan.next_seg + len(cont), plan.nseg_total, 1.0, 0.0)
    assert "(8/8 segments)" in line and "100.0%" in line


@frozen
def test_resume_rejects_wrong_nseg(tmp_path):
    """nseg must exceed the checkpoint segment (total-from-origin convention)."""
    cp = tmp_path / "c.ckpt"
    subprocess.run([str(FROZEN_BIN), "32", "alpha", "1.0", "2.0", str(tmp_path / "a.csv"),
                    "--dt", "0.1", "--stride", "50", "--nseg", "5", "--toda",
                    "--checkpoint-file", str(cp)], check=True, capture_output=True)
    with pytest.raises(ValueError, match="does not exceed"):
        build_plan(cp, tmp_path, nominal_duration=5 * 50 * 0.1)  # nseg=5 == next_seg
