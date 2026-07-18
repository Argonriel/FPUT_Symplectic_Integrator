"""Resume runner: continue an alpha-pilot checkpoint from nominal 1e7 to 1e8.

Frozen solver diag-v1 (commit 4a66fec). This is a *runner only* — it never
modifies the solver, manifest, or raw data. It resumes an existing final pilot
checkpoint into a NEW continuation CSV + NEW continuation checkpoint (the pilot
files are never overwritten), with live progress reporting and resume-integrity
assertions.

Resume/nseg convention (verified against the solver source):
  * The continuation START segment comes from the checkpoint's stored ``next_seg``
    (the pilot's final checkpoint has next_seg = 500). It is NOT taken from the CLI.
  * ``--nseg`` on a resume is TOTAL-FROM-ORIGIN (absolute snapshot count from t=0),
    and must exceed next_seg. To reach nominal duration T:
        nseg = round(T / (stride * dt))
    For T = 1e8 with stride=200000, dt=0.1  ->  nseg = 5000.
  * The solver writes a NEW continuation CSV (the positional SavePath) holding
    snapshots [next_seg .. nseg-1]; it does NOT append to the pilot CSV. The full
    trajectory is pilot rows [0..next_seg-1] ++ continuation rows [next_seg..nseg-1].
  * Seam: pilot last saved time = (next_seg-1)*stride*dt = 9.98e6; continuation
    first saved time = next_seg*stride*dt = 1.0e7 = 9.98e6 + stride*dt. No dup/skip.

Checkpoint portability: the format is field-by-field fixed-width (int32/int64 +
IEEE-754 doubles), no struct padding, native byte order. x86_64 (WSL) and arm64
(macOS) are both little-endian, so it is portable between them — but portability
must still be verified per machine by the tiny-resume preflight (``--preflight``).

Paths are machine-local (no hard-coded absolutes): ``--repo-root``,
``--pilot-data-root``, ``--output-root``, ``--binary``.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import socket
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

EXPECT_COMMIT = "4a66fec"
CKPT_MAGIC = b"FPUTCKPT"

# Checkpoint scalar-field byte offsets (little-endian; see fput_checkpoint.hpp).
_OFF = {
    "version": (8, "<i"), "origin_git_dirty": (53, "<i"),
    "N": (57, "<i"), "model_flag": (61, "<i"),
    "value": (65, "<d"), "amplitude": (73, "<d"), "dt": (81, "<d"),
    "stride": (89, "<q"),
    "entropy": (97, "<i"), "toda": (101, "<i"), "toda_debug": (105, "<i"),
    "lyapunov": (109, "<i"),
    "lyap_renorm_steps": (113, "<q"), "lyap_seed": (121, "<q"),
    "next_seg": (129, "<q"), "completed_steps": (137, "<q"),
    "current_time": (145, "<d"),
}


# --------------------------------------------------------------------------- #
# Checkpoint parsing (portable, little-endian) and hashing
# --------------------------------------------------------------------------- #
def parse_checkpoint_header(path: Path) -> dict:
    """Parse the scalar header of a checkpoint (no trajectory arrays).

    Raises ValueError on bad magic / short file. Assumes little-endian, valid on
    both x86_64 and arm64 (the solver writes/reads in native byte order and both
    target platforms are LE).
    """
    if sys.byteorder != "little":
        raise ValueError(f"host byte order is {sys.byteorder}; checkpoint format is little-endian")
    with open(path, "rb") as f:
        b = f.read(160)
    if len(b) < 153 or b[0:8] != CKPT_MAGIC:
        raise ValueError(f"{path}: not a valid checkpoint (magic mismatch or truncated)")
    out = {"magic": b[0:8].decode(errors="replace"),
           "origin_git_commit": b[12:53].split(b"\x00")[0].decode(errors="replace")}
    for name, (off, fmt) in _OFF.items():
        out[name] = struct.unpack_from(fmt, b, off)[0]
    out["model"] = "alpha" if out["model_flag"] == 0 else "beta"
    return out


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def platform_record(binary: Path) -> dict:
    u = os.uname()
    return {
        "hostname": socket.gethostname(),
        "uname_s": u.sysname, "uname_m": u.machine, "uname_r": u.release,
        "python": sys.version.split()[0], "byteorder": sys.byteorder,
        "sizeof_double": struct.calcsize("d"),
        "sizeof_int32": struct.calcsize("<i"), "sizeof_int64": struct.calcsize("<q"),
        "binary": str(binary), "binary_sha256": sha256_file(binary) if binary.exists() else None,
    }


# --------------------------------------------------------------------------- #
# Derived timing
# --------------------------------------------------------------------------- #
@dataclass
class ResumePlan:
    checkpoint: Path
    N: int
    model: str
    value: float
    amplitude: float
    dt: float
    stride: int
    next_seg: int
    nseg_total: int          # total-from-origin to reach the target nominal duration
    origin_commit: str
    entropy: bool
    toda: bool
    lyapunov: bool
    lyap_renorm_steps: int
    lyap_seed: int
    eps_label: float         # nominal epsilon from the mode-1 IC (display only)
    out_csv: Path
    out_ckpt: Path
    checkpoint_every: int

    @property
    def snapshot_spacing(self) -> float:
        return self.stride * self.dt

    @property
    def pilot_last_saved(self) -> float:
        return (self.next_seg - 1) * self.snapshot_spacing

    @property
    def continuation_first_saved(self) -> float:
        return self.next_seg * self.snapshot_spacing

    @property
    def nominal_duration(self) -> float:
        return self.nseg_total * self.snapshot_spacing

    @property
    def continuation_last_saved(self) -> float:
        return (self.nseg_total - 1) * self.snapshot_spacing

    @property
    def new_segments(self) -> int:
        return self.nseg_total - self.next_seg


def _nominal_epsilon_alpha(A: float, N: int) -> float:
    """Nominal epsilon of the alpha mode-1 IC (harmonic; label only)."""
    z = A * A * math.sin(math.pi / (2.0 * N)) ** 2
    return N * z / (N - 1)


def build_plan(checkpoint: Path, output_root: Path, nominal_duration: float,
               checkpoint_every: int = 50, tag: str = "cont") -> ResumePlan:
    hdr = parse_checkpoint_header(checkpoint)
    dt = hdr["dt"]; stride = hdr["stride"]; N = hdr["N"]
    spacing = stride * dt
    nseg_total = int(round(nominal_duration / spacing))
    if nseg_total <= hdr["next_seg"]:
        raise ValueError(
            f"target nominal {nominal_duration:g} -> nseg {nseg_total} does not exceed "
            f"checkpoint next_seg {hdr['next_seg']}")
    eps = _nominal_epsilon_alpha(hdr["amplitude"], N) if hdr["model"] == "alpha" else float("nan")
    stem = (f"{hdr['model']}_N{N}_eps{eps:.1e}_A{hdr['amplitude']:.6f}"
            f"_{tag}_nom{nominal_duration:.0e}")
    return ResumePlan(
        checkpoint=checkpoint, N=N, model=hdr["model"], value=hdr["value"],
        amplitude=hdr["amplitude"], dt=dt, stride=stride, next_seg=hdr["next_seg"],
        nseg_total=nseg_total, origin_commit=hdr["origin_git_commit"],
        entropy=bool(hdr["entropy"]), toda=bool(hdr["toda"]), lyapunov=bool(hdr["lyapunov"]),
        lyap_renorm_steps=hdr["lyap_renorm_steps"], lyap_seed=hdr["lyap_seed"],
        eps_label=eps, out_csv=output_root / f"{stem}.csv",
        out_ckpt=output_root / f"{stem}.ckpt", checkpoint_every=checkpoint_every)


def build_command(binary: Path, plan: ResumePlan) -> list[str]:
    """Reconstruct the resume command from the checkpoint's own parameters so the
    solver's dynamically-relevant-parameter check can never mismatch."""
    # Use repr() (shortest round-tripping form) for the doubles so stod() on the
    # solver side reproduces the exact bits stored in the checkpoint -> the
    # solver's exact-match resume check on value/amplitude/dt cannot spuriously fail.
    cmd = [str(binary), str(plan.N), plan.model, repr(plan.value), repr(plan.amplitude),
           str(plan.out_csv), "--dt", repr(plan.dt), "--stride", str(plan.stride),
           "--nseg", str(plan.nseg_total)]
    if plan.entropy:
        cmd.append("--entropy")
    if plan.toda:
        cmd.append("--toda")
    if plan.lyapunov:
        cmd += ["--lyapunov", "--lyap-renorm-steps", str(plan.lyap_renorm_steps),
                "--lyap-seed", str(plan.lyap_seed)]
    cmd += ["--resume", str(plan.checkpoint),
            "--checkpoint-file", str(plan.out_ckpt),
            "--checkpoint-every", str(plan.checkpoint_every)]
    return cmd


# --------------------------------------------------------------------------- #
# Live progress
# --------------------------------------------------------------------------- #
def _fmt_dur(sec: float) -> str:
    sec = int(max(0, sec))
    h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def format_progress_line(machine: str, eps: float, N: int, done: int, total: int,
                         elapsed: float, eta: float) -> str:
    """Pure, testable formatter (done/total are from-origin segment counts)."""
    pct = 100.0 * done / total if total else 0.0
    eps_s = f"{eps:.1e}" if math.isfinite(eps) else "n/a"
    return (f"[{machine} eps={eps_s} N={N}] {pct:5.1f}%  ({done}/{total} segments)  "
            f"elapsed {_fmt_dur(elapsed)}  ETA ~{_fmt_dur(eta)}")


def _count_data_rows(csv: Path) -> int:
    if not csv.exists():
        return 0
    n = 0
    with open(csv, "r") as f:
        for line in f:
            if line and not line.startswith("#") and not line.startswith("Time"):
                n += 1
    return n


def run_with_progress(binary: Path, plan: ResumePlan, log, machine: str,
                      interval_s: float = 30.0, every_segments: int | None = None) -> int:
    """Launch the resume and emit progress lines derived from continuation rows.

    Progress polling only reads the continuation CSV row count; it never touches
    the solver process or the trajectory.
    """
    cmd = build_command(binary, plan)
    plan.out_csv.parent.mkdir(parents=True, exist_ok=True)
    solver_log = plan.out_csv.with_suffix(".solver.log")
    t0 = time.time()
    _log(log, f"RESUME START cmd: {' '.join(cmd)}")
    with open(solver_log, "w") as sl:
        proc = subprocess.Popen(cmd, stdout=sl, stderr=subprocess.STDOUT)
        last_emit = 0.0
        last_seg_emit = plan.next_seg
        while proc.poll() is None:
            time.sleep(1.0)
            rows = _count_data_rows(plan.out_csv)
            done = plan.next_seg + rows
            now = time.time(); elapsed = now - t0
            due_time = (now - last_emit) >= interval_s
            due_seg = (every_segments is not None and (done - last_seg_emit) >= every_segments)
            if due_time or due_seg:
                per_seg = elapsed / max(rows, 1)
                eta = per_seg * max(0, plan.new_segments - rows)
                line = format_progress_line(machine, plan.eps_label, plan.N, done,
                                            plan.nseg_total, elapsed, eta)
                print(line, flush=True)
                _log(log, line)
                last_emit = now; last_seg_emit = done
        rc = proc.returncode
    _log(log, f"RESUME END exit={rc} wall={time.time()-t0:.1f}s")
    return rc


# --------------------------------------------------------------------------- #
# Resume-integrity validation
# --------------------------------------------------------------------------- #
def validate_continuation(plan: ResumePlan, pilot_csv: Path | None,
                          expect_commit: str | None) -> tuple[bool, list[str]]:
    """Section-5 assertions. Returns (ok, messages)."""
    import numpy as np
    import pandas as pd
    msgs = []; ok = True

    def check(cond, msg):
        nonlocal ok
        msgs.append(("PASS " if cond else "FAIL ") + msg); ok = ok and cond

    # metadata header of the continuation CSV
    meta = {}
    for line in open(plan.out_csv):
        if not line.startswith("#"):
            break
        c = line.lstrip("#").strip()
        if ":" in c:
            k, v = c.split(":", 1); meta[k.strip()] = v.strip()

    df = pd.read_csv(plan.out_csv, comment="#"); df.columns = df.columns.str.strip()
    t = df["Time"].to_numpy()

    # (a) seam: first new saved time == next expected after pilot last (no dup/skip)
    expected_first = plan.continuation_first_saved
    check(abs(t[0] - expected_first) < 1e-3,
          f"seam: continuation first t={t[0]:.6g} == expected {expected_first:.6g} "
          f"(= pilot last {plan.pilot_last_saved:.6g} + spacing {plan.snapshot_spacing:.6g})")
    check(abs(t[-1] - plan.continuation_last_saved) < 1.0,
          f"continuation last t={t[-1]:.6g} == {plan.continuation_last_saved:.6g}")
    check(len(df) == plan.new_segments,
          f"continuation rows={len(df)} == new segments {plan.new_segments}")

    # (b) provenance
    if expect_commit is not None:
        check(meta.get("SolverGitCommit") == expect_commit,
              f"SolverGitCommit={meta.get('SolverGitCommit')} == {expect_commit}")
    check(meta.get("SolverGitDirty") == "0", f"SolverGitDirty={meta.get('SolverGitDirty')} == 0")
    check(meta.get("CheckpointOriginCommit") == plan.origin_commit,
          f"CheckpointOriginCommit={meta.get('CheckpointOriginCommit')} == {plan.origin_commit}")

    # (c) epsilon_actual matches the pilot point (energy is conserved across resume)
    reqcols = ["TotalEnergy", "TodaJ", "Eta", "LyapunovFTLE", "LyapunovLocal"]
    if pilot_csv is not None and Path(pilot_csv).exists():
        pil = pd.read_csv(pilot_csv, comment="#"); pil.columns = pil.columns.str.strip()
        E0 = float(pil["TotalEnergy"].iloc[0]); eps_pilot = E0 / (plan.N - 1)
        Ec = float(df["TotalEnergy"].iloc[0]); eps_cont = Ec / (plan.N - 1)
        rel = abs(eps_cont - eps_pilot) / eps_pilot
        check(rel < 1e-3, f"epsilon_actual matches pilot: cont {eps_cont:.6e} vs pilot {eps_pilot:.6e} (rel {rel:.2e})")

    # (d) finiteness
    finite = all(np.all(np.isfinite(df[c].to_numpy())) for c in reqcols if c in df.columns)
    check(finite, "no NaN/inf in required columns")

    # (e) energy drift finite & bounded
    E = df["TotalEnergy"].to_numpy()
    drift = float(np.max(np.abs(E - E[0]) / abs(E[0])))
    check(math.isfinite(drift) and drift < 1e-3, f"energy drift finite & bounded ({drift:.2e})")

    # (f) continuation checkpoint exists and parses
    check(plan.out_ckpt.exists(), f"continuation checkpoint exists: {plan.out_ckpt.name}")
    if plan.out_ckpt.exists():
        try:
            h = parse_checkpoint_header(plan.out_ckpt)
            check(h["N"] == plan.N and h["next_seg"] == plan.nseg_total,
                  f"continuation checkpoint parses (N={h['N']}, next_seg={h['next_seg']})")
        except ValueError as e:
            check(False, f"continuation checkpoint parse failed: {e}")
    return ok, msgs


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(log, msg: str) -> None:
    if log is not None:
        log.write(f"[{_ts()}] {msg}\n"); log.flush()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m analysis.extend_pilot",
                                 description="Resume an alpha-pilot checkpoint to a larger nominal duration.")
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--pilot-data-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--checkpoint", required=True,
                    help="Path to the pilot FINAL checkpoint to resume (machine-local).")
    ap.add_argument("--pilot-csv", default=None,
                    help="Optional pilot CSV for the epsilon_actual cross-check.")
    ap.add_argument("--nominal-duration", type=float, default=1e8)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--progress-interval", type=float, default=30.0)
    ap.add_argument("--progress-every-segments", type=int, default=None)
    ap.add_argument("--machine", default=None)
    ap.add_argument("--expect-commit", default=EXPECT_COMMIT,
                    help="Expected SolverGitCommit in continuation output (default 4a66fec).")
    ap.add_argument("--preflight", type=int, default=None, metavar="K",
                    help="Portability preflight: tiny resume of +K segments into output-root, "
                         "validate, and exit (do NOT run the long continuation).")
    ap.add_argument("--dry-run", action="store_true", help="Print the resume command and exit.")
    args = ap.parse_args(argv)

    binary = Path(args.binary); checkpoint = Path(args.checkpoint)
    output_root = Path(args.output_root)
    machine = args.machine or socket.gethostname()

    if not binary.exists():
        print(f"ERROR: binary not found: {binary}", file=sys.stderr); return 2
    if not checkpoint.exists():
        print(f"ERROR: checkpoint not found: {checkpoint}", file=sys.stderr); return 2

    # Preflight: tiny resume of +K segments (Section 8 per-machine portability check).
    if args.preflight is not None:
        hdr = parse_checkpoint_header(checkpoint)
        tiny_nominal = (hdr["next_seg"] + args.preflight) * hdr["stride"] * hdr["dt"]
        plan = build_plan(checkpoint, output_root, tiny_nominal,
                          checkpoint_every=max(1, args.preflight), tag="preflight")
    else:
        plan = build_plan(checkpoint, output_root, args.nominal_duration,
                          checkpoint_every=args.checkpoint_every)

    if args.dry_run:
        print(" ".join(build_command(binary, plan)))
        print(f"# start_seg={plan.next_seg} nseg_total={plan.nseg_total} "
              f"nominal={plan.nominal_duration:.3e} last_saved={plan.continuation_last_saved:.3e} "
              f"new_segments={plan.new_segments}")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / (plan.out_csv.stem + ".runner.log")
    with open(log_path, "w") as log:
        rec = platform_record(binary)
        rec["checkpoint"] = str(checkpoint)
        rec["checkpoint_sha256"] = sha256_file(checkpoint)
        rec["resume_start_time_physical"] = plan.continuation_first_saved
        rec["start_ts"] = _ts()
        _log(log, "PLATFORM " + str(rec))
        _log(log, "PLAN " + str(asdict(plan) | {"checkpoint": str(plan.checkpoint),
             "out_csv": str(plan.out_csv), "out_ckpt": str(plan.out_ckpt)}))
        print(f"[{machine}] resuming {checkpoint.name}: seg {plan.next_seg} -> {plan.nseg_total} "
              f"(nominal {plan.nominal_duration:.2e}); output {plan.out_csv.name}", flush=True)

        rc = run_with_progress(binary, plan, log, machine,
                               interval_s=args.progress_interval,
                               every_segments=args.progress_every_segments)
        _log(log, f"end_ts {_ts()} exit {rc}")
        if rc != 0:
            print(f"[{machine}] SOLVER EXIT {rc} — see {plan.out_csv.with_suffix('.solver.log')}",
                  file=sys.stderr)
            return rc
        ok, msgs = validate_continuation(plan, Path(args.pilot_csv) if args.pilot_csv else None,
                                         args.expect_commit if args.preflight is None else plan.origin_commit)
        for m in msgs:
            print("  " + m); _log(log, m)
        print(f"[{machine}] resume-integrity: {'PASS' if ok else 'FAIL'}")
        _log(log, f"resume-integrity {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
