"""Analysis for the FPUT-alpha=1 master epsilon-N grid (v1).

Read-only companion to ``analysis/pilot_diagnostics.py`` and
``analysis/alpha_ext_v2.py``. It NEVER runs or resumes a simulation and NEVER
edits any raw CSV, checkpoint, frozen manifest, log, or solver source. It
consumes the frozen trajectories under ``data/`` and writes every figure
(PNG+PDF), CSV, JSON, and Markdown into a dedicated gitignored directory
(default ``figures/alpha_grid_v1/``). It does not overwrite any earlier
``figures/`` directory.

Scope (see ``docs/MANUSCRIPT_WORDING_TODO.md``, strictly followed here):

  Master grid   N in {512,1024,2048}, eps_target in
                {6e-5,1e-4,2e-4,4e-4,6e-4,8e-4}, each trajectory to nominal 1e8.
  Source rule   For every (N, eps_target) cell, gather ALL candidates and apply
                a frozen precedence:
                  1. complete+valid pilot+continuation splice to nominal 1e8;
                  2. else complete+valid from-t=0 gap run;
                  3. if both exist, the second is kept as an independent
                     reproducibility check, never silently discarded.
                Exactly one CANONICAL source is required per cell after the
                rule (not exactly one raw candidate). Cells whose canonical
                source cannot be built safely are aborted individually and
                reported; independent valid cells continue.

  Task A   master eps-N summary: tail-mean J/(2 eps_act), cumulative FTLE, and
           Eta vs log eps_actual, one curve per N, with temporal tail std.
  Task B   persistent Toda timescales T10/T90 using a TRAILING 2e6 moving mean
           of Phi_J; right-censored (>1e8) when not reached.
  Task C   low-eps deep-time block growth rates of S(t)=t*FTLE(t).
  Task D   dt and Lyapunov-cadence robustness at eps=1e-4, N=512.

Frozen conventions (restated in the generated report):
  * epsilon_actual = TotalEnergy[0]/(N-1). For a from-t=0 run it is the run's
    OWN t=0 row. For a pilot+continuation splice it is the PILOT's t=0 row only;
    a continuation's energy is used only to validate consistency across the seam,
    never to (re)define eps_actual. Independent runs are never cross-normalized.
  * J = 2*eps is the alpha *equilibrium estimate* (not a theorem, not a beta
    result). The mode-1 IC gives J(0) ~ 3*eps, so J approaches 2*eps from above
    and Phi_J = (J-J0)/(2 eps - J0) rises 0 -> 1. Phi_J is NOT clipped to [0,1].
  * Seams are validated in PHYSICAL time: expected_next = prev_last + stride*dt.
  * Fixed physical tail window [8e7, 1e8); reported standard deviations are
    TEMPORAL variation over the window, NOT statistical uncertainty across
    realizations. No power-law fit (Td~eps^-gamma, Teq, critical eps) is made.
  * Toda drift is judged by block means / smoothed trends, never by pointwise
    monotonicity of raw J. eps=1e-4 is not called a stable positive Lyapunov
    plateau unless the block-rate AND LyapunovLocal criteria support it.
  * Any reference/guide line is a visual guide, not a fit.
  * matplotlib only; no seaborn; global styles are not modified. A series is
    encoded by color AND line/marker style (never color alone).

Usage:
    python -m analysis.alpha_grid_v1 --output-dir figures/alpha_grid_v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend (not a style change)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_VIS = os.path.join(os.path.dirname(__file__), "..", "visualization")
if _VIS not in sys.path:
    sys.path.insert(0, _VIS)
from plot_utils import get_metadata  # noqa: E402

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
REQUIRED_COLUMNS = ["Time", "TotalEnergy", "TodaJ", "Eta", "LyapunovFTLE", "LyapunovLocal"]
EXPECT_COMMIT = "4a66fec"

# Deterministic ordering (frozen).
N_ORDER = [512, 1024, 2048]
EPS_ORDER = [6e-5, 1e-4, 2e-4, 4e-4, 6e-4, 8e-4]

SAVE_INTERVAL = 2.0e4          # stride*dt for every trajectory here (0.1*2e5 == 0.05*4e5)
TAIL_LO, TAIL_HI = 8.0e7, 1.0e8    # fixed physical tail window [8e7, 1e8)
SMOOTH_WINDOW = 2.0e6          # physical window for the TRAILING Phi_J moving mean
T10_LEVEL = 0.10
T90_LEVEL = 0.90
NOM_1E8 = 1.0e8

# tolerances
T_TOL = 1.0                    # physical-time tolerance (units of t) for grid/seam checks
AMP_RTOL = 1.0e-3              # relative amplitude match tolerance (headers store ~6 sig figs)
EPS_SEAM_RTOL = 1.0e-3         # epsilon_actual consistency across a seam
CELL_RTOL = 0.25              # max |eps_actual - eps_target|/eps_target to assign a grid cell
S_SEAM_RTOL = 0.15             # descriptive flag for a possible FTLE reset if |dS|/S exceeds this

# ---- FROZEN observable-specific criteria (declared BEFORE viewing any curve) ----
# These are sampled-bracket criteria, NOT fitted thresholds and NOT critical-eps
# claims. They report "no onset at eps_i; criterion satisfied at eps_{i+1}".
FROZEN_TODA_ONSET_PHIJ = 0.50      # Toda relaxation "substantially underway": tail <Phi_J> >= 0.5
FROZEN_TODA_NEAR_EQ = 1.10         # Toda "near the 2eps estimate": tail <J/2eps> <= 1.10
# Chaos "stable positive finite-time plateau" requires ALL THREE legs (wording B):
#   (i) cumulative FTLE flat: |FTLE_final - FTLE_tail_mean| <= 0.25*|FTLE_tail_mean|
#   (ii) LyapunovLocal tail mean > 0
#   (iii) local fluctuations around a positive value: LyapLocal_tail_mean > LyapLocal_tail_std
FROZEN_FTLE_FLAT_RTOL = 0.25

# Per-N visual style: (color, linestyle, marker) so N is never color-only.
STYLE_N = {
    512:  ("C0", "-", "o"),
    1024: ("C1", "--", "s"),
    2048: ("C3", "-.", "^"),
}
# Per-dt style (Task D).
STYLE_DT = {
    0.1:  ("C0", "-", "o"),
    0.05: ("C3", "--", "s"),
}

SUP = "FPUT-$\\alpha$ ($\\alpha=1$, mode-1 IC)"


class ValidationError(RuntimeError):
    """Raised when a required input fails validation (aborts the affected cell)."""


# --------------------------------------------------------------------------- #
# Segment loading + per-file validation
# --------------------------------------------------------------------------- #
@dataclass
class Segment:
    path: Path
    meta: dict
    df: pd.DataFrame
    N: int
    amplitude: float
    dt: float
    stride: int
    save_interval: float
    resume_from: int
    num_segments: int
    renorm_steps: int
    first_t: float
    last_t: float
    nrows: int
    E0: float
    checks: list = field(default_factory=list)   # (name, ok, detail)


def _num(meta: dict, key: str) -> float:
    if key not in meta:
        raise ValidationError(f"metadata missing '{key}'")
    return float(meta[key])


def load_segment(csv: Path, expect_N: int | None = None, expect_amp: float | None = None,
                 expect_dt: float | None = None) -> Segment:
    """Validate one CSV against the requested physical point; return a Segment.

    Verifies (per task spec): model==alpha; Alpha==1; N/amplitude/dt/stride and
    diagnostics flags present and consistent; required columns exist and are
    finite; Time strictly increasing; saved spacing uniform and == stride*dt;
    row count == NumSegments-ResumeFromSegment; SolverGitCommit==4a66fec;
    SolverGitDirty==0. Raises ValidationError on any hard failure.
    """
    meta = get_metadata(str(csv))
    checks: list = []

    if meta.get("Model", "").lower() != "alpha":
        raise ValidationError(f"{csv.name}: Model={meta.get('Model')!r} != alpha")
    if abs(_num(meta, "Alpha") - 1.0) > 1e-12:
        raise ValidationError(f"{csv.name}: Alpha={meta.get('Alpha')} != 1")
    N = int(_num(meta, "N"))
    if expect_N is not None and N != expect_N:
        raise ValidationError(f"{csv.name}: N={N} != expected {expect_N}")
    amp = _num(meta, "Amplitude")
    if expect_amp is not None and abs(amp - expect_amp) > AMP_RTOL * abs(expect_amp):
        raise ValidationError(
            f"{csv.name}: Amplitude={amp} not within {AMP_RTOL:.0e} of expected {expect_amp}")
    dt = _num(meta, "dt")
    if expect_dt is not None and abs(dt - expect_dt) > 1e-12:
        raise ValidationError(f"{csv.name}: dt={dt} != expected {expect_dt}")
    stride = int(_num(meta, "Stride"))
    save_interval = stride * dt
    if abs(save_interval - SAVE_INTERVAL) > 1e-6 * SAVE_INTERVAL:
        raise ValidationError(
            f"{csv.name}: stride*dt={save_interval} != expected save interval {SAVE_INTERVAL}")
    # diagnostics flags present + enabled (Entropy/TodaIntegral/Lyapunov)
    for flag in ("Entropy", "TodaIntegral", "Lyapunov"):
        if str(meta.get(flag, "0")).strip() != "1":
            raise ValidationError(f"{csv.name}: diagnostic flag {flag}={meta.get(flag)} != 1")
    renorm_steps = int(_num(meta, "LyapRenormSteps"))

    if meta.get("SolverGitCommit") != EXPECT_COMMIT:
        raise ValidationError(f"{csv.name}: SolverGitCommit={meta.get('SolverGitCommit')} != {EXPECT_COMMIT}")
    if meta.get("SolverGitDirty") != "0":
        raise ValidationError(f"{csv.name}: SolverGitDirty={meta.get('SolverGitDirty')} != 0")

    df = pd.read_csv(csv, comment="#")
    df.columns = df.columns.str.strip()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValidationError(f"{csv.name}: missing required columns {missing}")
    for c in REQUIRED_COLUMNS:
        if not np.all(np.isfinite(df[c].to_numpy())):
            raise ValidationError(f"{csv.name}: non-finite (NaN/inf) in column {c}")

    t = df["Time"].to_numpy()
    if np.any(np.diff(t) <= 0):
        raise ValidationError(f"{csv.name}: Time is not strictly increasing")
    d = np.diff(t)
    if d.size and np.max(np.abs(d - save_interval)) > T_TOL:
        raise ValidationError(
            f"{csv.name}: non-uniform saved-time spacing (max dev "
            f"{np.max(np.abs(d - save_interval)):.3e} from {save_interval})")

    resume = int(_num(meta, "ResumeFromSegment")) if "ResumeFromSegment" in meta else 0
    num_seg = int(_num(meta, "NumSegments"))
    nrows = len(df)
    if nrows != num_seg - resume:
        raise ValidationError(
            f"{csv.name}: {nrows} rows != NumSegments-ResumeFromSegment "
            f"({num_seg}-{resume}={num_seg - resume})")
    if abs(t[0] - resume * save_interval) > T_TOL:
        raise ValidationError(
            f"{csv.name}: first time {t[0]:.6e} != ResumeFromSegment*interval {resume * save_interval:.6e}")
    if abs(t[-1] - (num_seg - 1) * save_interval) > T_TOL:
        raise ValidationError(
            f"{csv.name}: last time {t[-1]:.6e} != (NumSegments-1)*interval "
            f"{(num_seg - 1) * save_interval:.6e}")
    checks.append(("model/alpha/N/amp/dt/stride/diagnostics", True, "ok"))
    checks.append(("columns_finite_increasing_uniform_rowcount", True,
                   f"{nrows} rows, si={save_interval:.3e}, t[0]={t[0]:.4e}, t[-1]={t[-1]:.4e}"))
    checks.append(("provenance", True, f"commit={EXPECT_COMMIT} dirty=0"))

    return Segment(
        path=csv, meta=meta, df=df, N=N, amplitude=amp, dt=dt, stride=stride,
        save_interval=save_interval, resume_from=resume, num_segments=num_seg,
        renorm_steps=renorm_steps, first_t=float(t[0]), last_t=float(t[-1]),
        nrows=nrows, E0=float(df["TotalEnergy"].iloc[0]), checks=checks)


# --------------------------------------------------------------------------- #
# Trajectory building + seam validation (physical time)
# --------------------------------------------------------------------------- #
@dataclass
class Trajectory:
    label: str
    source_type: str           # "pilot+continuation" | "gap-from-zero"
    segments: list             # list[Segment]
    df: pd.DataFrame           # concatenated
    eps_actual: float          # from the FIRST (pilot / from-zero) segment's E0
    seam_reports: list = field(default_factory=list)


def _s_of(df: pd.DataFrame) -> np.ndarray:
    """Accumulated stretch S(t) = t * LyapunovFTLE(t)."""
    return df["Time"].to_numpy() * df["LyapunovFTLE"].to_numpy()


def _tlab(x: float) -> str:
    """Compact time label, e.g. 1e7 -> '1e7', 1.4e8 -> '1.4e8' (no collisions)."""
    exp = int(np.floor(np.log10(x)))
    mant = x / 10.0 ** exp
    mant_s = f"{mant:.0f}" if abs(mant - round(mant)) < 1e-9 else f"{mant:.1f}"
    return f"{mant_s}e{exp}"


def _blabel(a: float, b: float) -> str:
    return f"[{_tlab(a)},{_tlab(b)}]"


def build_trajectory(label: str, source_type: str, segments: list) -> Trajectory:
    """Concatenate ordered segments, validating every seam in physical time.

    eps_actual is defined ONLY from the first segment's t=0 row (pilot / from-zero).
    """
    seg0 = segments[0]
    eps_actual = seg0.E0 / (seg0.N - 1)
    seam_reports = []

    for a, b in zip(segments[:-1], segments[1:]):
        expected_next = a.last_t + a.save_interval       # PHYSICAL time (stride*dt)
        gap_ok = abs(b.first_t - expected_next) <= T_TOL
        # provenance / dynamical parameters match on both sides
        eps_a = a.E0 / (a.N - 1)
        eps_b = b.E0 / (b.N - 1)                          # continuation energy: consistency ONLY
        eps_rel = abs(eps_b - eps_a) / eps_a
        eps_ok = eps_rel <= EPS_SEAM_RTOL
        params_ok = (a.N == b.N and abs(a.amplitude - b.amplitude) <= AMP_RTOL * abs(a.amplitude)
                     and abs(a.dt - b.dt) < 1e-12 and a.stride == b.stride)
        prov_ok = (b.meta.get("SolverGitCommit") == EXPECT_COMMIT and
                   b.meta.get("SolverGitDirty") == "0" and
                   b.meta.get("CheckpointOriginCommit", EXPECT_COMMIT) == EXPECT_COMMIT and
                   b.meta.get("CheckpointOriginDirty", "0") == "0")

        def _last(seg, col):
            return float(seg.df[col].iloc[-1])

        def _first(seg, col):
            return float(seg.df[col].iloc[0])

        # cumulative Lyapunov state not reset: S(t)=t*FTLE(t) continuous across the seam
        s_prev = a.last_t * _last(a, "LyapunovFTLE")
        s_next = b.first_t * _first(b, "LyapunovFTLE")
        s_rel = abs(s_next - s_prev) / abs(s_prev) if s_prev != 0 else float("nan")
        ftle_not_reset = np.isfinite(s_rel) and s_rel <= S_SEAM_RTOL
        # no diagnostic column becomes nonfinite at the seam
        seam_finite = all(np.isfinite([_last(a, c), _first(b, c)]).all() for c in REQUIRED_COLUMNS)

        if not gap_ok:
            raise ValidationError(
                f"[{label}] seam {a.path.name} -> {b.path.name}: first time "
                f"{b.first_t:.6e} != prev_last+stride*dt {expected_next:.6e}")
        if not eps_ok:
            raise ValidationError(
                f"[{label}] seam {a.path.name} -> {b.path.name}: epsilon_actual "
                f"mismatch {eps_a:.6e} vs {eps_b:.6e} (rel {eps_rel:.2e})")
        if not params_ok:
            raise ValidationError(
                f"[{label}] seam {a.path.name} -> {b.path.name}: dynamical parameter mismatch")
        if not prov_ok:
            raise ValidationError(
                f"[{label}] seam {a.path.name} -> {b.path.name}: provenance mismatch")

        seam_reports.append({
            "label": label, "prev_file": a.path.name, "next_file": b.path.name,
            "prev_last_t": a.last_t, "next_first_t": b.first_t,
            "expected_next_t": expected_next, "gap_ok": bool(gap_ok),
            "no_dup_or_skip": True,             # uniform+validated spacing => guaranteed
            "eps_prev": eps_a, "eps_next": eps_b, "eps_rel_diff": eps_rel, "eps_ok": bool(eps_ok),
            "params_ok": bool(params_ok), "provenance_ok": bool(prov_ok),
            "seam_columns_finite": bool(seam_finite),
            "ftle_not_reset(S_continuous)": bool(ftle_not_reset),
            "S_prev_last": s_prev, "S_next_first": s_next, "S_rel_diff": s_rel,
            "cont_TotalEnergy": (_last(a, "TotalEnergy"), _first(b, "TotalEnergy")),
            "cont_TodaJ": (_last(a, "TodaJ"), _first(b, "TodaJ")),
        })

    full = pd.concat([s.df for s in segments], ignore_index=True)
    t = full["Time"].to_numpy()
    if np.any(np.diff(t) <= 0):
        raise ValidationError(f"[{label}] concatenated Time not strictly increasing")
    if np.max(np.abs(np.diff(t) - seg0.save_interval)) > T_TOL:
        raise ValidationError(f"[{label}] concatenated saved-time spacing non-uniform")
    for c in REQUIRED_COLUMNS:
        if not np.all(np.isfinite(full[c].to_numpy())):
            raise ValidationError(f"[{label}] non-finite in concatenated column {c}")

    return Trajectory(label=label, source_type=source_type, segments=segments,
                      df=full, eps_actual=eps_actual, seam_reports=seam_reports)


# --------------------------------------------------------------------------- #
# Metadata-driven candidate discovery + frozen source resolution
# --------------------------------------------------------------------------- #
def _nearest_eps(eps: float) -> float:
    return min(EPS_ORDER, key=lambda e: abs(e - eps))


def _scan_from_zero(directory: Path) -> list[dict]:
    """Return metadata records for every from-t=0 alpha CSV (resume==0) in a dir.

    Cell (eps_target) is assigned from eps_actual = E0/(N-1) (own t=0 row), mapped
    to the nearest master eps_target and accepted only within CELL_RTOL. Identity
    comes from metadata + the t=0 energy, never from the filename.
    """
    recs = []
    if not directory.is_dir():
        return recs
    for p in sorted(directory.glob("*.csv")):
        m = get_metadata(str(p))
        if m.get("Model", "").lower() != "alpha":
            continue
        try:
            if abs(float(m.get("Alpha", "nan")) - 1.0) > 1e-12:
                continue
            resume = int(float(m.get("ResumeFromSegment", "0")))
            if resume != 0:
                continue
            N = int(float(m["N"]))
            dt = float(m["dt"])
        except (ValueError, TypeError, KeyError):
            continue
        # need t=0 energy for eps_actual
        e0 = pd.read_csv(p, comment="#", usecols=lambda c: c.strip() == "TotalEnergy",
                         nrows=1)
        e0.columns = e0.columns.str.strip()
        E0 = float(e0["TotalEnergy"].iloc[0])
        eps_actual = E0 / (N - 1)
        eps_t = _nearest_eps(eps_actual)
        rel = (eps_actual - eps_t) / eps_t
        recs.append({"path": p, "meta": m, "N": N, "dt": dt,
                     "amplitude": float(m["Amplitude"]), "E0": E0,
                     "eps_actual": eps_actual, "eps_target": eps_t,
                     "eps_rel_disc": rel, "in_cell": abs(rel) <= CELL_RTOL})
    return recs


def _find_continuations(directory: Path, N: int, amp: float, resume: int) -> list[Path]:
    """Continuation CSVs in ``directory`` matching (N, amplitude, ResumeFromSegment)."""
    hits = []
    if not directory.is_dir():
        return hits
    for p in sorted(directory.glob("*.csv")):
        m = get_metadata(str(p))
        if m.get("Model", "").lower() != "alpha":
            continue
        try:
            if int(float(m["N"])) != N:
                continue
            if abs(float(m["Amplitude"]) - amp) > AMP_RTOL * abs(amp):
                continue
            if int(float(m.get("ResumeFromSegment", "0"))) != resume:
                continue
        except (ValueError, TypeError, KeyError):
            continue
        hits.append(p)
    return hits


def resolve_sources(data_root: Path) -> tuple[dict, list, list]:
    """Build the canonical trajectory for every master (N, eps_target) cell.

    Returns (cells, source_map_rows, seam_rows) where ``cells`` maps
    (N, eps_target) -> {canonical: Trajectory|None, duplicate: Trajectory|None,
    notes: [...], status: 'ok'|'missing'|'failed'}.
    """
    pilot_dir = data_root / "alpha_pilot_v1"
    ext_dir = data_root / "alpha_pilot_v1_ext"
    gap_dir = data_root / "alpha_gap_v1"

    pilots = [r for r in _scan_from_zero(pilot_dir) if r["in_cell"] and r["eps_target"] in EPS_ORDER]
    gaps = [r for r in _scan_from_zero(gap_dir) if r["in_cell"] and r["eps_target"] in EPS_ORDER]
    # index by cell
    pilot_by_cell = {(r["N"], r["eps_target"]): r for r in pilots}
    gap_by_cell = {(r["N"], r["eps_target"]): r for r in gaps}

    cells: dict = {}
    source_rows: list = []
    seam_rows: list = []

    for eps_t in EPS_ORDER:
        for N in N_ORDER:
            key = (N, eps_t)
            notes = []
            candidates = []            # human-readable candidate list
            canonical = None
            duplicate = None
            status = "missing"

            pilot_rec = pilot_by_cell.get(key)
            gap_rec = gap_by_cell.get(key)

            # ---- candidate 1: pilot + continuation splice to 1e8 ----
            pilot_cont_traj = None
            if pilot_rec is not None:
                conts = _find_continuations(ext_dir, N, pilot_rec["amplitude"], resume=500)
                candidates.append(f"pilot(1e7) {pilot_rec['path'].parent.name}/{pilot_rec['path'].name}"
                                  f" [{len(conts)} matching continuation(s)]")
                if len(conts) == 1:
                    try:
                        seg_p = load_segment(pilot_rec["path"], N, pilot_rec["amplitude"], expect_dt=0.1)
                        seg_e = load_segment(conts[0], N, pilot_rec["amplitude"], expect_dt=0.1)
                        pilot_cont_traj = build_trajectory(
                            f"eps{eps_t:.0e}_N{N}_pilotcont", "pilot+continuation", [seg_p, seg_e])
                    except ValidationError as e:
                        notes.append(f"pilot+continuation FAILED validation: {e}")
                        status = "failed"
                elif len(conts) == 0:
                    notes.append("pilot present but NO continuation to 1e8 -> "
                                 "pilot alone is 1e7-only (insufficient for the [8e7,1e8) tail)")
                else:
                    notes.append(f"pilot has {len(conts)} continuations (ambiguous) -> not used")

            # ---- candidate 2: gap-from-zero to 1e8 ----
            gap_traj = None
            if gap_rec is not None:
                candidates.append(f"gap-from-zero {gap_rec['path'].parent.name}/{gap_rec['path'].name}")
                try:
                    seg_g = load_segment(gap_rec["path"], N, gap_rec["amplitude"], expect_dt=0.1)
                    gap_traj = build_trajectory(
                        f"eps{eps_t:.0e}_N{N}_gap", "gap-from-zero", [seg_g])
                except ValidationError as e:
                    notes.append(f"gap-from-zero FAILED validation: {e}")

            # ---- frozen precedence ----
            if pilot_cont_traj is not None:
                canonical = pilot_cont_traj
                status = "ok"
                if gap_traj is not None:
                    duplicate = gap_traj      # independent reproducibility check, never discarded
                    notes.append("independent gap-from-zero present -> kept as duplicate check")
            elif gap_traj is not None:
                canonical = gap_traj
                status = "ok"
            else:
                if status != "failed":
                    status = "missing"
                    notes.append("no valid pilot+continuation and no valid gap-from-zero on disk")

            cells[key] = {"canonical": canonical, "duplicate": duplicate,
                          "notes": notes, "status": status}

            # source-map row
            row = {
                "eps_target": eps_t, "N": N, "status": status,
                "canonical_source_type": canonical.source_type if canonical else "NA",
                "eps_actual": canonical.eps_actual if canonical else float("nan"),
                "eps_rel_disc": ((canonical.eps_actual - eps_t) / eps_t) if canonical else float("nan"),
                "canonical_files": ";".join(f"{s.path.parent.name}/{s.path.name}"
                                            for s in canonical.segments) if canonical else "NA",
                "duplicate_source_type": duplicate.source_type if duplicate else "NA",
                "duplicate_files": ";".join(f"{s.path.parent.name}/{s.path.name}"
                                            for s in duplicate.segments) if duplicate else "NA",
                "all_candidates": " | ".join(candidates) if candidates else "none",
                "solver_commit": EXPECT_COMMIT, "solver_dirty": 0,
                "notes": " ; ".join(notes) if notes else "",
            }
            source_rows.append(row)
            if canonical:
                for sm in canonical.seam_reports:
                    seam_rows.append({"cell": f"eps{eps_t:.0e}_N{N}", "role": "canonical", **sm})
            if duplicate:
                for sm in duplicate.seam_reports:
                    seam_rows.append({"cell": f"eps{eps_t:.0e}_N{N}", "role": "duplicate", **sm})

    return cells, source_rows, seam_rows


# --------------------------------------------------------------------------- #
# Derived quantities
# --------------------------------------------------------------------------- #
def phi_j(J: np.ndarray, eps_actual: float) -> tuple[np.ndarray, float]:
    """Phi_J = (J - J0)/(2 eps - J0). J0 > 2 eps (mode-1 IC) => denom < 0.

    NOT clipped to [0, 1] (per the wording discipline).
    """
    denom = 2.0 * eps_actual - J[0]
    return (J - J[0]) / denom, denom


def s_at(t: np.ndarray, S: np.ndarray, target: float) -> tuple[float, float, str]:
    """S(target): exact stored row within T_TOL if present, else linear interp.

    If target is beyond the last stored time (nominal terminal endpoints are one
    snapshot short) the last stored row is used and the method is reported.
    Returns (S_value, actual_time_used, method).
    """
    idx = np.where(np.abs(t - target) <= T_TOL)[0]
    if idx.size:
        i = idx[0]
        return float(S[i]), float(t[i]), "exact"
    if target > t[-1]:
        return float(S[-1]), float(t[-1]), "last_saved(beyond_end)"
    if target < t[0]:
        return float(S[0]), float(t[0]), "first_saved(before_start)"
    return float(np.interp(target, t, S)), float(target), "interpolated"


def block_rate(t: np.ndarray, S: np.ndarray, t1: float, t2: float) -> dict:
    """lambda_block = [S(t2)-S(t1)]/(t2-t1) using stored/interpolated endpoints."""
    S1, ta1, m1 = s_at(t, S, t1)
    S2, ta2, m2 = s_at(t, S, t2)
    return {"t1_req": t1, "t2_req": t2, "t1_used": ta1, "t2_used": ta2,
            "S1": S1, "S2": S2, "method_t1": m1, "method_t2": m2,
            "lambda_block": float((S2 - S1) / (ta2 - ta1))}


def _ms(a: np.ndarray) -> tuple[float, float]:
    if a.size < 2:
        return (float(a[0]) if a.size else float("nan")), float("nan")
    return float(np.mean(a)), float(np.std(a, ddof=1))


def tail_window_stats(df: pd.DataFrame, eps_actual: float,
                      lo: float = TAIL_LO, hi: float = TAIL_HI) -> dict:
    """Temporal mean/std (ddof=1) over the physical window [lo, hi)."""
    t = df["Time"].to_numpy()
    m = (t >= lo) & (t < hi)
    n = int(np.sum(m))
    twoeps = 2.0 * eps_actual
    J = df["TodaJ"].to_numpy()
    phi, denom = phi_j(J, eps_actual)
    jr = J / twoeps
    ftle = df["LyapunovFTLE"].to_numpy()
    loc = df["LyapunovLocal"].to_numpy()
    eta = df["Eta"].to_numpy()

    jm, js = _ms(jr[m]); pm, ps = _ms(phi[m])
    fm, fs = _ms(ftle[m]); lm, ls = _ms(loc[m]); em, es = _ms(eta[m])
    return {
        "tail_lo": lo, "tail_hi": hi, "tail_rows": n,
        "tail_t_first": float(t[m][0]) if n else float("nan"),
        "tail_t_last": float(t[m][-1]) if n else float("nan"),
        "J_over_2eps_tail_mean": jm, "J_over_2eps_tail_std": js,
        "PhiJ_tail_mean": pm, "PhiJ_tail_std": ps, "PhiJ_denominator": float(denom),
        "FTLE_tail_mean": fm, "FTLE_tail_std": fs, "FTLE_final": float(ftle[-1]),
        "LyapLocal_tail_mean": lm, "LyapLocal_tail_std": ls,
        "Eta_tail_mean": em, "Eta_tail_std": es,
    }


def trailing_mean(y: np.ndarray, win_rows: int) -> np.ndarray:
    """TRAILING moving mean over ``win_rows`` complete samples (min_periods=win_rows).

    Entries before a full window are NaN. A trailing (not centered) window is
    used deliberately: it avoids ambiguous edge handling at the trajectory end.
    """
    return pd.Series(y).rolling(window=win_rows, min_periods=win_rows).mean().to_numpy()


def t_first(df: pd.DataFrame, eps_actual: float, level: float) -> float | None:
    """First saved time at which raw Phi_J >= level."""
    t = df["Time"].to_numpy()
    phi, _ = phi_j(df["TodaJ"].to_numpy(), eps_actual)
    hit = np.where(phi >= level)[0]
    return float(t[hit[0]]) if hit.size else None


def t_persistent(df: pd.DataFrame, eps_actual: float, level: float,
                 win_rows: int) -> tuple[float | None, int]:
    """Earliest END time of a COMPLETE trailing window whose trailing-mean Phi_J
    is >= level AND for which every later complete trailing window (through the
    final available time) also stays >= level. Returns (time_or_None, win_rows).

    None => not reached (right-censored, reported as > last available time).
    """
    t = df["Time"].to_numpy()
    phi, _ = phi_j(df["TodaJ"].to_numpy(), eps_actual)
    sm = trailing_mean(phi, win_rows)
    valid = ~np.isnan(sm)                      # complete-window indices only
    ok = valid & (sm >= level)
    # suffix-AND over the complete-window indices: does ok hold for all valid j>=i?
    n = len(sm)
    all_true_from = np.zeros(n, dtype=bool)
    acc = True
    for j in range(n - 1, -1, -1):
        if valid[j]:
            acc = acc and bool(ok[j])
        all_true_from[j] = acc
    for i in range(n):
        if valid[i] and all_true_from[i]:
            return float(t[i]), win_rows
    return None, win_rows


def energy_drift(df: pd.DataFrame) -> float:
    E = df["TotalEnergy"].to_numpy()
    return float(np.max(np.abs(E - E[0]) / abs(E[0])))


def read_min_bond_strain(segments: list) -> dict:
    """Most-negative bond strain from run telemetry (rounded 'Done!' log line).

    The value is available ONLY in rounded log output (1 decimal place), so its
    precision is +-0.05. Reported as such; NA where no log line is present. We do
    not infer precision the log does not carry.
    """
    val = None
    src = None
    for seg in segments:
        for cand in (seg.path.with_suffix(".log"),
                     seg.path.with_suffix(".runlog"),
                     seg.path.with_suffix(".runner.log"),
                     seg.path.with_suffix(".solver.log")):
            if not cand.exists():
                continue
            for line in cand.read_text(errors="ignore").splitlines():
                if "min_bond_strain=" in line:
                    try:
                        v = float(line.split("min_bond_strain=")[1].split()[0].rstrip("|").strip())
                        val = v if val is None else min(val, v)
                        src = cand.name
                    except ValueError:
                        pass
    if val is None:
        return {"min_bond_strain": None, "precision": None, "source": "NA (no telemetry)"}
    return {"min_bond_strain": val, "precision": 0.05,
            "source": f"{src} (rounded log, 1 decimal => +-0.05)"}


# --------------------------------------------------------------------------- #
# Master-grid statistics (section 4)
# --------------------------------------------------------------------------- #
def master_grid_stats(cells: dict) -> list[dict]:
    """One record per constructable (N, eps_target) cell using the canonical source."""
    recs = []
    for eps_t in EPS_ORDER:
        for N in N_ORDER:
            cell = cells[(N, eps_t)]
            traj = cell["canonical"]
            if traj is None:
                recs.append({"eps_target": eps_t, "N": N, "status": cell["status"],
                             "source_type": "NA", "note": " ; ".join(cell["notes"])})
                continue
            df = traj.df
            t = df["Time"].to_numpy()
            eps = traj.eps_actual
            twoeps = 2.0 * eps
            J = df["TodaJ"].to_numpy()
            tail = tail_window_stats(df, eps)
            bond = read_min_bond_strain(traj.segments)
            recs.append({
                "eps_target": eps_t, "N": N, "status": cell["status"],
                "source_type": traj.source_type,
                "eps_actual": eps, "eps_rel_disc": (eps - eps_t) / eps_t,
                "J0_over_2eps": float(J[0] / twoeps),
                "J_tail_mean_over_2eps": tail["J_over_2eps_tail_mean"],
                "J_tail_std_over_2eps": tail["J_over_2eps_tail_std"],
                "PhiJ_tail_mean": tail["PhiJ_tail_mean"], "PhiJ_tail_std": tail["PhiJ_tail_std"],
                "PhiJ_denominator": tail["PhiJ_denominator"],
                "FTLE_final": tail["FTLE_final"], "FTLE_tail_mean": tail["FTLE_tail_mean"],
                "FTLE_tail_std": tail["FTLE_tail_std"],
                "LyapLocal_tail_mean": tail["LyapLocal_tail_mean"],
                "LyapLocal_tail_std": tail["LyapLocal_tail_std"],
                "Eta_tail_mean": tail["Eta_tail_mean"], "Eta_tail_std": tail["Eta_tail_std"],
                "max_abs_rel_energy_drift": energy_drift(df),
                "tail_rows": tail["tail_rows"],
                "tail_t_first": tail["tail_t_first"], "tail_t_last": tail["tail_t_last"],
                "last_time": float(t[-1]),
                "min_bond_strain": bond["min_bond_strain"],
                "min_bond_strain_precision": bond["precision"],
                "min_bond_strain_source": bond["source"],
                "note": " ; ".join(cell["notes"]),
            })
    return recs


# --------------------------------------------------------------------------- #
# Duplicate-source reproducibility audit (section 5)
# --------------------------------------------------------------------------- #
def duplicate_audit(cells: dict) -> list[dict]:
    """Compare canonical vs independent gap duplicate for cells that have both.

    Each source uses its OWN eps_actual (never cross-normalized). Compares coarse
    observables only; pointwise/bitwise agreement across machines is not required.
    """
    rows = []
    for eps_t in EPS_ORDER:
        for N in N_ORDER:
            cell = cells[(N, eps_t)]
            can, dup = cell["canonical"], cell["duplicate"]
            if can is None or dup is None:
                continue
            win = int(round(SMOOTH_WINDOW / SAVE_INTERVAL))

            def _pack(traj):
                df = traj.df
                eps = traj.eps_actual
                tail = tail_window_stats(df, eps)
                t10p, _ = t_persistent(df, eps, T10_LEVEL, win)
                t90p, _ = t_persistent(df, eps, T90_LEVEL, win)
                return {
                    "eps_actual": eps,
                    "J_tail_mean_over_2eps": tail["J_over_2eps_tail_mean"],
                    "PhiJ_tail_mean": tail["PhiJ_tail_mean"],
                    "FTLE_tail_mean": tail["FTLE_tail_mean"],
                    "LyapLocal_tail_mean": tail["LyapLocal_tail_mean"],
                    "Eta_tail_mean": tail["Eta_tail_mean"],
                    "T10_persistent": t10p, "T90_persistent": t90p,
                    "max_abs_rel_energy_drift": energy_drift(df),
                }

            a, b = _pack(can), _pack(dup)

            def _reldiff(x, y):
                if x is None or y is None:
                    return None
                if x == 0 and y == 0:
                    return 0.0
                denom = max(abs(x), abs(y))
                return abs(x - y) / denom if denom else None

            rows.append({
                "eps_target": eps_t, "N": N,
                "canonical_type": can.source_type, "duplicate_type": dup.source_type,
                "eps_actual_canonical": a["eps_actual"], "eps_actual_duplicate": b["eps_actual"],
                "J_tail_reldiff": _reldiff(a["J_tail_mean_over_2eps"], b["J_tail_mean_over_2eps"]),
                "PhiJ_tail_reldiff": _reldiff(a["PhiJ_tail_mean"], b["PhiJ_tail_mean"]),
                "FTLE_tail_reldiff": _reldiff(a["FTLE_tail_mean"], b["FTLE_tail_mean"]),
                "LyapLocal_tail_reldiff": _reldiff(a["LyapLocal_tail_mean"], b["LyapLocal_tail_mean"]),
                "Eta_tail_reldiff": _reldiff(a["Eta_tail_mean"], b["Eta_tail_mean"]),
                "T10_persistent_canonical": a["T10_persistent"], "T10_persistent_duplicate": b["T10_persistent"],
                "T90_persistent_canonical": a["T90_persistent"], "T90_persistent_duplicate": b["T90_persistent"],
                "max_drift_canonical": a["max_abs_rel_energy_drift"],
                "max_drift_duplicate": b["max_abs_rel_energy_drift"],
                "canonical_detail": a, "duplicate_detail": b,
            })
    return rows


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def _save(fig, out_dir: Path, stem: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        paths.append(str(p))
    plt.close(fig)
    return paths


def _logt(ax, t, y, style, label, **kw):
    """Plot y vs t on a log-time axis, omitting only the t==0 sample."""
    color, ls, mk = style
    m = t > 0
    me = max(1, int(np.sum(m) / 60))
    ax.plot(t[m], y[m], color=color, linestyle=ls, marker=mk, markersize=3,
            markevery=me, linewidth=kw.pop("lw", 1.3), label=label, **kw)


def _plateau_legs(rec: dict) -> dict:
    """Frozen positive-plateau legs (i)-(iii) for a master-grid record."""
    fm = rec["FTLE_tail_mean"]; ff = rec["FTLE_final"]
    lm = rec["LyapLocal_tail_mean"]; ls = rec["LyapLocal_tail_std"]
    leg_i = (fm != 0) and abs(ff - fm) <= FROZEN_FTLE_FLAT_RTOL * abs(fm)
    leg_ii = lm > 0
    leg_iii = lm > ls
    return {"ftle_flat": bool(leg_i), "loc_positive": bool(leg_ii),
            "loc_above_noise": bool(leg_iii),
            "positive_plateau_all3": bool(leg_i and leg_ii and leg_iii)}


# --------------------------------------------------------------------------- #
# TASK A — master epsilon-N summary
# --------------------------------------------------------------------------- #
def _sampled_bracket(grid_by_N: dict, N: int, predicate) -> dict:
    """Walk sampled eps (ascending) for one N; report the onset bracket.

    predicate(rec) -> bool. Returns a dict describing the sampled bracket:
    the highest eps where predicate is False immediately below the lowest eps
    where it is True, or an explicit "always"/"never"/"gap" statement. No value
    between samples is invented.
    """
    avail = [(e, grid_by_N[N][e]) for e in EPS_ORDER if e in grid_by_N.get(N, {})]
    flags = [(e, bool(predicate(r))) for e, r in avail]
    sampled = [f"{e:.0e}:{'Y' if v else 'N'}" for e, v in flags]
    if not flags:
        return {"result": "no sampled cells for this N", "sampled": sampled}
    trues = [e for e, v in flags if v]
    if not trues:
        return {"result": f"not satisfied at any sampled eps up to {flags[-1][0]:.0e}",
                "sampled": sampled}
    first_true = min(trues)
    below = [e for e, v in flags if e < first_true]
    if not below:
        return {"result": f"already satisfied at the lowest sampled eps ({first_true:.0e})",
                "sampled": sampled}
    prev = max(below)
    prev_v = dict(flags)[prev]
    if prev_v:
        return {"result": f"satisfied by {first_true:.0e} (also at the sample below)",
                "sampled": sampled}
    return {"result": f"no onset at {prev:.0e}; criterion satisfied at {first_true:.0e}",
            "bracket_low": prev, "bracket_high": first_true, "sampled": sampled}


def run_task_A(grid_recs: list, out_dir: Path) -> dict:
    ok = [r for r in grid_recs if r.get("source_type", "NA") != "NA"]
    grid_by_N = {}
    for r in ok:
        grid_by_N.setdefault(r["N"], {})[r["eps_target"]] = r

    # ---- figure ----
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(17, 5.4))
    for N in N_ORDER:
        color, ls, mk = STYLE_N[N]
        sub = sorted(grid_by_N.get(N, {}).values(), key=lambda r: r["eps_actual"])
        if not sub:
            continue
        eps = [r["eps_actual"] for r in sub]
        axA.errorbar(eps, [r["J_tail_mean_over_2eps"] for r in sub],
                     yerr=[r["J_tail_std_over_2eps"] for r in sub],
                     color=color, ls=ls, marker=mk, capsize=3, lw=1.3, label=f"N = {N}")
        axB.errorbar(eps, [r["FTLE_tail_mean"] for r in sub],
                     yerr=[r["FTLE_tail_std"] for r in sub],
                     color=color, ls=ls, marker=mk, capsize=3, lw=1.3, label=f"N = {N}")
        axC.errorbar(eps, [r["Eta_tail_mean"] for r in sub],
                     yerr=[r["Eta_tail_std"] for r in sub],
                     color=color, ls=ls, marker=mk, capsize=3, lw=1.3, label=f"N = {N}")
    for ax in (axA, axB, axC):
        ax.set_xscale("log"); ax.grid(True, which="both", alpha=0.3)
        ax.set_xlabel(r"energy density  $\epsilon_{\rm act}$ (log)")
    axA.axhline(1.0, color="0.35", ls=":", lw=1.0, label=r"$J/2\epsilon=1$ ($\alpha$ equilib. estimate)")
    axA.axhline(1.5, color="0.6", ls="--", lw=0.9, label=r"$J/2\epsilon=1.5$ (mode-1 $J(0)$ ref)")
    axA.set_ylabel(r"$\langle J/2\epsilon_{\rm act}\rangle_{\rm tail}$")
    axA.set_title("A. Toda action relaxation (tail)")
    axB.set_yscale("log")
    axB.set_ylabel(r"$\langle\lambda_{\max}\rangle_{\rm tail}$  (finite-time est.)")
    axB.set_title("B. finite-time cumulative FTLE (tail)")
    axC.set_ylabel(r"$\langle\eta\rangle_{\rm tail}$")
    axC.set_title("C. spectral entropy (tail)")
    axA.legend(fontsize=8, frameon=True)
    fig.suptitle(SUP + r" — master grid, fixed physical tail $[8\times10^{7},10^{8})$"
                 "\nerror bars = TEMPORAL std over the tail window (NOT realization uncertainty); "
                 "guide lines are not fits", y=1.02)
    fig.tight_layout()
    figs = _save(fig, out_dir, "taskA_master_summary_vs_epsilon")

    # ---- frozen observable-specific sampled brackets ----
    brackets = {"per_N": {}}
    for N in N_ORDER:
        brackets["per_N"][N] = {
            "toda_underway_PhiJ>=0.5": _sampled_bracket(
                grid_by_N, N, lambda r: r["PhiJ_tail_mean"] >= FROZEN_TODA_ONSET_PHIJ),
            "toda_near_eq_J/2eps<=1.10": _sampled_bracket(
                grid_by_N, N, lambda r: r["J_tail_mean_over_2eps"] <= FROZEN_TODA_NEAR_EQ),
            "chaos_positive_plateau_all3": _sampled_bracket(
                grid_by_N, N, lambda r: _plateau_legs(r)["positive_plateau_all3"]),
        }
    # entropy: report growth across the grid, no threshold
    brackets["entropy_note"] = (
        "eta indicates harmonic-mode spreading only; no thermalization threshold is "
        "assigned. Growth across the sampled grid is reported in the table/figure and "
        "contrasted with Toda and Lyapunov, which change at different sampled eps.")
    return {"figure": figs, "brackets": brackets,
            "plateau_legs": {f"eps{r['eps_target']:.0e}_N{r['N']}": _plateau_legs(r) for r in ok}}


# --------------------------------------------------------------------------- #
# TASK B — persistent Toda timescales (trailing window)
# --------------------------------------------------------------------------- #
def run_task_B(cells: dict, out_dir: Path) -> dict:
    win = int(round(SMOOTH_WINDOW / SAVE_INTERVAL))
    recs = []
    for eps_t in EPS_ORDER:
        for N in N_ORDER:
            traj = cells[(N, eps_t)]["canonical"]
            if traj is None:
                continue
            df = traj.df
            eps = traj.eps_actual
            t = df["Time"].to_numpy()
            last_t = float(t[-1])
            nrows = len(df)
            t10f = t_first(df, eps, T10_LEVEL)
            t90f = t_first(df, eps, T90_LEVEL)
            t10p, _ = t_persistent(df, eps, T10_LEVEL, win)
            t90p, _ = t_persistent(df, eps, T90_LEVEL, win)
            recs.append({
                "eps_target": eps_t, "N": N, "eps_actual": eps,
                "source_type": traj.source_type, "nrows": nrows, "last_time": last_t,
                "window_rows": win, "window_phys": SMOOTH_WINDOW,
                "T10_first": t10f, "T10_first_censored": t10f is None,
                "T90_first": t90f, "T90_first_censored": t90f is None,
                "T10_persistent": t10p, "T10_persistent_censored": t10p is None,
                "T90_persistent": t90p, "T90_persistent_censored": t90p is None,
            })

    # ---- figures: T10 and T90 vs eps_actual, per N, censored marked as bounds ----
    def _fig(level_key, level_lbl, stem):
        fig, ax = plt.subplots(figsize=(9, 6))
        for N in N_ORDER:
            color, ls, mk = STYLE_N[N]
            sub = sorted([r for r in recs if r["N"] == N], key=lambda r: r["eps_actual"])
            if not sub:
                continue
            eps_meas, y_meas = [], []
            eps_cens, y_cens = [], []
            for r in sub:
                val = r[f"{level_key}_persistent"]
                if val is None:
                    eps_cens.append(r["eps_actual"]); y_cens.append(r["last_time"])
                else:
                    eps_meas.append(r["eps_actual"]); y_meas.append(val)
            # measured: connect with the per-N line style
            if eps_meas:
                ax.plot(eps_meas, y_meas, color=color, ls=ls, marker=mk, lw=1.3,
                        label=f"N = {N} (measured)")
            # censored: caret-up lower-bound markers at the last available time, NOT connected
            if eps_cens:
                ax.scatter(eps_cens, y_cens, marker="^", s=70, facecolors="none",
                           edgecolors=color, linewidths=1.4,
                           label=f"N = {N} (censored, $>$ last time)")
        ax.axhline(NOM_1E8, color="0.6", ls=":", lw=1.0)
        ax.text(ax.get_xlim()[0], NOM_1E8, " nominal $10^8$", va="bottom", ha="left",
                fontsize=7, color="0.4")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlabel(r"energy density  $\epsilon_{\rm act}$ (log)")
        ax.set_ylabel(f"{level_lbl} (persistent, trailing $2\\times10^6$ window)")
        ax.set_title(SUP + f" — {level_lbl}: persistent Toda timescale vs $\\epsilon$"
                     "\ncensored points = lower bounds (not reached by the end); not connected, not fit")
        ax.legend(fontsize=8, frameon=True)
        fig.tight_layout()
        return _save(fig, out_dir, stem)

    fig10 = _fig("T10", "$T_{10}$", "taskB_T10_persistent_vs_epsilon")
    fig90 = _fig("T90", "$T_{90}$", "taskB_T90_persistent_vs_epsilon")

    # ---- summary readings ----
    def _lowest_reached(level_key):
        by_N = {}
        for N in N_ORDER:
            sub = sorted([r for r in recs if r["N"] == N], key=lambda r: r["eps_actual"])
            reached = [r["eps_target"] for r in sub if r[f"{level_key}_persistent"] is not None]
            by_N[N] = (min(reached) if reached else None)
        return by_N

    low10 = _lowest_reached("T10")
    low90 = _lowest_reached("T90")
    consistent10 = len({v for v in low10.values() if v is not None}) <= 1
    consistent90 = len({v for v in low90.values() if v is not None}) <= 1
    return {"records": recs, "figures": {"T10": fig10, "T90": fig90},
            "lowest_eps_T10_persistent_by_N": low10,
            "lowest_eps_T90_persistent_by_N": low90,
            "T10_bracket_consistent_across_N": consistent10,
            "T90_bracket_consistent_across_N": consistent90}


# --------------------------------------------------------------------------- #
# TASK C — low-epsilon deep-time behavior
# --------------------------------------------------------------------------- #
def _classify_finite_time(block_rate_vals: list, loc_mean: float, loc_std: float) -> str:
    """Descriptive finite-time classification from block rates + LyapunovLocal.

    One of: 'Lyapunov block rates trending toward zero';
    'block rates stabilizing at a positive value';
    'intermittent or non-monotone block rates';
    'inconclusive over the observed duration'.
    Never calls a small positive cumulative FTLE a stable plateau on its own.
    """
    r = np.array(block_rate_vals, dtype=float)
    allpos = np.all(r > 0)
    signs_mixed = np.any(r > 0) and np.any(r < 0)
    decreasing = np.all(np.diff(r) < 0)
    scale = np.mean(np.abs(r)) if np.mean(np.abs(r)) > 0 else 1.0
    roughly_constant = (np.max(r) - np.min(r)) <= 0.5 * scale
    loc_pos_above_noise = (loc_mean > 0) and (loc_mean > loc_std)   # leg (iii)

    if allpos and roughly_constant and loc_pos_above_noise:
        return "block rates stabilizing at a positive value"
    if decreasing and r[-1] < 0.5 * max(abs(r[0]), 1e-30):
        return "Lyapunov block rates trending toward zero"
    if signs_mixed or not roughly_constant:
        return "intermittent or non-monotone block rates"
    return "inconclusive over the observed duration"


def _block_toda_loc(t, jr, phi, loc, a, b):
    """Block means over [a, min(b,t[-1])]: J/2eps, Phi_J, start->end dJ/2eps, LyapLocal."""
    m = (t >= a - T_TOL) & (t <= min(b, t[-1]) + T_TOL)
    js, ps, ls = jr[m], phi[m], loc[m]
    lm, lsd = _ms(ls)
    return {
        "block_mean_J_over_2eps": float(np.mean(js)),
        "block_mean_PhiJ": float(np.mean(ps)),
        "start_J_over_2eps": float(js[0]), "end_J_over_2eps": float(js[-1]),
        "start_to_end_change_J_over_2eps": float(js[-1] - js[0]),
        "LyapLocal_block_mean": lm, "LyapLocal_block_std": lsd,
        "n_rows": int(np.sum(m)), "t_first": float(t[m][0]), "t_last": float(t[m][-1]),
    }


def _build_1p4e8(data_root: Path, N: int, amp: float) -> Trajectory | None:
    """Build the eps=1e-4 N=2048 trajectory to nominal 1.4e8 (pilot+cont+cont14)."""
    pilot = _find_continuations(data_root / "alpha_pilot_v1", N, amp, resume=0)
    ec1 = _find_continuations(data_root / "alpha_pilot_v1_ext", N, amp, resume=500)
    ec2 = _find_continuations(data_root / "alpha_pilot_v1_ext14", N, amp, resume=5000)
    if len(pilot) != 1 or len(ec1) != 1 or len(ec2) != 1:
        return None
    segs = [load_segment(pilot[0], N, amp, 0.1),
            load_segment(ec1[0], N, amp, 0.1),
            load_segment(ec2[0], N, amp, 0.1)]
    return build_trajectory(f"eps1e-4_N{N}_1.4e8", "pilot+continuation", segs)


def run_task_C(cells: dict, data_root: Path, out_dir: Path) -> dict:
    points = []           # (label, eps_t, N, trajectory, intervals, is_1p4e8)
    base_blocks = [(1e7, 3e7), (3e7, 6e7), (6e7, 1e8)]
    notes = []

    # eps=6e-5, all three N (N=1024 may be missing)
    for N in N_ORDER:
        traj = cells[(N, 6e-5)]["canonical"]
        if traj is None:
            notes.append(f"eps=6e-5 N={N}: no canonical source on disk -> omitted from Task C")
            continue
        points.append((f"eps6e-5_N{N}", 6e-5, N, traj, base_blocks, False))

    # eps=1e-4: N=512,1024 to 1e8; N=2048 to 1.4e8
    for N in (512, 1024):
        traj = cells[(N, 1e-4)]["canonical"]
        if traj is None:
            notes.append(f"eps=1e-4 N={N}: no canonical source -> omitted")
            continue
        points.append((f"eps1e-4_N{N}", 1e-4, N, traj, base_blocks, False))
    # N=2048 to 1.4e8
    amp2048 = None
    can2048 = cells[(2048, 1e-4)]["canonical"]
    if can2048 is not None:
        amp2048 = can2048.segments[0].amplitude
    traj14 = _build_1p4e8(data_root, 2048, amp2048) if amp2048 is not None else None
    if traj14 is not None:
        points.append(("eps1e-4_N2048_1.4e8", 1e-4, 2048,
                       traj14, base_blocks + [(1e8, 1.4e8)], True))
    else:
        notes.append("eps=1e-4 N=2048 to 1.4e8: could not build pilot+cont+cont14 -> "
                     "falling back to 1e8 canonical")
        if can2048 is not None:
            points.append(("eps1e-4_N2048", 1e-4, 2048, can2048, base_blocks, False))

    recs = []
    for label, eps_t, N, traj, intervals, is14 in points:
        df = traj.df
        t = df["Time"].to_numpy()
        S = _s_of(df)
        eps = traj.eps_actual
        twoeps = 2.0 * eps
        J = df["TodaJ"].to_numpy()
        phi, _ = phi_j(J, eps)
        jr = J / twoeps
        loc = df["LyapunovLocal"].to_numpy()

        blk = {}
        for a, b in intervals:
            br = block_rate(t, S, a, b)
            tl = _block_toda_loc(t, jr, phi, loc, a, b)
            blk[_blabel(a, b)] = {**br, **tl}
        rate_vals = [blk[k]["lambda_block"] for k in blk]
        # classification uses the last-block LyapunovLocal stats
        last_key = list(blk.keys())[-1]
        classification = _classify_finite_time(
            rate_vals, blk[last_key]["LyapLocal_block_mean"], blk[last_key]["LyapLocal_block_std"])
        interp_used = any(blk[k]["method_t1"] != "exact" or
                          blk[k]["method_t2"] not in ("exact", "last_saved(beyond_end)")
                          for k in blk)
        recs.append({
            "label": label, "eps_target": eps_t, "N": N, "eps_actual": eps,
            "source_type": traj.source_type, "last_time": float(t[-1]),
            "to_1.4e8": is14, "blocks": blk, "block_rate_vals": rate_vals,
            "classification": classification, "any_interpolated_endpoint": interp_used,
        })

    # ---- figure: block rates (A) and block-mean J/2eps (B) ----
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))
    palette = ["C0", "C1", "C3", "C2", "C4", "C5", "C6"]
    for i, r in enumerate(recs):
        color = palette[i % len(palette)]
        keys = list(r["blocks"].keys())
        x = np.arange(len(keys))
        lbl = f"$\\epsilon$={r['eps_target']:.0e}, N={r['N']}" + ("  (1.4e8)" if r["to_1.4e8"] else "")
        axA.plot(x, [r["blocks"][k]["lambda_block"] for k in keys], marker="o",
                 color=color, lw=1.3, label=lbl)
        axB.plot(x, [r["blocks"][k]["block_mean_J_over_2eps"] for k in keys], marker="s",
                 color=color, lw=1.3, label=lbl)
    maxblocks = max((len(r["blocks"]) for r in recs), default=3)
    xticklabels = ["[1e7,3e7]", "[3e7,6e7]", "[6e7,1e8]", "[1e8,1.4e8]"][:maxblocks]
    for ax in (axA, axB):
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_xticklabels(xticklabels, rotation=15, fontsize=8)
        ax.grid(True, alpha=0.3); ax.set_xlabel("block interval")
    axA.axhline(0.0, color="0.3", lw=1.0)
    axA.set_ylabel(r"$\lambda_{\rm block}=[S(t_2)-S(t_1)]/(t_2-t_1)$")
    axA.set_title("A. block Lyapunov growth rate (accumulated stretch)")
    axB.axhline(1.0, color="0.35", ls=":", lw=1.0)
    axB.axhline(1.5, color="0.6", ls="--", lw=0.9)
    axB.set_ylabel(r"block-mean $J/(2\epsilon_{\rm act})$")
    axB.set_title("B. block-mean Toda action (approach to $2\\epsilon$ estimate from above)")
    axA.legend(fontsize=7, frameon=True)
    fig.suptitle(SUP + " — low-$\\epsilon$ deep-time block diagnostics"
                 "\nblock slopes of $S(t)=t\\,\\lambda_{\\rm FTLE}(t)$ (preferred over noisy "
                 "$\\lambda_{\\rm loc}$); no power-law fit", y=1.02)
    fig.tight_layout()
    figs = _save(fig, out_dir, "taskC_lowEps_block_rates")
    return {"records": recs, "figure": figs, "notes": notes}


# --------------------------------------------------------------------------- #
# TASK D — dt and Lyapunov-cadence robustness (eps=1e-4, N=512)
# --------------------------------------------------------------------------- #
def _window_block(df, eps, a, b):
    """Physical-window [a,b) stats + block lambda for a dt-robustness comparison."""
    t = df["Time"].to_numpy()
    S = _s_of(df)
    twoeps = 2.0 * eps
    J = df["TodaJ"].to_numpy()
    phi, _ = phi_j(J, eps)
    jr = J / twoeps
    loc = df["LyapunovLocal"].to_numpy()
    eta = df["Eta"].to_numpy()
    ftle = df["LyapunovFTLE"].to_numpy()
    m = (t >= a - T_TOL) & (t < b)
    jm, js = _ms(jr[m]); pm, ps = _ms(phi[m]); em, es = _ms(eta[m]); lm, ls = _ms(loc[m])
    br = block_rate(t, S, a, b)
    # cumulative FTLE at endpoints (exact/interp handled by s_at on S/t not needed here)
    f1 = float(np.interp(a, t, ftle)); f2 = float(np.interp(min(b, t[-1]), t, ftle))
    return {
        "window": [a, b], "n_rows": int(np.sum(m)),
        "J_over_2eps_mean": jm, "J_over_2eps_std": js,
        "PhiJ_mean": pm, "PhiJ_std": ps,
        "PhiJ_progress_start_to_end": float(phi[m][-1] - phi[m][0]) if np.sum(m) else float("nan"),
        "Eta_mean": em, "Eta_std": es,
        "LyapLocal_mean": lm, "LyapLocal_std": ls,
        "cumFTLE_start": f1, "cumFTLE_end": f2,
        "lambda_block": br["lambda_block"],
    }


def run_task_D(cells: dict, data_root: Path, out_dir: Path) -> dict:
    N, eps_t = 512, 1e-4
    canA = cells[(N, eps_t)]["canonical"]        # dt=0.1 renorm100 pilot+cont (0->1e8)
    if canA is None or canA.source_type != "pilot+continuation":
        raise ValidationError("Task D: dt=0.1 canonical (pilot+cont) for eps=1e-4 N=512 not available")
    amp = canA.segments[0].amplitude

    # dt=0.05 renorm200 (cadence-matched, 0->3e7)
    rn200 = _find_continuations(data_root / "alpha_dt05_renorm200", N, amp, resume=0)
    # dt=0.05 renorm100 (long, non-matched, 0->1e8)
    rn100 = _find_continuations(data_root / "alpha_dt05_check", N, amp, resume=0)
    if len(rn200) != 1 or len(rn100) != 1:
        raise ValidationError(f"Task D: dt=0.05 inputs ambiguous/missing "
                              f"(renorm200={len(rn200)}, renorm100={len(rn100)})")
    segB = load_segment(rn200[0], N, amp, expect_dt=0.05)
    trajB = build_trajectory("dt0.05_renorm200_3e7", "gap-from-zero", [segB])
    segL = load_segment(rn100[0], N, amp, expect_dt=0.05)
    trajL = build_trajectory("dt0.05_renorm100_1e8", "gap-from-zero", [segL])

    mA = canA.segments[0].meta
    mB = segB.meta
    mL = segL.meta

    def _renorm_interval(seg):
        return seg.renorm_steps * seg.dt

    epsA = canA.eps_actual
    epsB = trajB.eps_actual
    epsL = trajL.eps_actual

    # config validation
    def _same(a, b, k):
        return a.get(k) == b.get(k)
    config = {
        "N_match": segB.N == segL.N == N,
        "model_match": (mA.get("Model") == mB.get("Model") == mL.get("Model") == "alpha"),
        "alpha_match": _same(mA, mB, "Alpha") and _same(mA, mL, "Alpha"),
        "shape_ic_match": _same(mA, mB, "Shape") and _same(mA, mL, "Shape"),
        "diagnostics_match": all(_same(mA, mB, k) and _same(mA, mL, k)
                                 for k in ("Entropy", "TodaIntegral", "Lyapunov")),
        "lyap_seed": mA.get("LyapSeed"),
        "lyap_seed_match": _same(mA, mB, "LyapSeed") and _same(mA, mL, "LyapSeed"),
        "amplitude_A": canA.segments[0].amplitude, "amplitude_B": segB.amplitude,
        "amplitude_L": segL.amplitude,
        "eps_actual_dt01": epsA, "eps_actual_dt05_rn200": epsB, "eps_actual_dt05_rn100": epsL,
        "renorm_interval_dt01": _renorm_interval(canA.segments[0]),
        "renorm_interval_dt05_rn200": _renorm_interval(segB),
        "renorm_interval_dt05_rn100": _renorm_interval(segL),
        "cadence_matched_A_vs_B": abs(_renorm_interval(canA.segments[0]) - _renorm_interval(segB)) < 1e-9,
        "cadence_matched_A_vs_L": abs(_renorm_interval(canA.segments[0]) - _renorm_interval(segL)) < 1e-9,
    }

    # ---- cadence-matched comparison over [1e7,3e7] (A vs B) ----
    matched = {
        "block": [1e7, 3e7],
        "dt0.1_renorm100": _window_block(canA.df, epsA, 1e7, 3e7),
        "dt0.05_renorm200": _window_block(trajB.df, epsB, 1e7, 3e7),
        "energy_drift_dt0.1_full": energy_drift(canA.df),
        "energy_drift_dt0.05_rn200_full": energy_drift(trajB.df),
    }
    # sub-window [1e7,2e7) as an extra block supported by both
    matched["subblock_[1e7,2e7)"] = {
        "dt0.1_renorm100": _window_block(canA.df, epsA, 1e7, 2e7),
        "dt0.05_renorm200": _window_block(trajB.df, epsB, 1e7, 2e7),
    }

    # ---- long non-cadence-matched tail comparison over [8e7,1e8) (A vs L) ----
    tailA = tail_window_stats(canA.df, epsA)
    tailL = tail_window_stats(trajL.df, epsL)
    driftA = energy_drift(canA.df)
    driftL = energy_drift(trajL.df)
    long_cmp = {
        "tail_window": [TAIL_LO, TAIL_HI],
        "dt0.1_renorm100": {
            "J_tail_mean": tailA["J_over_2eps_tail_mean"], "J_tail_std": tailA["J_over_2eps_tail_std"],
            "PhiJ_tail_mean": tailA["PhiJ_tail_mean"], "PhiJ_tail_std": tailA["PhiJ_tail_std"],
            "Eta_tail_mean": tailA["Eta_tail_mean"], "Eta_tail_std": tailA["Eta_tail_std"],
            "FTLE_tail_mean": tailA["FTLE_tail_mean"], "max_abs_rel_energy_drift": driftA},
        "dt0.05_renorm100": {
            "J_tail_mean": tailL["J_over_2eps_tail_mean"], "J_tail_std": tailL["J_over_2eps_tail_std"],
            "PhiJ_tail_mean": tailL["PhiJ_tail_mean"], "PhiJ_tail_std": tailL["PhiJ_tail_std"],
            "Eta_tail_mean": tailL["Eta_tail_mean"], "Eta_tail_std": tailL["Eta_tail_std"],
            "FTLE_tail_mean": tailL["FTLE_tail_mean"], "max_abs_rel_energy_drift": driftL,
            "cumFTLE_caveat": "different physical renorm cadence (5 vs 10); FTLE order-of-magnitude only",
            "LyapunovLocal_excluded": "NOT compared apples-to-apples (renorm cadence differs)"},
        "energy_drift_ratio_dt01_over_dt005": driftA / driftL if driftL else None,
        "drift_ratio_vs_16_4thorder_expectation": (driftA / driftL / 16.0) if driftL else None,
    }

    # ---- figure: matched block (top) + long tail (bottom) ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    (axJ, axP), (axE, axEn) = axes
    # top row: A vs B over [1e7,3e7]
    for lab, traj, eps, style in [("dt=0.10 renorm100", canA, epsA, STYLE_DT[0.1]),
                                  ("dt=0.05 renorm200", trajB, epsB, STYLE_DT[0.05])]:
        df = traj.df; t = df["Time"].to_numpy()
        msk = (t >= 1e7 - T_TOL) & (t <= 3e7 + T_TOL)
        J = df["TodaJ"].to_numpy(); phi, _ = phi_j(J, eps)
        color, ls, mk = style
        axJ.plot(t[msk], (J / (2 * eps))[msk], color=color, ls=ls, lw=1.3, label=lab)
        axP.plot(t[msk], phi[msk], color=color, ls=ls, lw=1.3, label=lab)
    axJ.set_title("A. cadence-matched $[10^7,3\\times10^7]$: $J/2\\epsilon$")
    axJ.axhline(1.5, color="0.6", ls="--", lw=0.9); axJ.axhline(1.0, color="0.35", ls=":", lw=1.0)
    axJ.set_ylabel(r"$J/(2\epsilon_{\rm act})$")
    axP.set_title("B. cadence-matched $[10^7,3\\times10^7]$: $\\Phi_J$")
    axP.set_ylabel(r"$\Phi_J$")
    # bottom row: A vs L full 0->1e8 (Eta and energy drift)
    for lab, traj, eps, style in [("dt=0.10 renorm100", canA, epsA, STYLE_DT[0.1]),
                                  ("dt=0.05 renorm100", trajL, epsL, STYLE_DT[0.05])]:
        df = traj.df; t = df["Time"].to_numpy()
        _logt(axE, t, df["Eta"].to_numpy(), style, lab)
        E = df["TotalEnergy"].to_numpy()
        _logt(axEn, t, np.abs(E - E[0]) / abs(E[0]), style, lab)
    axE.set_title("C. long $0\\to10^8$: spectral entropy $\\eta$"); axE.set_ylabel(r"$\eta$")
    axE.set_ylim(-0.02, 1.02); axE.set_xscale("log")
    axEn.set_title("D. long $0\\to10^8$: relative energy error"); axEn.set_ylabel(r"$|E-E_0|/|E_0|$")
    axEn.set_xscale("log"); axEn.set_yscale("log")
    for ax in (axJ, axP):
        ax.grid(True, alpha=0.3); ax.set_xlabel("time $t$")
    for ax in (axE, axEn):
        ax.grid(True, which="both", alpha=0.3); ax.set_xlabel(r"time $t$ (log; $t=0$ omitted)")
    for ax in (axJ, axP, axE, axEn):
        ax.legend(fontsize=8, frameon=True)
    fig.suptitle(SUP + r", $\epsilon=10^{-4}$, $N=512$ — dt / Lyapunov-cadence robustness"
                 "\ntop: cadence-MATCHED (renorm interval 10) over $[10^7,3\\times10^7]$; "
                 "bottom: long non-matched $0\\to10^8$ (FTLE caveated, $\\lambda_{\\rm loc}$ excluded)",
                 y=1.0)
    fig.tight_layout()
    figs = _save(fig, out_dir, "taskD_dt_cadence_robustness")
    return {"config_validation": config, "cadence_matched_3e7": matched,
            "long_nonmatched_1e8": long_cmp, "figure": figs,
            "inputs": {
                "dt0.1_renorm100": [f"{s.path.parent.name}/{s.path.name}" for s in canA.segments],
                "dt0.05_renorm200": f"{segB.path.parent.name}/{segB.path.name}",
                "dt0.05_renorm100": f"{segL.path.parent.name}/{segL.path.name}"}}


# --------------------------------------------------------------------------- #
# Reporting: JSON, CSV tables, Markdown
# --------------------------------------------------------------------------- #
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _fmt(x, e=False):
    if x is None:
        return "NA"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        return f"{x:.4e}" if e else f"{x:.4f}"
    except (TypeError, ValueError):
        return str(x)


def write_tables(out_dir, source_rows, seam_rows, grid_recs, dup_rows,
                 taskB, taskC, taskD) -> dict:
    paths = {}
    pd.DataFrame(source_rows).to_csv(out_dir / "grid_source_resolution.csv", index=False)
    paths["source_resolution"] = str(out_dir / "grid_source_resolution.csv")

    if seam_rows:
        pd.DataFrame(seam_rows).to_csv(out_dir / "grid_seam_validation.csv", index=False)
    else:
        pd.DataFrame([{"note": "no splice seams (all canonical sources single-segment)"}]).to_csv(
            out_dir / "grid_seam_validation.csv", index=False)
    paths["seam_validation"] = str(out_dir / "grid_seam_validation.csv")

    pd.DataFrame(grid_recs).to_csv(out_dir / "grid_master_statistics.csv", index=False)
    paths["master_grid"] = str(out_dir / "grid_master_statistics.csv")

    if dup_rows:
        flat = [{k: v for k, v in r.items() if k not in ("canonical_detail", "duplicate_detail")}
                for r in dup_rows]
        pd.DataFrame(flat).to_csv(out_dir / "grid_duplicate_reproducibility.csv", index=False)
    else:
        pd.DataFrame([{"note": "NO cell has both a pilot+continuation-to-1e8 canonical source "
                       "and an independent gap-from-zero-to-1e8 source; duplicate audit is empty"}]).to_csv(
            out_dir / "grid_duplicate_reproducibility.csv", index=False)
    paths["duplicate"] = str(out_dir / "grid_duplicate_reproducibility.csv")

    pd.DataFrame(taskB["records"]).to_csv(out_dir / "taskB_T10_T90_timescales.csv", index=False)
    paths["taskB"] = str(out_dir / "taskB_T10_T90_timescales.csv")

    # Task C: flatten blocks
    crows = []
    for r in taskC["records"]:
        for k, b in r["blocks"].items():
            crows.append({
                "label": r["label"], "eps_target": r["eps_target"], "N": r["N"],
                "eps_actual": r["eps_actual"], "block": k, "lambda_block": b["lambda_block"],
                "block_mean_J_over_2eps": b["block_mean_J_over_2eps"],
                "block_mean_PhiJ": b["block_mean_PhiJ"],
                "start_to_end_change_J_over_2eps": b["start_to_end_change_J_over_2eps"],
                "LyapLocal_block_mean": b["LyapLocal_block_mean"],
                "LyapLocal_block_std": b["LyapLocal_block_std"],
                "endpoint_method": f"{b['method_t1']}/{b['method_t2']}",
                "classification": r["classification"],
            })
    pd.DataFrame(crows).to_csv(out_dir / "taskC_lowEps_block_rates.csv", index=False)
    paths["taskC"] = str(out_dir / "taskC_lowEps_block_rates.csv")

    # Task D: flatten to a few rows
    m = taskD["cadence_matched_3e7"]; lng = taskD["long_nonmatched_1e8"]
    drows = [
        {"comparison": "cadence-matched [1e7,3e7]", "quantity": "J/2eps block mean",
         "dt0.1_rn100": m["dt0.1_renorm100"]["J_over_2eps_mean"],
         "dt0.05_rn200": m["dt0.05_renorm200"]["J_over_2eps_mean"]},
        {"comparison": "cadence-matched [1e7,3e7]", "quantity": "PhiJ block mean",
         "dt0.1_rn100": m["dt0.1_renorm100"]["PhiJ_mean"],
         "dt0.05_rn200": m["dt0.05_renorm200"]["PhiJ_mean"]},
        {"comparison": "cadence-matched [1e7,3e7]", "quantity": "Eta block mean",
         "dt0.1_rn100": m["dt0.1_renorm100"]["Eta_mean"],
         "dt0.05_rn200": m["dt0.05_renorm200"]["Eta_mean"]},
        {"comparison": "cadence-matched [1e7,3e7]", "quantity": "lambda_block (S slope)",
         "dt0.1_rn100": m["dt0.1_renorm100"]["lambda_block"],
         "dt0.05_rn200": m["dt0.05_renorm200"]["lambda_block"]},
        {"comparison": "cadence-matched [1e7,3e7]", "quantity": "LyapLocal block mean",
         "dt0.1_rn100": m["dt0.1_renorm100"]["LyapLocal_mean"],
         "dt0.05_rn200": m["dt0.05_renorm200"]["LyapLocal_mean"]},
        {"comparison": "cadence-matched [1e7,3e7]", "quantity": "LyapLocal block std",
         "dt0.1_rn100": m["dt0.1_renorm100"]["LyapLocal_std"],
         "dt0.05_rn200": m["dt0.05_renorm200"]["LyapLocal_std"]},
        {"comparison": "long tail [8e7,1e8)", "quantity": "J/2eps tail mean",
         "dt0.1_rn100": lng["dt0.1_renorm100"]["J_tail_mean"],
         "dt0.05_rn200": lng["dt0.05_renorm100"]["J_tail_mean"]},
        {"comparison": "long tail [8e7,1e8)", "quantity": "PhiJ tail mean",
         "dt0.1_rn100": lng["dt0.1_renorm100"]["PhiJ_tail_mean"],
         "dt0.05_rn200": lng["dt0.05_renorm100"]["PhiJ_tail_mean"]},
        {"comparison": "long tail [8e7,1e8)", "quantity": "Eta tail mean",
         "dt0.1_rn100": lng["dt0.1_renorm100"]["Eta_tail_mean"],
         "dt0.05_rn200": lng["dt0.05_renorm100"]["Eta_tail_mean"]},
        {"comparison": "long tail [8e7,1e8)", "quantity": "max |dE|/E0",
         "dt0.1_rn100": lng["dt0.1_renorm100"]["max_abs_rel_energy_drift"],
         "dt0.05_rn200": lng["dt0.05_renorm100"]["max_abs_rel_energy_drift"]},
    ]
    pd.DataFrame(drows).to_csv(out_dir / "taskD_dt_robustness.csv", index=False)
    paths["taskD"] = str(out_dir / "taskD_dt_robustness.csv")
    return paths


def write_report(out_dir, source_rows, seam_rows, grid_recs, dup_rows,
                 taskA, taskB, taskC, taskD, notes) -> str:
    L = []
    A = L.append
    A("# FPUT-$\\alpha$ master $\\epsilon$-$N$ grid analysis (v1)\n")
    A("_Read-only analysis of frozen trajectories. No simulation was run or resumed; "
      "no raw CSV, checkpoint, manifest, log, or solver source was modified._\n")
    A("All finite-time language follows `docs/MANUSCRIPT_WORDING_TODO.md`: `J = 2ε` is the "
      "α **equilibrium estimate** (not a theorem, not a β result); the mode-1 IC gives "
      "`J(0) ≈ 3ε`, so `J` approaches `2ε` **from above** and `Φ_J` rises 0→1. Reported "
      "standard deviations are **temporal** variation over the stated window, not realization "
      "uncertainty. No power law is fit; no critical ε is asserted. Toda drift is judged by "
      "block means / smoothed trends. Guide lines are not fits.\n")

    # ---- source resolution ----
    A("\n## 1. Source resolution & validation\n")
    A("Frozen precedence per `(N, ε)`: (1) pilot+continuation splice to 1e8; "
      "(2) else gap-from-zero to 1e8; (3) if both, the second is an independent duplicate "
      "check, never discarded. Discovery is by CSV metadata + the t=0 energy, never by "
      "filename. Full map in `grid_source_resolution.csv`.\n")
    A("| ε_target | N | status | canonical | ε_actual | (ε_act−ε_t)/ε_t | note |")
    A("|---|---|---|---|---|---|---|")
    for r in source_rows:
        A(f"| {r['eps_target']:.0e} | {r['N']} | {r['status']} | {r['canonical_source_type']} "
          f"| {_fmt(r['eps_actual'], True)} | {_fmt(r['eps_rel_disc'], True)} | {r['notes']} |")
    missing = [r for r in source_rows if r["status"] != "ok"]
    missing_lbl = ", ".join("{:.0e}/N{}".format(m["eps_target"], m["N"]) for m in missing) or "none"
    n_ok = sum(1 for r in source_rows if r["status"] == "ok")
    A(f"\n**Constructable cells:** {n_ok}/18. "
      f"**Missing/failed:** {len(missing)} ({missing_lbl}).\n")

    # ---- seam validation ----
    A("\n## 2. Seam / provenance validation\n")
    if seam_rows:
        A(f"{len(seam_rows)} splice seam(s) validated in physical time "
          "(`expected_next = prev_last + stride·dt`), no duplicated/skipped row, ε consistent, "
          "provenance `commit=4a66fec`/`dirty=0` on both sides, and `S(t)=t·λ_FTLE` continuous "
          "(cumulative Lyapunov state not reset). Detail in `grid_seam_validation.csv`.\n")
        A("| cell | role | prev→next | expected_next | next_first | gap_ok | ε_rel | S_rel | FTLE_not_reset |")
        A("|---|---|---|---|---|---|---|---|---|")
        for s in seam_rows:
            A(f"| {s['cell']} | {s['role']} | {s['prev_file'][:22]}→{s['next_file'][:22]} "
              f"| {_fmt(s['expected_next_t'], True)} | {_fmt(s['next_first_t'], True)} "
              f"| {s['gap_ok']} | {_fmt(s['eps_rel_diff'], True)} | {_fmt(s['S_rel_diff'], True)} "
              f"| {s['ftle_not_reset(S_continuous)']} |")
    else:
        A("No splice seams: every canonical source is a single from-zero segment.\n")
    # tail row counts
    A("\n**Tail-window row counts** (fixed physical `[8e7,1e8)`, nominally 1000 rows):\n")
    A("| ε_target | N | source | tail_rows | tail_t_first | tail_t_last |")
    A("|---|---|---|---|---|---|")
    for r in grid_recs:
        if r.get("source_type", "NA") == "NA":
            continue
        A(f"| {r['eps_target']:.0e} | {r['N']} | {r['source_type']} | {r['tail_rows']} "
          f"| {_fmt(r['tail_t_first'], True)} | {_fmt(r['tail_t_last'], True)} |")

    # ---- master grid ----
    A("\n## 3. Master-grid statistics (Task A table)\n")
    A("| ε_t | N | src | ε_act | J(0)/2ε | ⟨J/2ε⟩ | σ(J/2ε) | ⟨Φ_J⟩ | ⟨FTLE⟩ | FTLE_fin | "
      "⟨λ_loc⟩ | σ(λ_loc) | ⟨η⟩ | max|ΔE|/E0 | min_bond |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in grid_recs:
        if r.get("source_type", "NA") == "NA":
            continue
        mb = r["min_bond_strain"]
        mb_s = "NA" if mb is None else f"{mb:.1f}±0.05"
        A(f"| {r['eps_target']:.0e} | {r['N']} | {'P+C' if r['source_type']=='pilot+continuation' else 'gap'} "
          f"| {_fmt(r['eps_actual'], True)} | {_fmt(r['J0_over_2eps'])} | {_fmt(r['J_tail_mean_over_2eps'])} "
          f"| {_fmt(r['J_tail_std_over_2eps'], True)} | {_fmt(r['PhiJ_tail_mean'])} "
          f"| {_fmt(r['FTLE_tail_mean'], True)} | {_fmt(r['FTLE_final'], True)} "
          f"| {_fmt(r['LyapLocal_tail_mean'], True)} | {_fmt(r['LyapLocal_tail_std'], True)} "
          f"| {_fmt(r['Eta_tail_mean'])} | {_fmt(r['max_abs_rel_energy_drift'], True)} | {mb_s} |")
    A("\nmin_bond_strain is coarse **rounded log telemetry** (1 decimal ⇒ ±0.05), a dynamic "
      "most-negative bond strain, not high-precision; `NA` where no telemetry line exists.\n")

    # ---- Task A brackets ----
    A(f"\n## Task A — observable-specific sampled crossover brackets\n")
    A(f"Figure: `{taskA['figure'][0]}` (+`.pdf`). Frozen criteria (declared before viewing "
      f"curves): Toda-underway `⟨Φ_J⟩≥{FROZEN_TODA_ONSET_PHIJ}`; Toda-near-eq "
      f"`⟨J/2ε⟩≤{FROZEN_TODA_NEAR_EQ}`; chaos positive plateau = all three legs "
      "(FTLE flat ∧ ⟨λ_loc⟩>0 ∧ ⟨λ_loc⟩>σ(λ_loc)). Brackets are sampled statements, not "
      "interpolated critical ε.\n")
    for N in N_ORDER:
        b = taskA["brackets"]["per_N"][N]
        A(f"- **N={N}**: Toda-underway → {b['toda_underway_PhiJ>=0.5']['result']}; "
          f"Toda-near-eq → {b['toda_near_eq_J/2eps<=1.10']['result']}; "
          f"chaos-plateau → {b['chaos_positive_plateau_all3']['result']}.")
    A(f"\n_Entropy:_ {taskA['brackets']['entropy_note']}\n")

    # ---- Task B ----
    A("\n## Task B — persistent Toda timescales (trailing 2e6 window)\n")
    A(f"Figures: `{taskB['figures']['T10'][0]}`, `{taskB['figures']['T90'][0]}` (+`.pdf`). "
      "Trailing (not centered) moving mean over 100 complete saved samples. Censored values "
      "are reported as `> last time`, marked as lower bounds, never substituted by 1e8, and "
      "excluded from any fit.\n")
    A("| ε_t | N | src | nrows | T10_first | T10_persistent | T90_first | T90_persistent |")
    A("|---|---|---|---|---|---|---|---|")
    for r in taskB["records"]:
        def cens(v, c):
            return (f"> {r['last_time']:.3e} (cens)" if c else _fmt(v, True))
        A(f"| {r['eps_target']:.0e} | {r['N']} | {'P+C' if r['source_type']=='pilot+continuation' else 'gap'} "
          f"| {r['nrows']} | {cens(r['T10_first'], r['T10_first_censored'])} "
          f"| {cens(r['T10_persistent'], r['T10_persistent_censored'])} "
          f"| {cens(r['T90_first'], r['T90_first_censored'])} "
          f"| {cens(r['T90_persistent'], r['T90_persistent_censored'])} |")
    A(f"\nLowest sampled ε with persistent T10 reached by 1e8, per N: "
      f"{taskB['lowest_eps_T10_persistent_by_N']} (consistent across N: "
      f"{taskB['T10_bracket_consistent_across_N']}). Persistent T90: "
      f"{taskB['lowest_eps_T90_persistent_by_N']} (consistent: "
      f"{taskB['T90_bracket_consistent_across_N']}).\n")

    # ---- Task C ----
    A("\n## Task C — low-ε deep-time block behavior\n")
    A(f"Figure: `{taskC['figure'][0]}` (+`.pdf`). Block rates of `S(t)=t·λ_FTLE(t)` over "
      "`[1e7,3e7],[3e7,6e7],[6e7,1e8]` (and `[1e8,1.4e8]` for the extended ε=1e-4 N=2048).\n")
    A("| point | ε_act | block | λ_block | mean J/2ε | mean Φ_J | Δ(J/2ε) | ⟨λ_loc⟩ | σ(λ_loc) | class |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in taskC["records"]:
        for k, b in r["blocks"].items():
            A(f"| {r['label']} | {_fmt(r['eps_actual'], True)} | {k} "
              f"| {_fmt(b['lambda_block'], True)} | {_fmt(b['block_mean_J_over_2eps'])} "
              f"| {_fmt(b['block_mean_PhiJ'])} | {_fmt(b['start_to_end_change_J_over_2eps'], True)} "
              f"| {_fmt(b['LyapLocal_block_mean'], True)} | {_fmt(b['LyapLocal_block_std'], True)} "
              f"| {r['classification']} |")
    if taskC["notes"]:
        A("\nTask C notes: " + "; ".join(taskC["notes"]) + "\n")

    # ---- Task D ----
    A("\n## Task D — dt / Lyapunov-cadence robustness (ε=1e-4, N=512)\n")
    cfg = taskD["config_validation"]
    A(f"Figure: `{taskD['figure'][0]}` (+`.pdf`).\n")
    A(f"- Config: N/model/α/Shape-IC/diagnostics/LyapSeed({cfg['lyap_seed']}) all match = "
      f"{cfg['N_match'] and cfg['model_match'] and cfg['alpha_match'] and cfg['shape_ic_match'] and cfg['diagnostics_match'] and cfg['lyap_seed_match']}.")
    A(f"- ε_actual: dt=0.1={_fmt(cfg['eps_actual_dt01'], True)}, "
      f"dt=0.05/rn200={_fmt(cfg['eps_actual_dt05_rn200'], True)}, "
      f"dt=0.05/rn100={_fmt(cfg['eps_actual_dt05_rn100'], True)} (each on its OWN ε).")
    A(f"- Physical renorm interval: dt=0.1→{_fmt(cfg['renorm_interval_dt01'])}, "
      f"dt=0.05/rn200→{_fmt(cfg['renorm_interval_dt05_rn200'])} "
      f"(**cadence-matched = {cfg['cadence_matched_A_vs_B']}**), "
      f"dt=0.05/rn100→{_fmt(cfg['renorm_interval_dt05_rn100'])} "
      f"(**matched = {cfg['cadence_matched_A_vs_L']}**).")
    m = taskD["cadence_matched_3e7"]; lng = taskD["long_nonmatched_1e8"]
    A("\n**Cadence-matched `[1e7,3e7]`** (λ_loc IS comparable here):\n")
    A("| quantity | dt=0.10 rn100 | dt=0.05 rn200 |")
    A("|---|---|---|")
    for q, a1, a2 in [
        ("J/2ε block mean", m["dt0.1_renorm100"]["J_over_2eps_mean"], m["dt0.05_renorm200"]["J_over_2eps_mean"]),
        ("Φ_J block mean", m["dt0.1_renorm100"]["PhiJ_mean"], m["dt0.05_renorm200"]["PhiJ_mean"]),
        ("Φ_J start→end", m["dt0.1_renorm100"]["PhiJ_progress_start_to_end"], m["dt0.05_renorm200"]["PhiJ_progress_start_to_end"]),
        ("Eta block mean", m["dt0.1_renorm100"]["Eta_mean"], m["dt0.05_renorm200"]["Eta_mean"]),
        ("λ_block (S slope)", m["dt0.1_renorm100"]["lambda_block"], m["dt0.05_renorm200"]["lambda_block"]),
        ("⟨λ_loc⟩", m["dt0.1_renorm100"]["LyapLocal_mean"], m["dt0.05_renorm200"]["LyapLocal_mean"]),
        ("σ(λ_loc)", m["dt0.1_renorm100"]["LyapLocal_std"], m["dt0.05_renorm200"]["LyapLocal_std"])]:
        A(f"| {q} | {_fmt(a1, True)} | {_fmt(a2, True)} |")
    A("\n**Long non-matched tail `[8e7,1e8)`** (dt=0.05 rn100; λ_loc EXCLUDED, FTLE caveated):\n")
    A("| quantity | dt=0.10 rn100 | dt=0.05 rn100 |")
    A("|---|---|---|")
    for q, k in [("J/2ε tail mean", "J_tail_mean"), ("Φ_J tail mean", "PhiJ_tail_mean"),
                 ("Eta tail mean", "Eta_tail_mean"), ("FTLE tail mean", "FTLE_tail_mean"),
                 ("max|ΔE|/E0", "max_abs_rel_energy_drift")]:
        A(f"| {q} | {_fmt(lng['dt0.1_renorm100'][k], True)} | {_fmt(lng['dt0.05_renorm100'][k], True)} |")
    A(f"\nEnergy-drift ratio (dt=0.1)/(dt=0.05) = "
      f"**{_fmt(lng['energy_drift_ratio_dt01_over_dt005'])}** "
      f"(4th-order expectation ≈16; ratio/16 = {_fmt(lng['drift_ratio_vs_16_4thorder_expectation'])}; "
      "an expectation, not a pass/fail).\n")

    # ---- duplicate audit ----
    A("\n## 4. Duplicate-source reproducibility audit\n")
    if dup_rows:
        A("| ε_t | N | J_tail reldiff | Φ_J reldiff | FTLE reldiff | λ_loc reldiff | η reldiff |")
        A("|---|---|---|---|---|---|---|")
        for r in dup_rows:
            A(f"| {r['eps_target']:.0e} | {r['N']} | {_fmt(r['J_tail_reldiff'], True)} "
              f"| {_fmt(r['PhiJ_tail_reldiff'], True)} | {_fmt(r['FTLE_tail_reldiff'], True)} "
              f"| {_fmt(r['LyapLocal_tail_reldiff'], True)} | {_fmt(r['Eta_tail_reldiff'], True)} |")
    else:
        A("**Empty by construction.** No `(N, ε)` cell has BOTH a pilot+continuation-to-1e8 "
          "canonical source AND an independent gap-from-zero-to-1e8 source: pilot+continuation "
          "covers ε∈{1e-4 (all N), 8e-4 (512,2048)}; the gap runs cover ε∈{6e-5, 2e-4, 4e-4, "
          "6e-4 (N=1024 only), 8e-4 (N=1024 only)} — the two sets are disjoint. No "
          "cross-machine reproducibility comparison is therefore possible on this grid, and "
          "none is fabricated. (The ε=8e-4 N=1024 pilot is 1e7-only, so it cannot supply an "
          "8e7–1e8 tail duplicate either.)\n")

    # ---- anomalies ----
    A("\n## 5. Anomalies / missing cells / censoring / ambiguity\n")
    for n in notes:
        A(f"- {n}")

    text = "\n".join(L) + "\n"
    p = out_dir / "alpha_grid_v1_report.md"
    p.write_text(text)
    return str(p)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m analysis.alpha_grid_v1")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--output-dir", default="figures/alpha_grid_v1")
    args = ap.parse_args(argv)

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[alpha_grid_v1] data_root={data_root}  output_dir={out_dir}")

    print("  Resolving sources (metadata-driven, frozen precedence) ...")
    cells, source_rows, seam_rows = resolve_sources(data_root)
    nok = sum(1 for r in source_rows if r["status"] == "ok")
    print(f"    constructable cells: {nok}/18")

    grid_recs = master_grid_stats(cells)
    dup_rows = duplicate_audit(cells)

    print("  Task A ...")
    taskA = run_task_A(grid_recs, out_dir)
    print("  Task B ...")
    taskB = run_task_B(cells, out_dir)
    print("  Task C ...")
    taskC = run_task_C(cells, data_root, out_dir)
    print("  Task D ...")
    taskD = run_task_D(cells, data_root, out_dir)

    # anomalies / notes
    notes = []
    for r in source_rows:
        if r["status"] != "ok":
            notes.append(f"cell {r['eps_target']:.0e}/N{r['N']}: {r['status']} — {r['notes']}")
    for r in taskB["records"]:
        if r["T90_persistent_censored"]:
            notes.append(f"T90_persistent right-censored (> {r['last_time']:.3e}) at "
                         f"{r['eps_target']:.0e}/N{r['N']}")
    for n in taskC["notes"]:
        notes.append("Task C: " + n)
    if not dup_rows:
        notes.append("Duplicate-source audit is empty: canonical (pilot+cont) and gap grids "
                     "are disjoint, so no independent cross-source reproducibility check exists.")
    if not notes:
        notes.append("None beyond the finite-time caveats in the discipline note.")

    tables = write_tables(out_dir, source_rows, seam_rows, grid_recs, dup_rows,
                          taskB, taskC, taskD)
    report_p = write_report(out_dir, source_rows, seam_rows, grid_recs, dup_rows,
                            taskA, taskB, taskC, taskD, notes)

    payload = {
        "meta": {
            "data_root": str(data_root), "expect_commit": EXPECT_COMMIT,
            "save_interval": SAVE_INTERVAL, "tail_window": [TAIL_LO, TAIL_HI],
            "smoothing_window": SMOOTH_WINDOW, "T10_level": T10_LEVEL, "T90_level": T90_LEVEL,
            "N_order": N_ORDER, "eps_order": EPS_ORDER,
            "frozen_criteria": {
                "toda_underway_PhiJ": FROZEN_TODA_ONSET_PHIJ,
                "toda_near_eq_J_over_2eps": FROZEN_TODA_NEAR_EQ,
                "ftle_flat_rtol": FROZEN_FTLE_FLAT_RTOL},
        },
        "source_resolution": source_rows,
        "seam_validation": seam_rows,
        "master_grid": grid_recs,
        "duplicate_audit": dup_rows,
        "taskA": taskA, "taskB": taskB, "taskC": taskC, "taskD": taskD,
        "anomalies": notes,
    }
    json_p = out_dir / "alpha_grid_v1_scalars.json"
    json_p.write_text(json.dumps(payload, indent=2, default=_json_default))

    print("\n  Outputs:")
    figs = (taskA["figure"] + taskB["figures"]["T10"] + taskB["figures"]["T90"]
            + taskC["figure"] + taskD["figure"])
    for f in figs:
        print(f"    {f}")
    for k, v in tables.items():
        print(f"    {v}")
    print(f"    {json_p}")
    print(f"    {report_p}")
    print("\n  Anomalies:")
    for n in notes:
        print(f"    - {n}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
