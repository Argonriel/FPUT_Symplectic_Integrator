"""Canonical CLI pipeline for the existing FPUT-beta Yoshida4 runs.

Usage::

    python -m analysis.summarize_beta summarize-beta \
        --input-dir data/yoshida_threshold_v2 \
        --output-dir results

Recursively discovers raw CSVs, validates each one (recording a reason for every
rejection), computes per-run quantities, resolves duplicates, writes the tabular
reports, computes descriptive collapse metrics, and renders the figures. Raw CSV
files are only ever read, never modified.

Outputs (under ``--output-dir``)::

    beta_runs.csv               one row per accepted, selected beta run
    beta_rejected_files.csv     every skipped file + reason
    beta_duplicates.csv         duplicate groups + selection
    beta_data_quality.csv       per-run quality flags
    beta_collapse_metrics.csv   per-grid-point across-N spread
    figures/                    fig1..fig6 as PNG and PDF
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import collapse as collapse_mod
from analysis import duplicates as dup_mod
from analysis import plotting
from analysis.metadata import relpath
from analysis.statistics import compute_run_stats, sha256_file
from analysis.validation import (
    Candidate,
    Rejection,
    iter_candidate_files,
    validate_file,
)

# Default CLI values (all overridable).
DEFAULT_INPUT_DIR = "data/yoshida_threshold_v2"
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_TAIL_FRACTION = 0.20
DEFAULT_MIN_SAVED_TIME = 1e8
DEFAULT_ENERGY_DRIFT_WARNING = 1e-4  # Yoshida-4 drift ~3e-6; this flags true anomalies
DEFAULT_ANALYTIC_MISMATCH_WARNING = 1e-2
DEFAULT_GRID_POINTS = 50


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Repository root (parent of the ``analysis`` package)."""
    return Path(__file__).resolve().parent.parent


def _rel(path: str) -> str:
    """Repo-relative path for machine-independent output; falls back to as-is."""
    try:
        return relpath(path, _repo_root())
    except Exception:  # noqa: BLE001
        return str(path)


def _build_run_row(cand: Candidate, tail_fraction: float) -> dict:
    """Compute the full per-run record for one accepted candidate."""
    meta = cand.metadata
    stats = compute_run_stats(
        cand.df,
        n=meta.n,
        beta=meta.beta,
        amplitude=meta.amplitude,
        nominal_duration=meta.nominal_duration,
        metadata_last_saved_time=meta.metadata_last_saved_time,
        tail_fraction=tail_fraction,
    )
    row = {
        "source_path": _rel(cand.path),
        "sha256": sha256_file(cand.path),
        "integrator": meta.integrator,
        "model": meta.model,
        "N": meta.n,
        "beta": meta.beta,
        "amplitude": meta.amplitude,
        "dt": meta.dt,
        "stride": meta.stride,
        "NumSegments": meta.num_segments,
    }
    row.update(asdict(stats))
    row.pop("extra", None)  # internal, not a column
    return row


def _quality_row(row: dict, *, energy_drift_warning: float, analytic_mismatch_warning: float) -> dict:
    """Derive data-quality flags for one per-run record."""
    max_drift = row["max_abs_rel_energy_error"]
    analytic_disc = row["rel_analytic_energy_discrepancy"]
    # A "complete" run reaches within one stride of its metadata end time.
    incomplete = row["last_saved_time"] < (row["metadata_last_saved_time"] - row["dt"] * row["stride"] * 0.5)
    return {
        "source_path": row["source_path"],
        "N": row["N"],
        "amplitude": row["amplitude"],
        "epsilon": row["epsilon"],
        "excessive_energy_drift": bool(np.isfinite(max_drift) and max_drift > energy_drift_warning),
        "max_abs_rel_energy_error": max_drift,
        "incomplete_run": bool(incomplete),
        "last_saved_time": row["last_saved_time"],
        "metadata_last_saved_time": row["metadata_last_saved_time"],
        "analytic_energy_mismatch": bool(
            np.isfinite(analytic_disc) and abs(analytic_disc) > analytic_mismatch_warning
        ),
        "rel_analytic_energy_discrepancy": analytic_disc,
        "missing_entropy": bool(not np.isfinite(row["eta_tail_mean"])),
        "nonfinite_values": bool(row["has_nonfinite"]),
        "suspicious_metadata": bool(
            row["N"] <= 1 or (row["beta"] is not None and row["beta"] <= 0)
            or (row["amplitude"] is not None and row["amplitude"] <= 0)
        ),
    }


def _duplicate_report_rows(groups: list[dup_mod.DuplicateGroup]) -> list[dict]:
    """Flatten duplicate groups (only groups with >1 member) into report rows."""
    rows: list[dict] = []
    for g in groups:
        if not g.is_duplicate:
            continue
        model, n, beta, amp, dt, stride, nseg = g.key
        for m in g.members:
            rows.append({
                "model": model, "N": n, "beta": beta, "amplitude": amp,
                "dt": dt, "stride": stride, "NumSegments": nseg,
                "source_path": _rel(m.path),
                "sha256": m.sha256,
                "last_saved_time": m.last_saved_time,
                "byte_identical_group": g.byte_identical,
                "conflict": g.conflict,
                "selected": m.path == g.selected_path,
                "selected_path": _rel(g.selected_path),
            })
    return rows


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

def run_summarize_beta(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"ERROR: input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    print(f"Scanning {input_dir} (exclude_legacy={not args.include_legacy}) ...")
    candidates: list[Candidate] = []
    rejections: list[Rejection] = []
    for path in iter_candidate_files(input_dir, exclude_legacy=not args.include_legacy):
        cand, rej = validate_file(
            path,
            min_saved_time=args.min_saved_time,
            exclude_legacy=not args.include_legacy,
        )
        if cand is not None:
            candidates.append(cand)
        else:
            rejections.append(rej)

    print(f"  accepted {len(candidates)} candidate beta run(s); rejected {len(rejections)} file(s).")

    # --- per-run statistics -------------------------------------------------
    run_rows = [_build_run_row(c, args.tail_fraction) for c in candidates]

    # --- duplicate resolution ----------------------------------------------
    refs = [
        dup_mod.RunRef(
            path=next(c.path for c in candidates if _rel(c.path) == r["source_path"]),
            sha256=r["sha256"], model=r["model"], n=r["N"], beta=r["beta"],
            amplitude=r["amplitude"], dt=r["dt"], stride=r["stride"],
            num_segments=r["NumSegments"], last_saved_time=r["last_saved_time"],
        )
        for r in run_rows
    ]
    groups = dup_mod.resolve_duplicates(refs)
    selected = dup_mod.selected_paths(groups)
    conflict_paths = {
        _rel(m.path) for g in groups if g.conflict for m in g.members
    }

    # Keep one selected row per physical key for aggregate outputs/plots.
    runs_df = pd.DataFrame(run_rows)
    if not runs_df.empty:
        runs_df["_abs_path"] = [c.path for c in candidates]
        runs_df["selected_for_plot"] = runs_df["_abs_path"].isin(selected)
        runs_df["duplicate_conflict"] = runs_df["source_path"].isin(conflict_paths)
        runs_df = runs_df.drop(columns=["_abs_path"])
        runs_df = runs_df.sort_values(["N", "epsilon", "source_path"]).reset_index(drop=True)

    # --- write tabular outputs ---------------------------------------------
    runs_path = output_dir / "beta_runs.csv"
    runs_df.to_csv(runs_path, index=False)
    print(f"  wrote {runs_path}  ({len(runs_df)} rows)")

    rej_df = pd.DataFrame(
        [{"source_path": _rel(r.path), "reason": r.reason, "model": r.model,
          "N": r.n, "last_saved_time": r.last_saved_time} for r in rejections]
    ).sort_values("source_path") if rejections else pd.DataFrame(
        columns=["source_path", "reason", "model", "N", "last_saved_time"]
    )
    rej_path = output_dir / "beta_rejected_files.csv"
    rej_df.to_csv(rej_path, index=False)
    print(f"  wrote {rej_path}  ({len(rej_df)} rows)")

    dup_rows = _duplicate_report_rows(groups)
    dup_df = pd.DataFrame(dup_rows) if dup_rows else pd.DataFrame(
        columns=["model", "N", "beta", "amplitude", "dt", "stride", "NumSegments",
                 "source_path", "sha256", "last_saved_time", "byte_identical_group",
                 "conflict", "selected", "selected_path"]
    )
    dup_path = output_dir / "beta_duplicates.csv"
    dup_df.to_csv(dup_path, index=False)
    n_dup_groups = sum(1 for g in groups if g.is_duplicate)
    print(f"  wrote {dup_path}  ({n_dup_groups} duplicate group(s), {len(dup_df)} member rows)")

    quality_rows = [
        _quality_row(r, energy_drift_warning=args.energy_drift_warning,
                     analytic_mismatch_warning=args.analytic_mismatch_warning)
        for r in run_rows
    ]
    quality_df = pd.DataFrame(quality_rows) if quality_rows else pd.DataFrame()
    if not quality_df.empty:
        quality_df = quality_df.sort_values(["N", "epsilon", "source_path"]).reset_index(drop=True)
    quality_path = output_dir / "beta_data_quality.csv"
    quality_df.to_csv(quality_path, index=False)
    print(f"  wrote {quality_path}  ({len(quality_df)} rows)")

    # --- collapse metrics ---------------------------------------------------
    plot_df = runs_df[runs_df["selected_for_plot"] & ~runs_df["duplicate_conflict"]].copy() \
        if not runs_df.empty else runs_df
    curves = []
    if not plot_df.empty:
        for n in sorted(plot_df["N"].unique()):
            sub = plot_df[plot_df["N"] == n]
            curves.append(collapse_mod.build_curve(
                int(n), sub["epsilon"].to_numpy(), sub["eta_tail_mean"].to_numpy()
            ))
    collapse_result = collapse_mod.compute_collapse(curves, grid_points=args.grid_points)
    collapse_path = output_dir / "beta_collapse_metrics.csv"
    collapse_result.metrics.to_csv(collapse_path, index=False)
    print(f"  wrote {collapse_path}  ({len(collapse_result.metrics)} grid rows)")
    print(
        f"  collapse: N={collapse_result.n_values}  "
        f"common eps=[{collapse_result.common_lo:.3e}, {collapse_result.common_hi:.3e}]  "
        f"RMS spread={collapse_result.rms_spread:.3e}  ({collapse_result.note})"
    )

    # --- figures ------------------------------------------------------------
    if not args.no_plots:
        written = plotting.make_all_figures(plot_df, curves, collapse_result, fig_dir)
        print(f"  wrote {len(written)} figure file(s) to {fig_dir}")
    else:
        print("  skipping figures (--no-plots)")

    print("Done.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.summarize_beta",
        description="Reproducible analysis pipeline for existing FPUT-beta Yoshida4 runs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("summarize-beta", help="Summarize FPUT-beta trajectories and make figures.")
    p.add_argument("--input-dir", default=DEFAULT_INPUT_DIR,
                   help=f"Directory of raw CSVs, searched recursively (default: {DEFAULT_INPUT_DIR}).")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help=f"Directory for reports and figures (default: {DEFAULT_OUTPUT_DIR}).")
    p.add_argument("--tail-fraction", type=float, default=DEFAULT_TAIL_FRACTION,
                   help=f"Fraction of trailing samples for tail statistics (default: {DEFAULT_TAIL_FRACTION}).")
    p.add_argument("--min-saved-time", type=float, default=DEFAULT_MIN_SAVED_TIME,
                   help=f"Reject runs whose last saved time is below this (default: {DEFAULT_MIN_SAVED_TIME:g}).")
    p.add_argument("--energy-drift-warning", type=float, default=DEFAULT_ENERGY_DRIFT_WARNING,
                   help=f"Flag runs whose max relative energy error exceeds this (default: {DEFAULT_ENERGY_DRIFT_WARNING:g}).")
    p.add_argument("--analytic-mismatch-warning", type=float, default=DEFAULT_ANALYTIC_MISMATCH_WARNING,
                   help=f"Flag runs whose analytic H0 relative discrepancy exceeds this (default: {DEFAULT_ANALYTIC_MISMATCH_WARNING:g}).")
    p.add_argument("--grid-points", type=int, default=DEFAULT_GRID_POINTS,
                   help=f"Number of shared log-epsilon grid points for collapse (default: {DEFAULT_GRID_POINTS}).")
    p.add_argument("--include-legacy", action="store_true",
                   help="Do NOT exclude legacy paths (off by default; legacy is excluded).")
    p.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    p.set_defaults(func=run_summarize_beta)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
