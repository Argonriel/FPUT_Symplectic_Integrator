"""Accept or reject a candidate raw CSV as a scientific beta trajectory.

Validation is deliberately conservative and *records a reason* for every
rejection (written to ``beta_rejected_files.csv``) so that pointing the pipeline
at a wider directory never silently pulls in short, non-beta, or non-Yoshida
runs. Directory-name heuristics are never trusted on their own — every file is
checked by metadata and by its actual saved times.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from analysis.metadata import RunMetadata, parse_metadata_safe
from analysis.statistics import REQUIRED_COLUMNS

# Directory names that never contain raw input trajectories.
EXCLUDED_DIR_NAMES = frozenset({"summary", "results", "output", "outputs", "figures", "__pycache__"})

# Substrings marking legacy/older data trees excluded by default.
DEFAULT_LEGACY_MARKERS = ("legacy",)

EXPECTED_INTEGRATOR = "yoshida4"
EXPECTED_MODEL = "beta"


@dataclass
class Candidate:
    """A raw CSV that passed validation, with its parsed pieces attached."""

    path: str
    metadata: RunMetadata
    df: pd.DataFrame


@dataclass
class Rejection:
    """A raw CSV that failed validation, with a machine-readable reason."""

    path: str
    reason: str
    # Best-effort metadata fields for context (may be missing on parse failure).
    model: Optional[str] = None
    n: Optional[int] = None
    last_saved_time: Optional[float] = None


def iter_candidate_files(
    input_dir: str | os.PathLike,
    *,
    exclude_legacy: bool = True,
    legacy_markers: tuple[str, ...] = DEFAULT_LEGACY_MARKERS,
) -> Iterator[Path]:
    """Recursively yield ``*.csv`` paths under ``input_dir``, skipping non-input dirs.

    Skips any directory whose name is in :data:`EXCLUDED_DIR_NAMES`, and (when
    ``exclude_legacy``) any path containing a legacy marker segment. Results are
    sorted for deterministic ordering.
    """
    root = Path(input_dir)
    files: list[Path] = []
    for path in root.rglob("*.csv"):
        parts_lower = {p.lower() for p in path.parts}
        if parts_lower & EXCLUDED_DIR_NAMES:
            continue
        if exclude_legacy and any(
            marker in part.lower() for part in path.parts for marker in legacy_markers
        ):
            continue
        files.append(path)
    return iter(sorted(files, key=lambda p: p.as_posix()))


def validate_file(
    path: str | os.PathLike,
    *,
    min_saved_time: float,
    exclude_legacy: bool = True,
    legacy_markers: tuple[str, ...] = DEFAULT_LEGACY_MARKERS,
) -> tuple[Optional[Candidate], Optional[Rejection]]:
    """Validate a single CSV.

    Returns exactly one of ``(Candidate, None)`` or ``(None, Rejection)``. The
    checks, in order: legacy path, header parse, integrator, model, required
    ``Beta``/``Amplitude`` keys, readability, required columns, non-empty, and
    minimum actual last saved time.
    """
    p = Path(path)
    spath = p.as_posix()

    if exclude_legacy and any(
        marker in part.lower() for part in p.parts for marker in legacy_markers
    ):
        return None, Rejection(spath, "legacy path excluded by default")

    meta, meta_err = parse_metadata_safe(p)
    if meta is None:
        return None, Rejection(spath, f"metadata parse failed: {meta_err}")

    if meta.integrator.lower() != EXPECTED_INTEGRATOR:
        return None, Rejection(
            spath, f"non-Yoshida integrator: {meta.integrator!r}", model=meta.model, n=meta.n
        )

    if meta.model != EXPECTED_MODEL:
        return None, Rejection(
            spath, f"non-beta model: {meta.model!r}", model=meta.model, n=meta.n
        )

    if meta.beta is None:
        return None, Rejection(spath, "missing Beta in header", model=meta.model, n=meta.n)
    if meta.amplitude is None:
        return None, Rejection(spath, "missing Amplitude in header", model=meta.model, n=meta.n)
    if meta.n <= 1:
        return None, Rejection(spath, f"invalid N={meta.n} (need N>1)", model=meta.model, n=meta.n)

    try:
        df = pd.read_csv(p, comment="#")
        df.columns = df.columns.str.strip()
    except Exception as exc:  # noqa: BLE001 - report the concrete read failure
        return None, Rejection(spath, f"could not read CSV body: {exc}", model=meta.model, n=meta.n)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return None, Rejection(
            spath, f"missing required columns: {', '.join(missing)}", model=meta.model, n=meta.n
        )

    if len(df) == 0:
        return None, Rejection(spath, "empty trajectory (0 data rows)", model=meta.model, n=meta.n)

    last_saved_time = float(df["Time"].iloc[-1])
    if last_saved_time < min_saved_time:
        return None, Rejection(
            spath,
            f"short run: last_saved_time={last_saved_time:.3e} < min_saved_time={min_saved_time:.3e}",
            model=meta.model,
            n=meta.n,
            last_saved_time=last_saved_time,
        )

    return Candidate(path=spath, metadata=meta, df=df), None
