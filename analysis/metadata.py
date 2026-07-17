"""Typed parsing of the commented CSV header.

The raw solver CSVs begin with a block of ``# key: value`` comment lines, e.g.::

    # Integrator: Yoshida4
    # Model: beta
    # N: 1024
    # Beta: 1
    # Amplitude: 8
    # dt: 0.1
    # Stride: 2800000
    # NumSegments: 500
    # Shape: 0
    # Entropy: 1

The low-level parsing already exists in ``visualization/plot_utils.py`` as
``get_metadata`` and is reused here verbatim; this module only adds typed
access, validation of required keys, and derived timing quantities.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ``get_metadata`` lives in ``visualization/`` — add it to the path the same way
# ``simulations_cpu/yoshida/aggregate_eta.py`` already does, so there is exactly
# one metadata parser in the repository.
_VIS_DIR = os.path.join(os.path.dirname(__file__), "..", "visualization")
if _VIS_DIR not in sys.path:
    sys.path.insert(0, _VIS_DIR)

from plot_utils import get_metadata  # noqa: E402  (path set up above)

# Keys required for a run to be interpretable at all. Missing any of these makes
# the file impossible to place in the (model, N, beta, amplitude, ...) space.
REQUIRED_KEYS = ("Integrator", "Model", "N", "dt", "Stride", "NumSegments")


class MetadataError(ValueError):
    """Raised when the header is missing a required key or has an unparsable value."""


@dataclass(frozen=True)
class RunMetadata:
    """Typed view of a raw CSV header.

    Attributes mirror the header keys. ``beta`` and ``amplitude`` are optional at
    the metadata level (a non-beta run legitimately has no ``Beta`` key); the
    beta-specific pipeline enforces their presence downstream in validation.
    """

    integrator: str
    model: str
    n: int
    dt: float
    stride: int
    num_segments: int
    beta: Optional[float] = None
    amplitude: Optional[float] = None
    shape: Optional[int] = None
    entropy: Optional[int] = None
    raw: Optional[dict] = None

    @property
    def nominal_duration(self) -> float:
        """Nominal simulated duration from metadata: ``NumSegments * Stride * dt``.

        This is *not* generally equal to the last saved time; the solver writes a
        snapshot before each advance, so both are exported separately downstream.
        """
        return self.num_segments * self.stride * self.dt

    @property
    def metadata_last_saved_time(self) -> float:
        """Last saved time implied by metadata: ``(NumSegments - 1) * Stride * dt``.

        Cross-checked against the actual maximum ``Time`` value read from the CSV.
        """
        return (self.num_segments - 1) * self.stride * self.dt


def _require(meta: dict, key: str) -> str:
    if key not in meta:
        raise MetadataError(f"missing required metadata key {key!r}")
    return meta[key]


def _as_int(value: str, key: str) -> int:
    try:
        # Tolerate integers written as floats, e.g. "1024.0".
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise MetadataError(f"could not parse {key!r}={value!r} as int") from exc


def _as_float(value: str, key: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise MetadataError(f"could not parse {key!r}={value!r} as float") from exc


def _opt_float(meta: dict, key: str) -> Optional[float]:
    return _as_float(meta[key], key) if key in meta else None


def _opt_int(meta: dict, key: str) -> Optional[int]:
    return _as_int(meta[key], key) if key in meta else None


def parse_metadata(path: str | os.PathLike) -> RunMetadata:
    """Parse and type the header of a raw CSV.

    Raises
    ------
    MetadataError
        If a required key is missing or a value cannot be parsed.
    """
    meta = get_metadata(os.fspath(path))
    for key in REQUIRED_KEYS:
        _require(meta, key)

    return RunMetadata(
        integrator=meta["Integrator"].strip(),
        model=meta["Model"].strip().lower(),
        n=_as_int(meta["N"], "N"),
        dt=_as_float(meta["dt"], "dt"),
        stride=_as_int(meta["Stride"], "Stride"),
        num_segments=_as_int(meta["NumSegments"], "NumSegments"),
        beta=_opt_float(meta, "Beta"),
        amplitude=_opt_float(meta, "Amplitude"),
        shape=_opt_int(meta, "Shape"),
        entropy=_opt_int(meta, "Entropy"),
        raw=dict(meta),
    )


def parse_metadata_safe(path: str | os.PathLike) -> tuple[Optional[RunMetadata], Optional[str]]:
    """Parse metadata, returning ``(metadata, None)`` or ``(None, reason)``.

    Convenience wrapper so callers can record a rejection reason instead of
    handling exceptions.
    """
    try:
        return parse_metadata(path), None
    except MetadataError as exc:
        return None, str(exc)
    except OSError as exc:  # unreadable file
        return None, f"could not read header: {exc}"


def relpath(path: str | os.PathLike, start: str | os.PathLike | None = None) -> str:
    """Repo-relative, forward-slash path for deterministic, machine-independent output."""
    p = Path(path)
    if start is not None:
        try:
            p = p.relative_to(Path(start))
        except ValueError:
            p = Path(os.path.relpath(path, start))
    return p.as_posix()
