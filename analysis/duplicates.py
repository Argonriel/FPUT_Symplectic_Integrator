"""Detect and deterministically resolve duplicate beta runs.

Two runs are considered duplicates when they share the full physical key::

    (model, N, beta, amplitude, dt, stride, num_segments)

Duplicates are never silently discarded. For every duplicate group a report row
is produced listing all paths, their hashes, whether they are byte-identical,
and which one (if any) is selected for aggregate plotting.

Selection rule (deterministic, documented):

* A single-member group selects its only member.
* If all members are byte-identical (same SHA-256): select the
  lexicographically-first path. ``conflict`` is ``False``.
* If members differ (conflicting content for the same physical key): select the
  member with the largest ``last_saved_time`` (longest completed run), breaking
  ties lexicographically by path, and set ``conflict`` to ``True`` so the
  conflict is visible in the data-quality report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# The physical identity of a run. Amplitude and beta are rounded to a stable
# number of decimals so float formatting differences never split a group.
_AMP_DECIMALS = 6
_BETA_DECIMALS = 6


@dataclass
class RunRef:
    """Minimal handle onto an accepted run, used for duplicate resolution."""

    path: str
    sha256: str
    model: str
    n: int
    beta: float
    amplitude: float
    dt: float
    stride: int
    num_segments: int
    last_saved_time: float


def dedup_key(ref: RunRef) -> tuple:
    """Physical key used to group duplicates."""
    return (
        ref.model,
        ref.n,
        round(ref.beta, _BETA_DECIMALS),
        round(ref.amplitude, _AMP_DECIMALS),
        ref.dt,
        ref.stride,
        ref.num_segments,
    )


@dataclass
class DuplicateGroup:
    """One physical key and every run that maps to it."""

    key: tuple
    members: list[RunRef]
    selected_path: str
    byte_identical: bool
    conflict: bool

    @property
    def is_duplicate(self) -> bool:
        return len(self.members) > 1


def resolve_duplicates(refs: Sequence[RunRef]) -> list[DuplicateGroup]:
    """Group ``refs`` by physical key and apply the documented selection rule.

    Returns one :class:`DuplicateGroup` per distinct key, in deterministic key
    order. Every group has a non-empty ``selected_path``.
    """
    groups: dict[tuple, list[RunRef]] = {}
    for ref in refs:
        groups.setdefault(dedup_key(ref), []).append(ref)

    resolved: list[DuplicateGroup] = []
    for key in sorted(groups.keys(), key=lambda k: tuple(str(x) for x in k)):
        members = sorted(groups[key], key=lambda r: r.path)
        hashes = {m.sha256 for m in members}
        byte_identical = len(hashes) == 1
        conflict = not byte_identical and len(members) > 1

        if conflict:
            # Longest run wins; tie-break lexicographically by path.
            selected = sorted(
                members, key=lambda r: (-r.last_saved_time, r.path)
            )[0]
        else:
            selected = members[0]  # already sorted by path

        resolved.append(
            DuplicateGroup(
                key=key,
                members=members,
                selected_path=selected.path,
                byte_identical=byte_identical,
                conflict=conflict,
            )
        )
    return resolved


def selected_paths(groups: Sequence[DuplicateGroup]) -> set[str]:
    """Set of paths chosen as the representative for each physical key."""
    return {g.selected_path for g in groups}
