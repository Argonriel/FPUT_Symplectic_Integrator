"""Tests for duplicate detection and deterministic selection."""

from __future__ import annotations

from analysis.duplicates import RunRef, resolve_duplicates, selected_paths


def _ref(path, sha, last_time=1.4e8, amp=8.0):
    return RunRef(path=path, sha256=sha, model="beta", n=512, beta=1.0,
                  amplitude=amp, dt=0.1, stride=2_800_000, num_segments=500,
                  last_saved_time=last_time)


def test_no_duplicates_all_selected():
    refs = [_ref("a.csv", "h1", amp=8.0), _ref("b.csv", "h2", amp=9.0)]
    groups = resolve_duplicates(refs)
    assert all(not g.is_duplicate for g in groups)
    assert selected_paths(groups) == {"a.csv", "b.csv"}


def test_byte_identical_duplicates_pick_first_lexicographic():
    refs = [_ref("z.csv", "same"), _ref("a.csv", "same")]
    groups = resolve_duplicates(refs)
    assert len(groups) == 1
    g = groups[0]
    assert g.is_duplicate and g.byte_identical and not g.conflict
    assert g.selected_path == "a.csv"


def test_conflicting_duplicates_pick_longest_run():
    refs = [
        _ref("short.csv", "h1", last_time=1.0e8),
        _ref("long.csv", "h2", last_time=1.4e8),
    ]
    groups = resolve_duplicates(refs)
    g = groups[0]
    assert g.is_duplicate and not g.byte_identical and g.conflict
    assert g.selected_path == "long.csv"


def test_key_includes_stride_and_numsegments():
    # Same (model,N,beta,amplitude,dt) but different stride/num_segments are NOT
    # duplicates — this is exactly the v2-vs-old confusable case.
    a = _ref("v2.csv", "h1")
    b = RunRef(path="old.csv", sha256="h2", model="beta", n=512, beta=1.0,
               amplitude=8.0, dt=0.1, stride=5000, num_segments=200,
               last_saved_time=1.0e5)
    groups = resolve_duplicates([a, b])
    assert len(groups) == 2
    assert all(not g.is_duplicate for g in groups)
