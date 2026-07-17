"""Tests for header metadata parsing."""

from __future__ import annotations

import pytest

from analysis.metadata import MetadataError, parse_metadata, parse_metadata_safe


def test_parse_metadata_from_commented_header(make_csv):
    path = make_csv("beta_run.csv", n=1024, beta=1.0, amplitude=8.0,
                    dt=0.1, stride=2_800_000, num_segments=500)
    meta = parse_metadata(path)
    assert meta.integrator == "Yoshida4"
    assert meta.model == "beta"
    assert meta.n == 1024
    assert meta.beta == 1.0
    assert meta.amplitude == 8.0
    assert meta.dt == 0.1
    assert meta.stride == 2_800_000
    assert meta.num_segments == 500


def test_derived_timing_uses_metadata_formulas(make_csv):
    path = make_csv("beta_run.csv", dt=0.1, stride=2_800_000, num_segments=500)
    meta = parse_metadata(path)
    # nominal_duration = NumSegments * Stride * dt ; last = (NumSegments-1)*Stride*dt
    assert meta.nominal_duration == pytest.approx(500 * 2_800_000 * 0.1)
    assert meta.metadata_last_saved_time == pytest.approx(499 * 2_800_000 * 0.1)


def test_model_is_lowercased(make_csv):
    path = make_csv("run.csv", model="Beta")
    assert parse_metadata(path).model == "beta"


def test_missing_required_key_raises(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("# Model: beta\nTime,Eta\n0,0\n")  # no Integrator/N/dt/...
    with pytest.raises(MetadataError):
        parse_metadata(p)


def test_parse_metadata_safe_returns_reason(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("# Model: beta\nTime,Eta\n0,0\n")
    meta, reason = parse_metadata_safe(p)
    assert meta is None
    assert reason and "missing required metadata key" in reason
