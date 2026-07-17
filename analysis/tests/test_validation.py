"""Tests for trajectory validation and file discovery."""

from __future__ import annotations

from analysis.validation import iter_candidate_files, validate_file

MIN_TIME = 1e8


def test_accepts_canonical_beta_run(make_csv):
    path = make_csv("beta.csv", model="beta", integrator="Yoshida4",
                    stride=2_800_000, num_segments=500)
    cand, rej = validate_file(path, min_saved_time=MIN_TIME)
    assert rej is None
    assert cand is not None
    assert cand.metadata.model == "beta"


def test_rejects_non_beta_run(make_csv):
    path = make_csv("alpha.csv", model="alpha")
    cand, rej = validate_file(path, min_saved_time=MIN_TIME)
    assert cand is None
    assert "non-beta" in rej.reason


def test_rejects_non_yoshida_run(make_csv):
    path = make_csv("euler.csv", integrator="Euler")
    cand, rej = validate_file(path, min_saved_time=MIN_TIME)
    assert cand is None
    assert "non-Yoshida" in rej.reason


def test_rejects_short_run(make_csv):
    # stride*dt*num_segments small -> last saved time well below 1e8
    path = make_csv("short.csv", stride=5000, num_segments=200)
    cand, rej = validate_file(path, min_saved_time=MIN_TIME)
    assert cand is None
    assert "short run" in rej.reason


def test_rejects_missing_required_column(tmp_path):
    p = tmp_path / "nocols.csv"
    p.write_text(
        "# Integrator: Yoshida4\n# Model: beta\n# N: 512\n# Beta: 1\n"
        "# Amplitude: 8\n# dt: 0.1\n# Stride: 2800000\n# NumSegments: 500\n"
        "Time,Mode1,TotalEnergy\n" + "\n".join(f"{i*2.8e5:.6e},0.1,1.0" for i in range(600)) + "\n"
    )
    cand, rej = validate_file(p, min_saved_time=MIN_TIME)
    assert cand is None
    assert "missing required columns" in rej.reason and "Eta" in rej.reason


def test_legacy_path_excluded(tmp_path):
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    from analysis.tests.conftest import write_run_csv
    p = write_run_csv(legacy_dir / "beta.csv")
    cand, rej = validate_file(p, min_saved_time=MIN_TIME, exclude_legacy=True)
    assert cand is None
    assert "legacy" in rej.reason


def test_iter_skips_summary_and_legacy_dirs(tmp_path):
    from analysis.tests.conftest import write_run_csv
    (tmp_path / "summary").mkdir()
    (tmp_path / "legacy").mkdir()
    write_run_csv(tmp_path / "keep.csv")
    write_run_csv(tmp_path / "summary" / "threshold_summary.csv")
    write_run_csv(tmp_path / "legacy" / "old.csv")
    found = [p.name for p in iter_candidate_files(tmp_path)]
    assert found == ["keep.csv"]
