"""End-to-end tests for the summarize-beta CLI pipeline."""

from __future__ import annotations

import pandas as pd

from analysis.summarize_beta import build_parser, run_summarize_beta
from analysis.tests.conftest import write_run_csv


def _run(input_dir, output_dir, **overrides):
    parser = build_parser()
    argv = ["summarize-beta", "--input-dir", str(input_dir), "--output-dir", str(output_dir)]
    for k, v in overrides.items():
        argv += [f"--{k}", str(v)]
    args = parser.parse_args(argv)
    return run_summarize_beta(args)


def test_pipeline_accepts_beta_rejects_others(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    write_run_csv(data / "beta_N512.csv", model="beta", n=512, amplitude=8.0)
    write_run_csv(data / "beta_N1024.csv", model="beta", n=1024, amplitude=16.0)
    write_run_csv(data / "alpha_N512.csv", model="alpha", n=512, amplitude=8.0)
    write_run_csv(data / "short.csv", model="beta", stride=5000, num_segments=200)

    out = tmp_path / "results"
    rc = _run(data, out, **{"min-saved-time": 1e8})
    assert rc == 0

    runs = pd.read_csv(out / "beta_runs.csv")
    assert len(runs) == 2
    assert set(runs["N"]) == {512, 1024}
    # epsilon = H0/(N-1) present and positive
    assert (runs["epsilon"] > 0).all()

    rejected = pd.read_csv(out / "beta_rejected_files.csv")
    reasons = " ".join(rejected["reason"].tolist())
    assert "non-beta" in reasons
    assert "short run" in reasons

    for name in ("beta_duplicates.csv", "beta_data_quality.csv", "beta_collapse_metrics.csv"):
        assert (out / name).exists()


def test_pipeline_flags_energy_drift(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    write_run_csv(data / "clean.csv", n=512, amplitude=8.0, energy_drift=3e-6)
    write_run_csv(data / "drifty.csv", n=1024, amplitude=16.0, energy_drift=1e-2)

    out = tmp_path / "results"
    _run(data, out)

    quality = pd.read_csv(out / "beta_data_quality.csv").sort_values("N")
    flags = dict(zip(quality["N"], quality["excessive_energy_drift"]))
    assert flags[512] == False  # noqa: E712  (3e-6 < 1e-4 default)
    assert flags[1024] == True  # noqa: E712  (1e-2 > 1e-4 default)


def test_pipeline_collapse_over_two_sizes(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    # Two N with overlapping amplitude-driven epsilon ranges.
    for amp in (6.0, 8.0, 10.0, 12.0):
        write_run_csv(data / f"beta_N512_A{amp}.csv", n=512, amplitude=amp)
    for amp in (10.0, 12.0, 14.0, 16.0):
        write_run_csv(data / f"beta_N1024_A{amp}.csv", n=1024, amplitude=amp)

    out = tmp_path / "results"
    _run(data, out)  # plots on, to exercise the figure code
    metrics = pd.read_csv(out / "beta_collapse_metrics.csv")
    assert not metrics.empty
    assert (metrics["epsilon"] > 0).all()
    # figures written
    assert (out / "figures" / "fig1_entropy_tailmean_vs_epsilon.png").exists()
    assert (out / "figures" / "fig6_finite_size_collapse.pdf").exists()
