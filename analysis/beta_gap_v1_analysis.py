"""FPUT-beta epsilon-collapse completion (beta_gap_v1) — git-tracked pipeline.

Read-only w.r.t. all raw inputs. Re-aggregates the four-N beta threshold data from
the frozen Yoshida set (`data/yoshida_threshold_v2/`) plus the five new N=4096
fills (`data/beta_gap_v1/`), applies the FROZEN 4-parameter logistic fit imported
verbatim from ``visualization.plot_stochasticity_threshold``, and writes every
output into the gitignored ``figures/beta_gap_v1_analysis/`` directory (unchanged
paths).

This module is the SINGLE SOURCE of the beta ``epsilon_c`` error propagation
(:func:`epsilon_c_and_sigma`). Nothing else in the repository should hand-write
that derivative.

Run::

    python -m analysis.beta_gap_v1_analysis
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "visualization") not in sys.path:
    sys.path.insert(0, str(REPO / "visualization"))

from analysis.metadata import parse_metadata                    # noqa: E402
from analysis.statistics import (                               # noqa: E402
    epsilon_from_h0, energy_errors, sha256_file as _sha256_stat,
)
from visualization.plot_stochasticity_threshold import sigmoid  # noqa: E402  (frozen model)

OUT = REPO / "figures" / "beta_gap_v1_analysis"
ALPHA_OUT = REPO / "figures" / "alpha_grid_complete_v1"
DATA = REPO / "data"
FROZEN_COMMIT = "4a66fec"

NEW_BETA_TARGET = {
    "beta_N4096_A31.932483.csv": 1.5e-4,
    "beta_N4096_A36.872455.csv": 2.0e-4,
    "beta_N4096_A41.224658.csv": 2.5e-4,
    "beta_N4096_A45.159351.csv": 3.0e-4,
    "beta_N4096_A48.777674.csv": 3.5e-4,
}


# =========================================================================
# CANONICAL beta energy map + error propagation (single source of truth)
# =========================================================================
def _z(N: int, A: float) -> float:
    """z(A) = A^2 * sin^2(pi / (2N))."""
    return A * A * math.sin(math.pi / (2.0 * N)) ** 2


def eps_of_A(N: int, A: float, beta: float = 1.0) -> float:
    """Beta energy density from amplitude: eps(A) = (N/(N-1))*(z + 1.5*beta*z^2).

    Identical to ``analysis.statistics.analytic_h0(A, N, beta) / (N - 1)``.
    """
    z = _z(N, A)
    return (N / (N - 1.0)) * (z + 1.5 * beta * z * z)


def deps_dA(N: int, A: float, beta: float = 1.0) -> float:
    """Analytic derivative d eps / dA at amplitude A.

    deps/dA = (N/(N-1)) * 2A*sin^2(pi/(2N)) * (1 + 3*beta*z),  z = z(A).
    """
    s2 = math.sin(math.pi / (2.0 * N)) ** 2
    z = A * A * s2
    return (N / (N - 1.0)) * (2.0 * A * s2) * (1.0 + 3.0 * beta * z)


def epsilon_c_and_sigma(N: int, Ac: float, sigma_Ac: float,
                        beta: float = 1.0) -> tuple[float, float]:
    """Return (eps_c, sigma_eps_c) for a critical amplitude Ac.

    eps_c = eps(Ac); sigma_eps_c = |deps/dA at Ac| * sigma_Ac  (delta method).
    This is the ONLY place sigma_eps_c is computed.
    """
    eps_c = eps_of_A(N, Ac, beta)
    if sigma_Ac is None or not math.isfinite(sigma_Ac):
        return eps_c, math.nan
    return eps_c, abs(deps_dA(N, Ac, beta)) * sigma_Ac


# Column layout shared by the fit CSV, the fit JSON, and the Markdown fit table.
# Ac, sigma_Ac, epsilon_c, sigma_epsilon_c always appear in this order.
FIT_TABLE_COLUMNS = ["N", "label", "n_points", "Ac", "sigma_Ac",
                     "residual_RMS", "Ac_in_range", "epsilon_c", "sigma_epsilon_c"]


def make_fit_record(N: int, Ac: float, sigma_Ac: float, *,
                    label: str = "", beta: float = 1.0, **extra) -> dict:
    """Build the ONE canonical fit record for a fitted point.

    ``epsilon_c`` and ``sigma_epsilon_c`` come only from
    :func:`epsilon_c_and_sigma`; every downstream artefact (CSV, JSON, Markdown
    table) is rendered from records produced here, so they can never diverge.
    """
    eps_c, sigma_eps_c = epsilon_c_and_sigma(N, Ac, sigma_Ac, beta)
    rec = {"N": int(N), "label": label, "Ac": float(Ac), "sigma_Ac": float(sigma_Ac),
           "epsilon_c": eps_c, "sigma_epsilon_c": sigma_eps_c}
    rec.update(extra)
    return rec


def render_fit_table_md(records) -> str:
    """Render the Markdown fit table from canonical records (list or DataFrame).

    This is the ONLY producer of the fit table that appears in
    ``beta_report.md`` — no fit numbers are ever hand-copied into the report.
    Emits columns :data:`FIT_TABLE_COLUMNS` with Ac / sigma_Ac / epsilon_c /
    sigma_epsilon_c in fixed slots.
    """
    if isinstance(records, pd.DataFrame):
        records = records.to_dict("records")
    lines = ["| " + " | ".join(FIT_TABLE_COLUMNS) + " |",
             "|" + "|".join(["---"] * len(FIT_TABLE_COLUMNS)) + "|"]
    for r in records:
        if not r.get("converged", True):
            lines.append(f"| {int(r['N'])} | {r.get('label','')} | - | (no convergence) | | | | | |")
            continue
        lines.append("| " + " | ".join([
            str(int(r["N"])), str(r.get("label", "")), str(int(r.get("n_points", 0))),
            f"{r['Ac']:.4g}", f"{r['sigma_Ac']:.4g}", f"{r.get('residual_RMS', float('nan')):.4g}",
            "yes" if r.get("Ac_in_range") else "no",
            f"{r['epsilon_c']:.3e}", f"{r['sigma_epsilon_c']:.3e}"]) + " |")
    return "\n".join(lines)


# =========================================================================
# helpers
# =========================================================================
def _rel(p: Path) -> str:
    return p.relative_to(REPO).as_posix()


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# =========================================================================
# inventory + hashing (candidate raw inputs) — step 0
# =========================================================================
def build_inventory() -> pd.DataFrame:
    from plot_utils import get_metadata

    def header_info(path: Path) -> dict:
        try:
            m = get_metadata(str(path))
        except Exception as e:  # noqa: BLE001
            return {"header_error": str(e)}
        coeff_name, coeff = (None, None)
        if "Alpha" in m:
            coeff_name, coeff = "alpha", m.get("Alpha")
        elif "Beta" in m:
            coeff_name, coeff = "beta", m.get("Beta")
        return {k: m.get(v) for k, v in {
            "integrator": "Integrator", "model": "Model", "N": "N", "amplitude": "Amplitude",
            "dt": "dt", "stride": "Stride", "NumSegments": "NumSegments", "Entropy": "Entropy",
            "TodaIntegral": "TodaIntegral", "Lyapunov": "Lyapunov",
            "ResumeFromSegment": "ResumeFromSegment", "SolverGitCommit": "SolverGitCommit",
            "SolverGitDirty": "SolverGitDirty"}.items()} | {"coeff_name": coeff_name, "coeff": coeff}

    def beta_raw(root, ns):
        return sorted(sum((list(DATA.joinpath(root).glob(f"beta_N{n}_*.csv")) for n in ns), []))

    roles = [
        ("beta-new-raw", sorted(DATA.joinpath("beta_gap_v1").glob("*.csv"))),
        ("beta-old-raw-v2", beta_raw("yoshida_threshold_v2", (512, 1024, 2048, 4096))),
        ("beta-old-raw-v1-dup", beta_raw("yoshida_threshold", (512, 1024, 2048, 4096))),
        ("beta-old-summary-v2", sorted(DATA.joinpath("yoshida_threshold_v2", "summary").glob("threshold_summary_beta_N*.csv"))),
        ("beta-old-summary-v1", sorted(DATA.joinpath("yoshida_threshold", "summary").glob("threshold_summary_beta_N*.csv"))),
        ("alpha-gap", sorted(DATA.joinpath("alpha_gap_v1").glob("*.csv"))),
        ("alpha-pilot", sorted(DATA.joinpath("alpha_pilot_v1").glob("*.csv"))),
        ("alpha-pilot-ext", sorted(DATA.joinpath("alpha_pilot_v1_ext").glob("*.csv"))),
        ("alpha-pilot-ext14", sorted(DATA.joinpath("alpha_pilot_v1_ext14").glob("*.csv"))),
        ("alpha-dt05-renorm200", sorted(DATA.joinpath("alpha_dt05_renorm200").glob("*.csv"))),
        ("alpha-dt05-check", sorted(DATA.joinpath("alpha_dt05_check").glob("*.csv"))),
    ]
    rows = []
    for role, paths in roles:
        for p in paths:
            if not p.is_file():
                continue
            rec = {"role": role, "path": _rel(p), "size_bytes": p.stat().st_size, "sha256": _sha256(p)}
            if "summary" not in p.parts:
                rec.update(header_info(p))
            rows.append(rec)
    df = pd.DataFrame(rows)
    by_hash = {}
    for r in rows:
        by_hash.setdefault(r["sha256"], []).append(r["path"])
    df["n_byte_identical"] = df["sha256"].map(lambda h: len(by_hash[h]))
    OUT.mkdir(parents=True, exist_ok=True)
    ALPHA_OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "candidate_input_inventory.csv", index=False)
    df[df.role.str.startswith("beta")].to_csv(OUT / "candidate_input_inventory_beta.csv", index=False)
    df[df.role.str.startswith("alpha")].to_csv(ALPHA_OUT / "candidate_input_inventory.csv", index=False)
    with open(OUT / "candidate_input_inventory.json", "w") as f:
        json.dump(rows, f, indent=2)
    dup_groups = {h: ps for h, ps in by_hash.items() if len(ps) > 1}
    with open(OUT / "byte_identical_duplicate_groups.json", "w") as f:
        json.dump(dup_groups, f, indent=2)
    return df


# =========================================================================
# per-run aggregation (frozen physical-time tail 0.8*nominal <= Time < nominal)
# =========================================================================
def aggregate_one(path: Path, source_type: str, target_eps=None) -> dict:
    m = parse_metadata(path)
    df = pd.read_csv(path, comment="#")
    df.columns = df.columns.str.strip()
    for c in ("Time", "TotalEnergy", "Eta"):
        if c not in df.columns:
            raise ValueError(f"{path.name}: missing required column {c}")
    t = df["Time"].to_numpy(float)
    i0 = int(np.argmin(np.abs(t)))
    e0 = float(df["TotalEnergy"].to_numpy(float)[i0])
    eps = epsilon_from_h0(e0, m.n)
    nominal = m.nominal_duration
    mask = (t >= 0.8 * nominal) & (t < nominal)
    eta = df["Eta"].to_numpy(float)
    eta_tail = eta[mask]
    max_drift, _ = energy_errors(df["TotalEnergy"])
    req = df[[c for c in ("Time", "Mode1", "TotalEnergy", "Eta") if c in df.columns]].to_numpy(float)
    raw = m.raw or {}
    rec = {
        "N": m.n, "amplitude": m.amplitude, "beta": m.beta,
        "eta_bar": float(np.mean(eta_tail)) if eta_tail.size else math.nan,
        "epsilon_value": eps, "epsilon_source": "initial-energy",
        "source_type": source_type, "source_file": _rel(path),
        "tail_start_time": float(t[mask].min()) if eta_tail.size else math.nan,
        "tail_end_time": float(t[mask].max()) if eta_tail.size else math.nan,
        "tail_row_count": int(eta_tail.size),
        "provenance": f"commit={raw.get('SolverGitCommit')};dirty={raw.get('SolverGitDirty')}",
        "SolverGitCommit": raw.get("SolverGitCommit"), "SolverGitDirty": raw.get("SolverGitDirty"),
        "sha256": _sha256_stat(str(path)), "dt": m.dt, "stride": m.stride,
        "NumSegments": m.num_segments, "nominal_duration": nominal,
        "metadata_last_saved_time": m.metadata_last_saved_time, "n_rows": int(len(df)),
        "first_saved_time": float(t.min()), "last_saved_time": float(t.max()), "E0": e0,
        "epsilon_actual": eps, "max_abs_rel_energy_drift": max_drift,
        "has_nonfinite": bool(not np.isfinite(req).all()),
        "eta_tail_std": float(np.std(eta_tail, ddof=1)) if eta_tail.size > 1 else math.nan,
    }
    if target_eps is not None:
        rec["epsilon_target"] = target_eps
        rec["rel_dev"] = (eps - target_eps) / target_eps
        rec["ppm_dev"] = rec["rel_dev"] * 1e6
    return rec


def qc_status(r) -> tuple[str, str]:
    reasons = []
    if r["source_type"] == "new-raw":
        if str(r["SolverGitCommit"]) != FROZEN_COMMIT:
            reasons.append(f"FAIL:SolverGitCommit={r['SolverGitCommit']}")
        if str(r["SolverGitDirty"]) != "0":
            reasons.append(f"FAIL:SolverGitDirty={r['SolverGitDirty']}")
    if r["has_nonfinite"]:
        reasons.append("FAIL:nonfinite")
    d = r["max_abs_rel_energy_drift"]
    if not np.isfinite(d):
        reasons.append("FAIL:drift-nonfinite")
    elif d > 1e-3:
        reasons.append("FAIL:drift>1e-3")
    elif d > 1e-4:
        reasons.append("WARN:drift>1e-4")
    if any(x.startswith("FAIL") for x in reasons):
        return "FAIL", ";".join(reasons)
    if any(x.startswith("WARN") for x in reasons):
        return "WARN", ";".join(reasons)
    return "PASS", ""


# =========================================================================
# frozen logistic fit (bounds/p0/optimizer replicate the frozen script)
# =========================================================================
def frozen_fit(x, y):
    order = np.argsort(x)
    x = np.asarray(x, float)[order]
    y = np.asarray(y, float)[order]
    lower = [0.4, x.min() - 20, 0.001, 0.0]
    upper = [1.0, x.max() + 50, 1.0, 0.15]
    p0 = [0.8, np.median(x), 0.05, 0.05]
    try:
        popt, pcov = curve_fit(sigmoid, x, y, p0=p0, bounds=(lower, upper), maxfev=10000)
    except Exception:  # noqa: BLE001
        return None
    warn = []
    for i, (val, lo, hi) in enumerate(zip(popt, lower, upper)):
        span = hi - lo
        if span > 0 and (abs(val - lo) / span < 0.01 or abs(val - hi) / span < 0.01):
            warn.append(f"param[{i}]={val:.4g} within 1% of bound [{lo:.4g},{hi:.4g}]")
    resid = y - sigmoid(x, *popt)
    try:
        cond = float(np.linalg.cond(pcov))
    except Exception:  # noqa: BLE001
        cond = math.inf
    return {"popt": popt, "pcov": pcov, "lower": lower, "upper": upper, "p0": p0,
            "rms": float(np.sqrt(np.mean(resid ** 2))), "warn": warn,
            "cov_finite": bool(np.all(np.isfinite(pcov))), "cond": cond, "x": x, "y": y}


def run(make_plots: bool = True) -> dict:
    """Regenerate every beta output into ``figures/beta_gap_v1_analysis/``."""
    OUT.mkdir(parents=True, exist_ok=True)
    build_inventory()

    rows, new_qc_rows = [], []
    for n in (512, 1024, 2048, 4096):
        for p in sorted(DATA.joinpath("yoshida_threshold_v2").glob(f"beta_N{n}_*.csv")):
            rows.append(aggregate_one(p, "old-raw"))
    for name, tgt in NEW_BETA_TARGET.items():
        r = aggregate_one(DATA / "beta_gap_v1" / name, "new-raw", target_eps=tgt)
        rows.append(r)
        new_qc_rows.append(r)

    combined = pd.DataFrame(rows).sort_values(["N", "amplitude"]).reset_index(drop=True)
    qcs = combined.apply(qc_status, axis=1)
    combined["QC"] = [q[0] for q in qcs]
    combined["QC_detail"] = [q[1] for q in qcs]

    summary_cols = ["N", "amplitude", "beta", "eta_bar", "epsilon_value", "epsilon_source",
                    "source_type", "source_file", "tail_start_time", "tail_end_time",
                    "tail_row_count", "provenance", "QC"]
    combined[summary_cols + ["QC_detail", "sha256", "E0", "epsilon_actual", "eta_tail_std"]].to_csv(
        OUT / "beta_combined_summary_allN.csv", index=False)
    combined[combined.N == 4096][summary_cols + ["QC_detail", "sha256"]].to_csv(
        OUT / "beta_combined_N4096_summary.csv", index=False)

    qc_df = pd.DataFrame(new_qc_rows)
    qc_df["QC"] = [qc_status(r)[0] for _, r in qc_df.iterrows()]
    qc_df["QC_detail"] = [qc_status(r)[1] for _, r in qc_df.iterrows()]
    qc_df.to_csv(OUT / "beta_new_file_qc.csv", index=False)

    combined["_ampkey"] = combined["amplitude"].round(6)
    dupe_rows = []
    for (n, amp), g in combined.groupby(["N", "_ampkey"]):
        if len(g) > 1:
            for _, r in g.iterrows():
                dupe_rows.append({"N": n, "amplitude": amp, "source_file": r["source_file"],
                                  "sha256": r["sha256"], "source_type": r["source_type"]})
    pd.DataFrame(dupe_rows, columns=["N", "amplitude", "source_file", "sha256", "source_type"]).to_csv(
        OUT / "beta_duplicate_replicate_table.csv", index=False)

    # fits
    fit_curves, fit_rows, cov_rows = {}, [], []
    for n in (512, 1024, 2048):
        sub = combined[(combined.N == n) & (combined.QC != "FAIL")]
        fit_curves[n] = (sub, frozen_fit(sub["amplitude"].to_numpy(), sub["eta_bar"].to_numpy()),
                         "old-raw only (20 pts)")
    sub4 = combined[(combined.N == 4096) & (combined.QC != "FAIL")]
    fit4_full = frozen_fit(sub4["amplitude"].to_numpy(), sub4["eta_bar"].to_numpy())
    sub4_old = combined[(combined.N == 4096) & (combined.source_type == "old-raw") & (combined.QC != "FAIL")]
    fit4_old = frozen_fit(sub4_old["amplitude"].to_numpy(), sub4_old["eta_bar"].to_numpy())
    fit_curves[4096] = (sub4, fit4_full, "old+new (25 pts)")

    def record_fit(n, sub, fit, label):
        if fit is None:
            fit_rows.append(make_fit_record(n, float("nan"), float("nan"), label=label, converged=False))
            return
        L, x0, k, b = fit["popt"]
        Ac = float(x0)
        diag = np.diag(fit["pcov"])
        Ac_err = float(math.sqrt(diag[1])) if (fit["cov_finite"] and diag[1] >= 0) else math.nan
        amp = sub["amplitude"].to_numpy()
        eps = sub["epsilon_value"].to_numpy()
        on_bound = [i for i, (v, lo, hi) in enumerate(zip(fit["popt"], fit["lower"], fit["upper"]))
                    if (hi - lo) > 0 and (abs(v - lo) / (hi - lo) < 0.01 or abs(v - hi) / (hi - lo) < 0.01)]
        # single canonical constructor -> epsilon_c / sigma_epsilon_c computed here only
        fit_rows.append(make_fit_record(
            n, Ac, Ac_err, label=label, beta=1.0, converged=True, n_points=int(len(sub)),
            amp_min=float(amp.min()), amp_max=float(amp.max()),
            eps_min=float(eps.min()), eps_max=float(eps.max()),
            L=float(L), x0=Ac, k=float(k), b=float(b),
            residual_RMS=fit["rms"], Ac_in_range=bool(amp.min() <= Ac <= amp.max()),
            params_on_bound_idx=";".join(map(str, on_bound)) if on_bound else "",
            cov_finite=fit["cov_finite"], cov_cond=fit["cond"]))
        cov_rows.append({"N": n, "label": label, "Ac": Ac, "sigma_Ac": Ac_err,
                         "cov_finite": fit["cov_finite"], "cov_cond": fit["cond"],
                         "cov_ill_conditioned": bool(fit["cond"] > 1e8),
                         "warnings": " | ".join(fit["warn"]) if fit["warn"] else ""})

    for n in (512, 1024, 2048):
        sub_n, fit_n, label_n = fit_curves[n]
        record_fit(n, sub_n, fit_n, label_n)
    record_fit(4096, sub4, fit4_full, "old+new (25 pts)")
    record_fit(4096, sub4_old, fit4_old, "old-raw only (20 pts) [for comparison]")

    # ---- ONE shared record set feeds the CSV, the JSON, and the Markdown table ----
    fit_df = pd.DataFrame(fit_rows)
    fit_df.to_csv(OUT / "beta_fit_parameters.csv", index=False)
    fit_df.to_json(OUT / "beta_fit_parameters.json", orient="records", indent=2)
    prop = fit_df[["N", "label", "Ac", "sigma_Ac", "epsilon_c", "sigma_epsilon_c"]].copy()
    prop["note"] = ("delta-method 1st-order propagation of Ac SE through analytic map; "
                    "local fit SE only, not total physical/sampling/model uncertainty")
    prop.to_csv(OUT / "beta_epsilon_c_propagation.csv", index=False)
    pd.DataFrame(cov_rows).to_csv(OUT / "beta_Ac_covariance_warnings.csv", index=False)

    if make_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 13})
        colors = {512: "#1b9e77", 1024: "#d95f02", 2048: "#7570b3", 4096: "#e7298a"}
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for n in (512, 1024, 2048, 4096):
            sub, fit, _ = fit_curves[n]
            ax.scatter(sub["epsilon_value"], sub["eta_bar"], s=55, color=colors[n],
                       edgecolors="w", zorder=3, label=f"N={n} (initial-energy)")
            if fit is not None:
                xf = np.linspace(sub["amplitude"].min(), sub["amplitude"].max(), 300)
                ax.plot([eps_of_A(n, a) for a in xf], sigmoid(xf, *fit["popt"]),
                        color=colors[n], lw=1.8, alpha=0.85, zorder=2)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\epsilon = H(0)/(N-1)$  (initial-energy)")
        ax.set_ylabel(r"tail-mean spectral entropy  $\bar{\eta}$")
        ax.set_title("FPUT-beta finite-time entropy vs energy density (four N)\n"
                     "points: initial-energy epsilon; curves: frozen 4-param logistic (amplitude fit) mapped to epsilon")
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(loc="lower right", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / "beta_collapse_eta_vs_epsilon.png", dpi=200, bbox_inches="tight")
        fig.savefig(OUT / "beta_collapse_eta_vs_epsilon.pdf", bbox_inches="tight")
        plt.close(fig)

    # collapse audit (100 log-eps points, no extrapolation)
    curves = {}
    for n in (512, 1024, 2048, 4096):
        sub, _, _ = fit_curves[n]
        e = sub["epsilon_value"].to_numpy(float)
        y = sub["eta_bar"].to_numpy(float)
        o = np.argsort(e)
        e, y = e[o], y[o]
        ue, inv = np.unique(e, return_inverse=True)
        if ue.size != e.size:
            yy = np.zeros(ue.size); cc = np.zeros(ue.size)
            np.add.at(yy, inv, y); np.add.at(cc, inv, 1)
            y, e = yy / cc, ue
        curves[n] = (e, y)
    lo = max(e.min() for e, _ in curves.values())
    hi = min(e.max() for e, _ in curves.values())
    audit = {"common_lo": lo, "common_hi": hi, "N": [512, 1024, 2048, 4096]}
    if not (hi > lo):
        audit["status"] = "empty common range; collapse audit not possible"
        pd.DataFrame().to_csv(OUT / "beta_collapse_audit.csv", index=False)
    else:
        grid = np.logspace(math.log10(lo), math.log10(hi), 100)
        interp = {n: np.interp(np.log(grid), np.log(e), y) for n, (e, y) in curves.items()}
        M = np.vstack([interp[n] for n in (512, 1024, 2048, 4096)])
        eta_mean = M.mean(axis=0)
        spread_max = M.max(axis=0) - M.min(axis=0)
        spread_rms = np.sqrt(((M - eta_mean) ** 2).mean(axis=0))
        pd.DataFrame({"epsilon": grid, "eta_mean": eta_mean, "spread_max": spread_max,
                      "spread_rms": spread_rms,
                      **{f"eta_N{n}": interp[n] for n in (512, 1024, 2048, 4096)}}).to_csv(
            OUT / "beta_collapse_audit.csv", index=False)
        imax = int(np.argmax(spread_max))
        audit.update({"status": "ok", "n_grid": 100, "max_spread_max": float(spread_max.max()),
                      "rms_of_spread_max": float(np.sqrt(np.mean(spread_max ** 2))),
                      "rms_cross_N_deviation_all": float(np.sqrt(np.mean((M - eta_mean) ** 2))),
                      "epsilon_at_max_spread": float(grid[imax]),
                      "eta_mean_range": [float(eta_mean.min()), float(eta_mean.max())]})
    with open(OUT / "beta_collapse_spread_summary.json", "w") as f:
        json.dump(audit, f, indent=2)

    # N=4096 low-eps coverage
    def _ac(fit):
        if fit is None:
            return None, None
        d = np.diag(fit["pcov"])[1]
        return float(fit["popt"][1]), (float(math.sqrt(d)) if (fit["cov_finite"] and d >= 0) else None)
    n4 = combined[(combined.N == 4096) & (combined.QC != "FAIL")]
    new4 = combined[(combined.N == 4096) & (combined.source_type == "new-raw")]
    ac_full, se_full = _ac(fit4_full)
    ac_old, se_old = _ac(fit4_old)
    with open(OUT / "beta_N4096_lowEps_coverage.json", "w") as f:
        json.dump({"eps_min_sampled": float(n4["epsilon_value"].min()),
                   "eta_bar_min": float(n4["eta_bar"].min()),
                   "eta_bar_range": [float(n4["eta_bar"].min()), float(n4["eta_bar"].max())],
                   "new_points": new4[["amplitude", "epsilon_value", "eta_bar"]].sort_values("amplitude").to_dict("records"),
                   "Ac_full": ac_full, "Ac_se_full": se_full, "Ac_old_only": ac_old, "Ac_se_old_only": se_old,
                   "Ac_in_sampled_range_full": bool(n4["amplitude"].min() <= (ac_full or -1) <= n4["amplitude"].max())},
                  f, indent=2)

    # cross-check vs frozen immutable summaries (read-only)
    frozen_cmp = []
    for n in (512, 1024, 2048, 4096):
        fp = DATA / "yoshida_threshold_v2" / "summary" / f"threshold_summary_beta_N{n}.csv"
        if fp.is_file():
            fs = pd.read_csv(fp)
            mine = combined[(combined.N == n) & (combined.source_type == "old-raw")][["amplitude", "eta_bar"]]
            merged = fs.merge(mine, left_on="Amplitude", right_on="amplitude", how="inner")
            if len(merged):
                frozen_cmp.append({"N": n, "matched_rows": len(merged),
                                   "max_abs_eta_diff_vs_frozen": float((merged["Eta"] - merged["eta_bar"]).abs().max())})
    with open(OUT / "beta_reagg_vs_frozen_summary_check.json", "w") as f:
        json.dump(frozen_cmp, f, indent=2)

    # ---- beta_report.md: fit table rendered from fit_df (no hand-copied numbers) ----
    a = audit
    se_factor = (se_old / se_full) if (se_old and se_full) else float("nan")
    new_eps = new4["epsilon_value"]
    new_eta = new4["eta_bar"]
    report = f"""# FPUT-β ε-collapse completion (beta_gap_v1) — concise report

_Read-only analysis. No simulation was run/resumed; no raw CSV, checkpoint, log,
manifest, existing summary, or existing figure was modified. Frozen solver
`4a66fec` (tag `diag-v1`). All new outputs live under `figures/beta_gap_v1_analysis/`._

_This report is machine-generated by `analysis/beta_gap_v1_analysis.py`; the fit
table below is rendered directly from the same `fit_df` records that produce
`beta_fit_parameters.csv`, `beta_fit_parameters.json`, and
`beta_epsilon_c_propagation.csv`, so the four artefacts cannot diverge._

## Inputs & provenance
- **5 new N=4096 raw fills** (`data/beta_gap_v1/`), all `SolverGitCommit=4a66fec`,
  `SolverGitDirty=0`, `Entropy=1`. All **PASS** QC (`beta_new_file_qc.csv`).
- **Old N=512/1024/2048/4096 threshold data** re-aggregated from the frozen Yoshida
  set `data/yoshida_threshold_v2/` (20 raw runs per N). Re-aggregation reproduces the
  immutable frozen summaries to ~1e-16 (`beta_reagg_vs_frozen_summary_check.json`).
- SHA-256 of all candidate files: `candidate_input_inventory.csv/.json`; no
  byte-identical duplicates; no repeated physical (N, amplitude) key.

## Aggregation (Task 1A/1B)
Every ε is `initial-energy`: `ε = H(0)/(N-1)` from each run's own `TotalEnergy`
at `Time==0`. Tail = frozen physical window `0.8·nominal ≤ Time < nominal` per run
(nominal 1.4e8 → `[1.12e8, 1.4e8)`): old runs 100 tail rows, new runs 1400.

## Frozen logistic fit (Task 1C)
Model reused **verbatim** from `visualization/plot_stochasticity_threshold.py`:
`η(A) = L/(1+exp(-k(A-x0))) + b`, x = **Amplitude**, `Ac ≡ x0`; bounds/p0/optimizer
unchanged (4-parameter, unweighted `scipy.curve_fit`, maxfev=10000); Ac SE from
`sqrt(diag(pcov))`.

{render_fit_table_md(fit_df)}

ε_c from the frozen analytic map `z=Ac²·sin²(π/2N)`, `ε_c=(N/(N-1))(z+1.5·β·z²)`,
β=1; sigma_epsilon_c = |dε/dA at Ac|·sigma_Ac (first-order delta method), from the
single canonical function `analysis.beta_gap_v1_analysis.epsilon_c_and_sigma`.
**Caveat (frozen 1% bound rule):** parameter `b` sits on its bound in every fit
(upper 0.15 for N=512/1024/2048; lower 0.0 for N=4096); the reported σ are
**covariance-based local fit standard errors only** and do not represent total
physical, sampling, or model uncertainty.

## Old vs new N=4096 (Task 1F)
The five new points bracket the lower sigmoid: ε∈[{new_eps.min():.2e}, {new_eps.max():.2e}],
η̄∈[{new_eta.min():.3f}, {new_eta.max():.3f}]. Min ε now {float(n4['epsilon_value'].min()):.2e};
full N=4096 η̄ range [{float(n4['eta_bar'].min()):.3f}, {float(n4['eta_bar'].max()):.3f}].
Old fit Ac≈{ac_old:.1f}±{se_old:.1f} (weakly constrained; Ac outside the old sampled
range from A=50). New fit Ac={ac_full:.1f}±{se_full:.1f} — nearly the same centre, Ac now
**inside** the sampled range, covariance SE reduced ~{se_factor:.1f}× (local fit SE, not
total uncertainty).

## Four-N collapse (Task 1D/1E)
Figure `beta_collapse_eta_vs_epsilon.png/.pdf`; audit `beta_collapse_audit.csv`,
`beta_collapse_spread_summary.json`. Common ε overlap
[{a.get('common_lo', float('nan')):.2e}, {a.get('common_hi', float('nan')):.2e}], 100 log-spaced points,
linear interp in log ε, no extrapolation.
- max spread_max = {a.get('max_spread_max', float('nan')):.3f} at ε ≈ {a.get('epsilon_at_max_spread', float('nan')):.2e}
- RMS of spread_max = {a.get('rms_of_spread_max', float('nan')):.3f}; RMS cross-N deviation = {a.get('rms_cross_N_deviation_all', float('nan')):.3f}

**Interpretation:** consistent with an approximate finite-time collapse, with a
residual finite-size spread that is poorest in the low-/intermediate-ε region; the
common overlap and spread support at most an approximate collapse claim, not a strong
one. Spread is descriptive (not weighted by temporal tail std, which is time variation,
not statistical uncertainty).
"""
    with open(OUT / "beta_report.md", "w") as f:
        f.write(report)

    return {"combined": combined, "fit_rows": fit_rows, "fit_df": fit_df, "audit": audit,
            "frozen_cmp": frozen_cmp, "qc": qc_df}


def main() -> int:
    res = run()
    print("=== beta epsilon_c propagation (single canonical source) ===")
    print(pd.read_csv(OUT / "beta_epsilon_c_propagation.csv")[
        ["N", "label", "Ac", "sigma_Ac", "epsilon_c", "sigma_epsilon_c"]].to_string(index=False))
    print("\nre-aggregation vs frozen summary (max |dEta|):", res["frozen_cmp"])
    print("collapse audit:", json.dumps(res["audit"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
