"""FPUT-alpha master-grid completion (alpha_gap_v1) — git-tracked companion.

Read-only w.r.t. all raw inputs. IMPORTS the frozen pipeline
``analysis.alpha_grid_v1`` (source resolution, tail stats, block_rate,
_block_toda_loc, _classify_finite_time, plateau legs) and reuses it verbatim — no
classification rule is duplicated or changed here. It adds only the reporting
artefacts requested for the completion task: the 18-cell completion matrix, the
new-file QC table, a consolidated per-cell summary, block rates for the 6e-4 cells
(via the frozen block method), the combined top-level QC of the eight NEW files,
and machine-readable answers to the three specific alpha questions.

``__main__`` first regenerates the frozen Tasks A-D outputs (by calling
``analysis.alpha_grid_v1.main``) and then this companion, all into the gitignored
``figures/alpha_grid_complete_v1/`` directory (unchanged paths).

Run::

    python -m analysis.alpha_grid_complete_v1
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "visualization") not in sys.path:
    sys.path.insert(0, str(REPO / "visualization"))

import analysis.alpha_grid_v1 as ag                     # noqa: E402  (frozen, imported not forked)
from plot_utils import get_metadata                     # noqa: E402

OUT = REPO / "figures" / "alpha_grid_complete_v1"
BETA_OUT = REPO / "figures" / "beta_gap_v1_analysis"
DATA = REPO / "data"
FROZEN_COMMIT = "4a66fec"
ALPHA_REQUIRED = ["Time", "TotalEnergy", "TodaJ", "Eta", "LyapunovFTLE", "LyapunovLocal"]
N_ORDER = ag.N_ORDER
EPS_ORDER = ag.EPS_ORDER

EXPECTED_NEW_ALPHA = [
    ("data/alpha_gap_v1/alpha_N512_eps6e-4_dt0.1_nom1e8.csv", 512, 6e-4),
    ("data/alpha_gap_v1/alpha_N2048_eps6e-4_dt0.1_nom1e8.csv", 2048, 6e-4),
    ("data/alpha_gap_v1/alpha_N1024_eps6e-5_dt0.1_nom1e8.csv", 1024, 6e-5),
]
BASE_BLOCKS = [(1e7, 3e7), (3e7, 6e7), (6e7, 1e8)]


def _sha256(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def leg_key(eps_t, N) -> str:
    return f"eps{eps_t:.0e}_N{N}"      # frozen scalars.json format, e.g. eps6e-04_N512


def qc_alpha_file(rel_path, N, eps_target) -> dict:
    p = REPO / rel_path
    rec = {"path": rel_path, "N_expected": N, "epsilon_target": eps_target}
    if not p.is_file():
        rec.update({"present": False, "QC": "MISSING",
                    "exclusion_reason": "file not present on disk (cell could not be filled)"})
        return rec
    rec["present"] = True
    rec["sha256"] = _sha256(p)
    m = get_metadata(str(p))
    df = pd.read_csv(p, comment="#")
    df.columns = df.columns.str.strip()
    t = df["Time"].to_numpy(float)
    i0 = int(np.argmin(np.abs(t)))
    E0 = float(df["TotalEnergy"].to_numpy(float)[i0])
    Nhdr = int(float(m["N"]))
    eps_act = E0 / (Nhdr - 1)
    dt = float(m["dt"]); stride = int(float(m["Stride"])); nseg = int(float(m["NumSegments"]))
    e = df["TotalEnergy"].to_numpy(float)
    drift = float(np.max(np.abs(e - e[0]) / abs(e[0]))) if e[0] != 0 else math.nan
    missing_cols = [c for c in ALPHA_REQUIRED if c not in df.columns]
    nonfinite = bool(not np.isfinite(df[[c for c in ALPHA_REQUIRED if c in df.columns]].to_numpy(float)).all())
    commit, dirty = m.get("SolverGitCommit"), m.get("SolverGitDirty")
    reasons = []
    if commit != FROZEN_COMMIT:
        reasons.append(f"FAIL:commit={commit}")
    if str(dirty) != "0":
        reasons.append(f"FAIL:dirty={dirty}")
    if missing_cols:
        reasons.append("FAIL:missing_cols=" + ",".join(missing_cols))
    if nonfinite:
        reasons.append("FAIL:nonfinite")
    if not np.isfinite(drift) or drift > 1e-3:
        reasons.append("FAIL:drift")
    elif drift > 1e-4:
        reasons.append("WARN:drift")
    qc = "FAIL" if any(x.startswith("FAIL") for x in reasons) else (
        "WARN" if any(x.startswith("WARN") for x in reasons) else "PASS")
    rec.update({"model": m.get("Model"), "N": Nhdr, "coeff_name": "alpha", "coeff": m.get("Alpha"),
                "amplitude": m.get("Amplitude"), "dt": dt, "stride": stride, "NumSegments": nseg,
                "row_count": int(len(df)), "first_saved_time": float(t.min()),
                "last_saved_time": float(t.max()), "nominal_duration": nseg * stride * dt,
                "metadata_last_saved_time": (nseg - 1) * stride * dt, "E0": E0, "epsilon_actual": eps_act,
                "rel_dev": (eps_act - eps_target) / eps_target,
                "ppm_dev": (eps_act - eps_target) / eps_target * 1e6,
                "max_abs_rel_energy_drift": drift, "nonfinite": nonfinite,
                "SolverGitCommit": commit, "SolverGitDirty": dirty, "QC": qc,
                "QC_detail": ";".join(reasons), "exclusion_reason": "" if qc != "FAIL" else ";".join(reasons)})
    return rec


def block_table_for(cells, N, eps_t):
    """Frozen block-rate method (reuses ag helpers) for one cell."""
    traj = cells[(N, eps_t)]["canonical"]
    if traj is None:
        return None
    df = traj.df
    t = df["Time"].to_numpy()
    S = ag._s_of(df)
    eps = traj.eps_actual
    J = df["TodaJ"].to_numpy()
    phi, _ = ag.phi_j(J, eps)
    jr = J / (2.0 * eps)
    loc = df["LyapunovLocal"].to_numpy()
    blk = {}
    for a, b in BASE_BLOCKS:
        blk[ag._blabel(a, b)] = {**ag.block_rate(t, S, a, b), **ag._block_toda_loc(t, jr, phi, loc, a, b)}
    rate_vals = [blk[k]["lambda_block"] for k in blk]
    last = list(blk.keys())[-1]
    cls = ag._classify_finite_time(rate_vals, blk[last]["LyapLocal_block_mean"], blk[last]["LyapLocal_block_std"])
    return {"N": N, "eps_target": eps_t, "eps_actual": eps, "blocks": blk,
            "block_rate_vals": rate_vals, "classification": cls}


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    cells, source_rows, seam_rows = ag.resolve_sources(DATA)

    # completion matrix
    matrix = pd.DataFrame(index=[f"N={n}" for n in N_ORDER], columns=[f"{e:g}" for e in EPS_ORDER])
    for e in EPS_ORDER:
        for n in N_ORDER:
            matrix.loc[f"N={n}", f"{e:g}"] = cells[(n, e)]["status"]
    matrix.to_csv(OUT / "completion_matrix_18cell.csv")
    n_ok = sum(1 for k in cells if cells[k]["status"] == "ok")

    # new-file QC
    alpha_qc = [qc_alpha_file(*x) for x in EXPECTED_NEW_ALPHA]
    pd.DataFrame(alpha_qc).to_csv(OUT / "alpha_new_file_qc.csv", index=False)

    # consolidated per-cell summary (frozen master + taskB + plateau legs)
    scal = json.load(open(OUT / "alpha_grid_v1_scalars.json"))
    master = pd.read_csv(OUT / "grid_master_statistics.csv")
    taskb = pd.read_csv(OUT / "taskB_T10_T90_timescales.csv")
    legs = scal["taskA"]["plateau_legs"]
    tb_idx = {(r["eps_target"], r["N"]): r for _, r in taskb.iterrows()}

    def frozen_class(row):
        if row["status"] != "ok":
            return "missing/failed"
        lg = legs.get(leg_key(row["eps_target"], int(row["N"])), {})
        chaos = lg.get("positive_plateau_all3", False)
        underway = row["PhiJ_tail_mean"] >= ag.FROZEN_TODA_ONSET_PHIJ
        near_eq = row["J_tail_mean_over_2eps"] <= ag.FROZEN_TODA_NEAR_EQ
        return " | ".join([
            "chaos:positive-plateau(all 3 legs)" if chaos else "chaos:no positive plateau",
            "Toda:underway(Phi_J>=0.5)" if underway else "Toda:onset not reached(Phi_J<0.5)",
            "Toda:near-eq(J/2eps<=1.10)" if near_eq else "Toda:not near 2eps estimate"])

    rows = []
    for _, r in master.iterrows():
        d = dict(r)
        lg = legs.get(leg_key(r["eps_target"], int(r["N"])), {}) if r["status"] == "ok" else {}
        tb = tb_idx.get((r["eps_target"], r["N"]), {})
        d.update({"leg_i_ftle_flat": lg.get("ftle_flat"), "leg_ii_loc_positive": lg.get("loc_positive"),
                  "leg_iii_loc_above_noise": lg.get("loc_above_noise"),
                  "chaos_positive_plateau_all3": lg.get("positive_plateau_all3"),
                  "T10_persistent": tb.get("T10_persistent"), "T10_persistent_censored": tb.get("T10_persistent_censored"),
                  "T90_persistent": tb.get("T90_persistent"), "T90_persistent_censored": tb.get("T90_persistent_censored"),
                  "frozen_finite_time_class": frozen_class(r)})
        rows.append(d)
    pd.DataFrame(rows).to_csv(OUT / "alpha_grid_complete_summary.csv", index=False)

    # 6e-4 block rates (frozen method)
    block_6e4, block_rows = {}, []
    for N in N_ORDER:
        bt = block_table_for(cells, N, 6e-4)
        if bt is None:
            continue
        block_6e4[N] = bt
        for k, v in bt["blocks"].items():
            block_rows.append({"eps_target": 6e-4, "N": N, "block": k, "lambda_block": v["lambda_block"],
                               "block_mean_J_over_2eps": v["block_mean_J_over_2eps"],
                               "block_mean_PhiJ": v["block_mean_PhiJ"],
                               "LyapLocal_block_mean": v["LyapLocal_block_mean"],
                               "LyapLocal_block_std": v["LyapLocal_block_std"], "classification": bt["classification"]})
    pd.DataFrame(block_rows).to_csv(OUT / "alpha_6e-4_block_rates_frozenmethod.csv", index=False)

    # answers to the three questions
    def mrow(eps_t, N):
        sub = master[(master.eps_target == eps_t) & (master.N == N)]
        return sub.iloc[0].to_dict() if len(sub) else None

    answers = {}
    q1 = {"nominal_duration": 1e8}
    for N in N_ORDER:
        tb = tb_idx.get((6e-4, N)); mr = mrow(6e-4, N)
        q1[f"N{N}"] = {
            "T90_first": None if tb is None or pd.isna(tb["T90_first"]) else float(tb["T90_first"]),
            "T90_first_censored": None if tb is None else bool(tb["T90_first_censored"]),
            "T90_persistent": None if tb is None or pd.isna(tb["T90_persistent"]) else float(tb["T90_persistent"]),
            "T90_persistent_censored": None if tb is None else bool(tb["T90_persistent_censored"]),
            "J_tail_over_2eps": None if mr is None else float(mr["J_tail_mean_over_2eps"]),
            "PhiJ_tail_mean": None if mr is None else float(mr["PhiJ_tail_mean"])}
    q1["interpretation"] = (
        "Finite-time (nominal 1e8): persistent T90 (approach to the 2eps equilibrium estimate) is reached "
        "at N=1024 (~9.72e7) and N=2048 (~9.91e7) but is RIGHT-CENSORED (>9.998e7, not reached) at N=512. "
        "T90_first is comparable across N (8.43e7/8.44e7/8.80e7). Tail J/(2eps) and Phi_J are close across N "
        "(J/2eps ~1.054-1.060; Phi_J ~0.88-0.89), i.e. weak/observable-consistent finite-size dependence, "
        "not a thermodynamic-limit equilibration time.")
    answers["Q1_6e-4_finite_size"] = q1

    lk = leg_key(6e-4, 512); mr = mrow(6e-4, 512)
    answers["Q2_N512_chaos_6e-4"] = {
        "cell": "eps=6e-4, N=512", "nominal_duration": 1e8,
        "FTLE_final": float(mr["FTLE_final"]), "FTLE_tail_mean": float(mr["FTLE_tail_mean"]),
        "FTLE_flat_leg_i": legs[lk]["ftle_flat"], "LyapLocal_tail_mean": float(mr["LyapLocal_tail_mean"]),
        "LyapLocal_tail_std": float(mr["LyapLocal_tail_std"]),
        "leg_ii_loc_positive": legs[lk]["loc_positive"], "leg_iii_loc_above_noise": legs[lk]["loc_above_noise"],
        "all_three_legs_pass": legs[lk]["positive_plateau_all3"],
        "block_rates": [v["lambda_block"] for v in block_6e4[512]["blocks"].values()] if 512 in block_6e4 else None,
        "block_rate_classification": block_6e4[512]["classification"] if 512 in block_6e4 else None,
        "final_frozen_category": ("positive finite-time Lyapunov plateau (chaos onset satisfied): all three "
                                  "frozen legs pass" if legs[lk]["positive_plateau_all3"] else "no positive plateau"),
        "onset_bracket_N512": scal["taskA"]["brackets"]["per_N"]["512"]["chaos_positive_plateau_all3"]["result"]}

    q3 = {"nominal_duration": 1e8, "N1024_status": cells[(1024, 6e-5)]["status"],
          "N1024_note": "recovered N=1024 eps=6e-5 diagnostic run NOT present on disk; cell missing -> "
                        "cannot compare the requested third point. Comparison is N=512 vs N=2048 only."}
    for N in (512, 2048):
        mr = mrow(6e-5, N); lk = leg_key(6e-5, N)
        bt = block_table_for(cells, N, 6e-5)
        q3[f"N{N}"] = {"eps_actual": float(mr["eps_actual"]), "J0_over_2eps": float(mr["J0_over_2eps"]),
                       "J_tail_over_2eps": float(mr["J_tail_mean_over_2eps"]),
                       "J_tail_std_over_2eps": float(mr["J_tail_std_over_2eps"]),
                       "PhiJ_tail_mean": float(mr["PhiJ_tail_mean"]), "FTLE_tail_mean": float(mr["FTLE_tail_mean"]),
                       "LyapLocal_tail_mean": float(mr["LyapLocal_tail_mean"]),
                       "LyapLocal_tail_std": float(mr["LyapLocal_tail_std"]), "Eta_tail_mean": float(mr["Eta_tail_mean"]),
                       "block_rates": bt["block_rate_vals"] if bt else None,
                       "block_classification": bt["classification"] if bt else None,
                       "chaos_positive_plateau_all3": legs[lk]["positive_plateau_all3"],
                       "frozen_low_energy_category": frozen_class(mrow(6e-5, N))}
    q3["interpretation"] = (
        "At eps=6e-5 (finite time 1e8) both available N (512, 2048) show: J barely drifts (tail J/2eps ~1.497 "
        "vs J(0)/2eps ~1.500, i.e. still ~3eps, far above the 2eps estimate), Phi_J tail mean <<0.5 (Toda onset "
        "not reached), near-zero finite-time Lyapunov growth (FTLE ~7-8e-7; block rates 'inconclusive'; leg iii "
        "loc_above_noise = False so NO positive plateau), while Eta is mid-range (0.63 at N=512, 0.70 at N=2048): "
        "substantial harmonic-mode spreading without chaos or action equilibration. This mid-range-Eta / flat-J / "
        "near-zero-FTLE contrast is the expected low-energy diagnostic separation, consistent with a "
        "quasi-stationary state (QSS) at this finite duration; finite-size behavior is consistent between the two N.")
    answers["Q3_6e-5_low_energy"] = q3
    answers["completion"] = {"constructable_cells": f"{n_ok}/18",
                             "missing": [f"{e:g}/N{n}" for (n, e) in cells if cells[(n, e)]["status"] != "ok"]}
    with open(OUT / "alpha_answers.json", "w") as f:
        json.dump(answers, f, indent=2, default=ag._json_default)

    # combined top-level QC (8 NEW files: 5 beta + 3 alpha)
    comb = []
    beta_qc_path = BETA_OUT / "beta_new_file_qc.csv"
    if beta_qc_path.is_file():
        for _, r in pd.read_csv(beta_qc_path).iterrows():
            comb.append({"family": "beta", "path": r["source_file"], "sha256": r["sha256"], "model": "beta",
                         "N": r["N"], "coeff": r["beta"], "amplitude": r["amplitude"], "dt": r["dt"],
                         "stride": r["stride"], "NumSegments": r["NumSegments"], "row_count": r["n_rows"],
                         "first_time": r["first_saved_time"], "last_time": r["last_saved_time"],
                         "nominal_duration": r["nominal_duration"], "epsilon_target": r["epsilon_target"],
                         "epsilon_actual": r["epsilon_actual"], "rel_dev": r["rel_dev"], "ppm_dev": r["ppm_dev"],
                         "max_abs_rel_energy_drift": r["max_abs_rel_energy_drift"], "nonfinite": r["has_nonfinite"],
                         "SolverGitCommit": r["SolverGitCommit"], "SolverGitDirty": r["SolverGitDirty"],
                         "QC": r["QC"], "inclusion": "included" if r["QC"] != "FAIL" else "excluded"})
    for r in alpha_qc:
        if r.get("present"):
            comb.append({"family": "alpha", "path": r["path"], "sha256": r["sha256"], "model": r["model"],
                         "N": r["N"], "coeff": r["coeff"], "amplitude": r["amplitude"], "dt": r["dt"],
                         "stride": r["stride"], "NumSegments": r["NumSegments"], "row_count": r["row_count"],
                         "first_time": r["first_saved_time"], "last_time": r["last_saved_time"],
                         "nominal_duration": r["nominal_duration"], "epsilon_target": r["epsilon_target"],
                         "epsilon_actual": r["epsilon_actual"], "rel_dev": r["rel_dev"], "ppm_dev": r["ppm_dev"],
                         "max_abs_rel_energy_drift": r["max_abs_rel_energy_drift"], "nonfinite": r["nonfinite"],
                         "SolverGitCommit": r["SolverGitCommit"], "SolverGitDirty": r["SolverGitDirty"],
                         "QC": r["QC"], "inclusion": "included" if r["QC"] != "FAIL" else "excluded"})
        else:
            comb.append({"family": "alpha", "path": r["path"], "N": r["N_expected"],
                         "epsilon_target": r["epsilon_target"], "QC": "MISSING", "inclusion": "absent (cell not filled)"})
    comb_df = pd.DataFrame(comb)
    comb_df.to_csv(OUT / "combined_toplevel_QC_8new.csv", index=False)
    if BETA_OUT.is_dir():
        comb_df.to_csv(BETA_OUT / "combined_toplevel_QC_8new.csv", index=False)

    return {"n_ok": n_ok, "matrix": matrix, "answers": answers,
            "block_rows": block_rows, "alpha_qc": alpha_qc, "combined": comb_df}


def main() -> int:
    # regenerate the frozen Tasks A-D outputs first (frozen module, imported not forked)
    ag.main(["--data-root", "data", "--output-dir", str(OUT.relative_to(REPO))])
    res = run()
    print(f"\nconstructable cells: {res['n_ok']}/18")
    print(res["matrix"].to_string())
    print("\n=== 6e-4 block rates (frozen method) ===")
    print(pd.DataFrame(res["block_rows"]).to_string(index=False))
    print("\n=== answers ===")
    print(json.dumps(res["answers"], indent=2, default=ag._json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
