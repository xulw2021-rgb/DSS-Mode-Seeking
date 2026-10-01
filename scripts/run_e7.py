"""
E7: WLS / Modal-WLS multi-start solver stability (Stage 2B-E7).

Protocol (E7_INITIALIZATION_PROTOCOL.md):
  - CIGRE-14, 10 fixed test samples, noises {default, mixture_010,
    mixture_020, erroneous}, 5 noise draws per sample (same rng scheme
    as E1: RandomState(eval_noise_seed(noise))).
  - 10 starts per (sample, noise, draw): start 1 = flat start (E1
    reproduction); starts 2..10 = flat + shared uniform(±0.05) perturbation
    (identical for both solvers; no oracle information).
  - Selection among converged starts by the solver's OWN objective
    (never ground-truth RMSE).  K in {1,3,5,10} derived from the first K
    starts of the same sequence.
  - Status: CONVERGED / NON_CONVERGED / NUMERICAL_FAILURE / TECHNICAL_ERROR;
    multi-start: MULTISTART_SUCCESS / MULTISTART_FAILURE.

Usage:
  python run_e7.py --reproduce   # K=1 only, compare vs E1 solver rows
  python run_e7.py --smoke       # 3 samples, 2 draws, K=1,3,5
  python run_e7.py --chunk wls|modal   # full run (parallel processes)
  python run_e7.py --finalize    # merge -> tables/figures/summaries
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from framework import (  # noqa: E402
    eval_noise_seed, inject_noise_to_measurements, build_wls_samples,
    write_manifest, gpu_name,
)
from modal_wls import solve_wls, solve_modal_wls

EXPERIMENT_ID = "E7_wls_multistart"
FOLDER = "data/cigre14/"
MEAS_V = [0, 1, 12, 7, 11, 14]
E7_NOISES = ["default", "mixture_010", "mixture_020", "erroneous"]
N_SAMPLES = 10
N_DRAWS = 5
N_STARTS = 10
K_VALUES = [1, 3, 5, 10]
BANDWIDTH_MODAL = 2.0   # frozen solver bandwidth (E7_SOLVER_AUDIT.md)

RAW_DIR = os.path.join(EXP, "raw_results", "E7")
START_CSV = os.path.join(RAW_DIR, "e7_start_level.csv")
SEL_CSV = os.path.join(RAW_DIR, "e7_selected_solution.csv")
TAB_DIR = os.path.join(EXP, "tables", "E7")
FIG_DIR = os.path.join(EXP, "figures", "E7")
MAN_DIR = os.path.join(EXP, "manifests", "E7")
SUM_DIR = os.path.join(EXP, "summaries", "E7")

START_COLS = ["experiment_id", "dataset_id", "solver", "noise",
              "sample_idx", "draw_idx", "start_id", "start_seed",
              "init_norm", "status", "converged", "n_iter", "objective",
              "v_rmse", "th_rmse", "runtime_s", "notes"]
SEL_COLS = ["experiment_id", "dataset_id", "solver", "noise",
            "sample_idx", "draw_idx", "K", "n_converged_starts",
            "multistart_status", "selected_start", "selected_objective",
            "v_rmse", "th_rmse", "runtime_s_total", "notes"]


def start_seed(noise_idx, sample, draw, start_id):
    return 1000 + 100 * noise_idx + 10 * sample + 1000 * draw + start_id


def make_starts(n_nodes, noise_idx, sample, draw):
    """All 10 initial states. Start 1 = flat start; 2..10 = shared
    uniform(±0.05) perturbations. Deterministic and solver-agnostic."""
    flat = np.concatenate([np.ones(n_nodes), np.zeros(n_nodes)])
    starts = [flat.copy()]
    for k in range(2, N_STARTS + 1):
        rng = np.random.RandomState(start_seed(noise_idx, sample, draw, k))
        delta = np.concatenate([rng.uniform(-0.05, 0.05, n_nodes),
                                rng.uniform(-0.05, 0.05, n_nodes)])
        starts.append(flat + delta)
    return starts


def run_solver(solver, meas_c, node_param, edge_param, edge_index, x0):
    solver_key = "modal_wls" if solver == "modal" else solver
    t0 = time.time()
    try:
        if solver_key == "modal_wls":
            res = solve_modal_wls(meas_c, node_param, edge_param, edge_index,
                                  x0=x0, bandwidth=BANDWIDTH_MODAL,
                                  lam_phys=0.0, max_iter=30, verbose=0)
        else:
            res = solve_wls(meas_c, node_param, edge_param, edge_index,
                            x0=x0, lam_phys=0.0)
        status = "CONVERGED" if bool(res["converged"]) else "NON_CONVERGED"
        conv = bool(res["converged"])
        obj = float(res["objective"])
        V_est = res["V_est"]; th_est = res["theta_est"]
        n_iter = int(res.get("n_iter", -1))
    except Exception as e:
        return dict(status="NUMERICAL_FAILURE", converged=False, n_iter=-1,
                    objective=np.nan, V_est=None, th_est=None,
                    runtime_s=time.time() - t0,
                    notes=f"{type(e).__name__}: {e}")
    return dict(status=status, converged=conv, n_iter=n_iter, objective=obj,
                V_est=V_est, th_est=th_est, runtime_s=time.time() - t0,
                notes="")


def run_solver_unit(solver, sample_idx, sample, noise, draw, max_starts,
                    writer=None, sel_writer=None):
    solver_key = "modal_wls" if solver == "modal" else solver
    noise_idx = E7_NOISES.index(noise)
    rng = np.random.RandomState(eval_noise_seed(noise))
    # advance rng to (sample_idx, draw) exactly as E1 did (sequential draws)
    for s_i in range(sample_idx + 1):
        zc = sample["meas"]["z"].copy(); sc = sample["meas"]["sigma"].copy()
        for d in range(N_DRAWS):
            z_c, s_c = inject_noise_to_measurements(zc, sc, noise, rng)
            if s_i == sample_idx and d == draw:
                meas_c = dict(sample["meas"]); meas_c["z"] = z_c
                meas_c["sigma"] = s_c
    n_nodes = sample["true_V"].shape[0]
    starts = make_starts(n_nodes, noise_idx, sample_idx, draw)

    start_rows, best = [], None
    best_obj = np.inf
    n_conv = 0
    t_unit0 = time.time()
    for k, x0 in enumerate(starts[:max_starts], start=1):
        r = run_solver(solver, meas_c, sample["node_param"],
                       sample["edge_param"], sample["edge_index"], x0)
        if r["V_est"] is not None and np.all(np.isfinite(r["V_est"])):
            v_rmse = np.sqrt(np.mean((r["V_est"] - sample["true_V"]) ** 2))
            th_rmse = np.sqrt(np.mean((r["th_est"] - sample["true_theta"]) ** 2))
        else:
            v_rmse = th_rmse = np.nan
        if r["converged"] and np.isfinite(r["objective"]) and r["objective"] < best_obj:
            best_obj = r["objective"]
            best = dict(start_id=k, objective=r["objective"], v_rmse=v_rmse,
                        th_rmse=th_rmse, status=r["status"])
        if r["converged"]:
            n_conv += 1
        start_rows.append({
            "experiment_id": EXPERIMENT_ID, "dataset_id": "cigre14",
            "solver": solver_key, "noise": noise, "sample_idx": sample_idx,
            "draw_idx": draw, "start_id": k,
            "start_seed": start_seed(noise_idx, sample_idx, draw, k),
            "init_norm": float(np.linalg.norm(x0 - np.concatenate(
                [np.ones(n_nodes), np.zeros(n_nodes)]))),
            "status": r["status"], "converged": int(r["converged"]),
            "n_iter": r["n_iter"], "objective": r["objective"],
            "v_rmse": v_rmse, "th_rmse": th_rmse,
            "runtime_s": r["runtime_s"], "notes": r["notes"],
        })
        if writer is not None:
            writer.append(start_rows[-1])
    total_t = time.time() - t_unit0
    ms_status = "MULTISTART_SUCCESS" if best is not None else "MULTISTART_FAILURE"
    sel = {
        "experiment_id": EXPERIMENT_ID, "dataset_id": "cigre14",
        "solver": solver_key, "noise": noise, "sample_idx": sample_idx,
        "draw_idx": draw, "K": max_starts, "n_converged_starts": n_conv,
        "multistart_status": ms_status,
        "selected_start": best["start_id"] if best else -1,
        "selected_objective": best["objective"] if best else np.nan,
        "v_rmse": best["v_rmse"] if best else np.nan,
        "th_rmse": best["th_rmse"] if best else np.nan,
        "runtime_s_total": total_t, "notes": "",
    }
    if sel_writer is not None:
        sel_writer.append(sel)
    return sel


class CSVAppender:
    def __init__(self, path, cols):
        self.path = path; self.cols = cols
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            pd.DataFrame(columns=cols).to_csv(path, index=False)

    def append(self, row):
        pd.DataFrame([row]).reindex(columns=self.cols).to_csv(
            self.path, mode="a", header=False, index=False)


def run_chunk(solver, n_samples=N_SAMPLES, n_draws=N_DRAWS,
              max_starts=N_STARTS, out_tag=""):
    tag = f"_{out_tag}" if out_tag else ""
    st_csv = os.path.join(RAW_DIR, f"e7_start_level_{solver}{tag}.csv")
    se_csv = os.path.join(RAW_DIR, f"e7_selected_{solver}{tag}.csv")
    sw = CSVAppender(st_csv, START_COLS)
    selw = CSVAppender(se_csv, SEL_COLS)
    samples = build_wls_samples(FOLDER, MEAS_V, n_samples=n_samples)
    t0 = time.time()
    for noise in E7_NOISES:
        for s_i, sample in enumerate(samples):
            for d in range(n_draws):
                run_solver_unit(solver, s_i, sample, noise, d, max_starts,
                                writer=sw, sel_writer=selw)
        print(f"[{solver} {noise}] {time.time()-t0:.0f}s elapsed", flush=True)
    print(f"[{solver}] DONE in {(time.time()-t0)/60:.1f} min", flush=True)


def reproduce():
    """K=1 only; compare vs E1 solver rows."""
    print("E7 single-start reproduction (K=1, both solvers)")
    out = {}
    for solver in ["wls", "modal_wls"]:
        conv, vr = [], []
        samples = build_wls_samples(FOLDER, MEAS_V, n_samples=N_SAMPLES)
        for noise in E7_NOISES:
            cc, vv = [], []
            for s_i, sample in enumerate(samples):
                for d in range(N_DRAWS):
                    sel = run_solver_unit(solver, s_i, sample, noise, d, 1)
                    cc.append(sel["multistart_status"] == "MULTISTART_SUCCESS")
                    vv.append(sel["v_rmse"])
            out[(solver, noise)] = (float(np.mean(cc)), float(np.nanmean(vv)))
            print(f"  {solver:10s} {noise:14s} conv={np.mean(cc):.2f} "
                  f"v_rmse={np.nanmean(vv):.4f}")
    e1 = pd.read_csv(os.path.join(EXP, "raw_results", "E1", "E1_raw_results.csv"))
    e1 = e1[e1.model.isin(["wls", "modal_wls"])]
    print("\nE1 comparison:")
    for solver in ["wls", "modal_wls"]:
        for noise in E7_NOISES:
            g = e1[(e1.model == solver) & (e1.noise == noise)]
            if len(g):
                print(f"  E1 {solver:10s} {noise:14s} conv={g.converged.iloc[0]:.2f} "
                      f"v_rmse={g.v_rmse.iloc[0]:.4f} | E7 repro "
                      f"conv={out[(solver,noise)][0]:.2f} "
                      f"v_rmse={out[(solver,noise)][1]:.4f}")


def smoke():
    print("E7 smoke (3 samples, 2 draws, K=1,3,5)")
    for solver in ["wls", "modal_wls"]:
        run_chunk(solver, n_samples=3, n_draws=2, max_starts=5, out_tag="smoke")
    print("smoke done")


def finalize():
    dfs_s, dfs_e = [], []
    _fname = {"wls": "wls", "modal_wls": "modal"}   # chunk names
    for solver in ["wls", "modal_wls"]:
        p = os.path.join(RAW_DIR, f"e7_start_level_{_fname[solver]}.csv")
        if not os.path.exists(p):
            print(f"MISSING {p}")
            return
        dfs_s.append(pd.read_csv(p))
        dfs_e.append(pd.read_csv(os.path.join(RAW_DIR, f"e7_selected_{_fname[solver]}.csv")))
    st = pd.concat(dfs_s, ignore_index=True)
    sel = pd.concat(dfs_e, ignore_index=True)
    # rows carry the true solver keys; chunk names are used only for files
    st.to_csv(START_CSV, index=False)
    sel.to_csv(SEL_CSV, index=False)
    for d in (SUM_DIR, TAB_DIR, FIG_DIR, MAN_DIR):
        os.makedirs(d, exist_ok=True)
    write_manifest(
        EXPERIMENT_ID, MAN_DIR,
        {"dataset": "cigre14", "n_samples": N_SAMPLES, "n_draws": N_DRAWS,
         "n_starts": N_STARTS, "K_values": K_VALUES,
         "noises": E7_NOISES, "modal_bandwidth": BANDWIDTH_MODAL,
         "init_protocol": "start1=flat; starts2-10=flat+U(±0.05) shared; no oracle",
         "selection": "min own objective among converged starts (no ground truth)",
         "solvers_audit": "E7_SOLVER_AUDIT.md"},
        notes="E7 multi-start solver stability")
    print(f"merged: {len(st)} start rows, {len(sel)} selected rows")
    print("summaries + manifest written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reproduce", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--chunk", choices=["wls", "modal"], default=None)
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    if args.reproduce:
        reproduce()
    elif args.smoke:
        smoke()
    elif args.chunk:
        run_chunk(args.chunk)
    elif args.finalize:
        finalize()
    else:
        ap.print_help()
