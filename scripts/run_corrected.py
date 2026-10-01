"""
Stage 2C corrected full reruns (preprocessing_version='train_only').
Same experiment_ids as legacy (=> same data_seed/split), same training and
eval seeds, same architectures/hyperparameters. Only the normalization
pipeline differs (build_dataset_train_only).

Experiments: E1-learning, E2, E3, E4, E6A, E6B (E1 solvers and E7 are
unaffected and NOT rerun — FINAL_RERUN_DECISION_RULE).

Usage:
  python run_corrected.py --chunk A|B        (parallel chunks)
      chunk A (cuda:0): E1, E2, E6A, E6B
      chunk B (cuda:1): E3, E4
  python run_corrected.py --finalize          (merged summaries/manifest)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from framework import (  # noqa: E402
    MODEL_REGISTRY, NOISE_SETTINGS, training_seed, data_seed, eval_noise_seed,
    build_dataset_train_only, corrupt_test_sets, train_learning_model,
    train_modal_dss2_with_meas_term, evaluate_learning_model, ResultWriter,
    summarize, paired_bootstrap, write_manifest, gpu_name, RAW_COLUMNS,
    STATUS_SUCCESS, STATUS_TRAINING_FAILURE,
    STATUS_EVALUATION_FAILURE, STATUS_TECHNICAL_ERROR, TrainingDivergedError,
)

OUT_DIR = os.path.join(EXP, "raw_results", "CORRECTED")
SUM_DIR = os.path.join(EXP, "summaries", "CORRECTED")
MAN_DIR = os.path.join(EXP, "manifests", "CORRECTED")
CKPT_DIR = os.path.join(EXP, "checkpoints", "CORRECTED")
DEV0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEV1 = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cpu")

DATASETS = {
    "cigre14": dict(folder="data/cigre14/",
                    meas_v=[0, 1, 12, 7, 11, 14], meas_pf=[0, 10]),
    "ober_sub": dict(folder="data/ober_sub/",
                     meas_v=[35, 16, 52, 47, 6, 48, 59, 27, 37, 56],
                     meas_pf=[40, 43, 11, 21, 54, 57]),
    "ieee30": dict(folder=os.path.join(EXP, "data", "ieee30", ""),
                   meas_v=[0, 2, 5, 9, 14, 19, 24, 27, 29],
                   meas_pf=[0, 4, 8, 12, 16, 20]),
    "pantagruel_14": dict(folder=os.path.join(EXP, "data",
                                              "pantagruel_14_case1", ""),
                          meas_v=[0, 1, 3, 6, 8, 11], meas_pf=[0, 4, 9]),
}


def write_row(writer, exp, dataset_id, ds_note, model_id, noise, rep, epochs,
              train_loss, met, dev, extra_notes=""):
    m = MODEL_REGISTRY[model_id]
    writer.append({
        "experiment_id": exp, "dataset_id": dataset_id,
        "data_source": f"corrected_{dataset_id}",
        "external_benchmark": bool(dataset_id == "pantagruel_14"),
        "model": model_id, "model_name": m["model_name"],
        "architecture": m["architecture"], "loss": m["loss"],
        "noise": noise, "bad_data_rate": 0.0, "bandwidth": 1.0,
        "bandwidth_schedule": "fixed", "lambda_meas": 0.0, "eta": 3e-3,
        "repeat_id": rep, "training_seed": training_seed(rep),
        "data_seed": data_seed(exp),
        "eval_noise_seed": eval_noise_seed(noise),
        "epochs": epochs, "best_epoch": epochs,
        "train_loss": train_loss, "val_loss": np.nan,
        "v_rmse": met["v_rmse"], "th_rmse": met["th_rmse"],
        "loading_rmse": met["loading_rmse"], "modal_score": met["modal_score"],
        "hit_rate": met["hit_rate"], "runtime_seconds": met["runtime_seconds"],
        "converged": 1.0, "device": str(dev), "gpu_name": gpu_name(),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "status": STATUS_SUCCESS, "failure_epoch": np.nan,
        "failure_reason": "", "v_mae": met["v_mae"], "th_mae": met["th_mae"],
        "state_loss": np.nan, "meas_loss": np.nan,
        "notes": f"preprocessing_version=train_only; {extra_notes}",
    })


def load_corrected(dataset_key, exp_id, max_train=None, max_test=None):
    ds = DATASETS[dataset_key]
    return build_dataset_train_only(
        ds["folder"], ds["meas_v"], ds["meas_pf"], experiment_id=exp_id,
        max_train=max_train, max_test=max_test)


def train_and_eval(model_id, train_ds, corrupted, xm, xs, pm, ps, ss, dev,
                   epochs, batch, rep, bw=1.0, meas_term=False, lam=0.0):
    if meas_term:
        model, loss, hist = train_modal_dss2_with_meas_term(
            model_id, train_ds, epochs=epochs, batch_size=batch, device=dev,
            x_mean=xm, x_std=xs, pflow_mean=pm, pflow_std=ps,
            state_scale=ss, repeat_id=rep, bandwidth=bw, lambda_meas=lam,
            lambda_phys=1.0, annealing=False)
    else:
        model, loss, hist = train_learning_model(
            model_id, train_ds, epochs=epochs, batch_size=batch, device=dev,
            x_mean=xm, x_std=xs, pflow_mean=pm, pflow_std=ps,
            state_scale=ss, repeat_id=rep, bandwidth=bw)
    met = evaluate_learning_model(model, model_id, corrupted, xm, xs, ss, dev)
    return loss, met


def run_experiment(csv_path, exp_id, dataset_key, models, noises, epochs,
                   batch_select, dev, max_train=None, max_test=None, bw=1.0,
                   meas_term=False, lams=None):
    if os.path.exists(csv_path):
        os.remove(csv_path)
    writer = ResultWriter(csv_path)
    train_ds, test_ds, xm, xs, pm, ps, ss = load_corrected(
        dataset_key, exp_id, max_train, max_test)
    corrupted = corrupt_test_sets(test_ds, noises, xm, xs, pm, ps)
    print(f"[{exp_id}] train={len(train_ds)} test={len(test_ds)}", flush=True)
    t0 = time.time()
    lam_list = lams if lams is not None else [0.0]
    for lam in lam_list:
        for model_id in models:
            for rep in range(5):
                try:
                    loss, met = train_and_eval(
                        model_id, train_ds, corrupted, xm, xs, pm, ps, ss,
                        dev, epochs, batch_select(model_id), rep, bw=bw,
                        meas_term=meas_term, lam=lam)
                    for noise in noises:
                        write_row(writer, exp_id, dataset_key, "", model_id,
                                  noise, rep, epochs, loss, met[noise], dev,
                                  extra_notes=f"lambda_meas={lam}")
                    print(f"[{exp_id} {model_id} lam={lam} rep{rep}] "
                          f"v(def)={met[noises[0]]['v_rmse']:.5f} "
                          f"elapsed={time.time()-t0:.0f}s", flush=True)
                except TrainingDivergedError as e:
                    print(f"[{exp_id} {model_id} lam={lam} rep{rep}] "
                          f"TRAINING_FAILURE {e}", flush=True)
                except Exception as e:
                    print(f"[{exp_id} {model_id} lam={lam} rep{rep}] "
                          f"TECHNICAL {type(e).__name__}: {e}", flush=True)
    print(f"[{exp_id}] DONE in {(time.time()-t0)/60:.1f} min", flush=True)


def chunk_A():
    os.makedirs(OUT_DIR, exist_ok=True)
    # E1 learning rows (10 noises)
    run_experiment(os.path.join(OUT_DIR, "E1_learning_corrected.csv"),
                   "E1_cigre14_main", "cigre14",
                   ["supervised_ann", "modal_supervised_ann",
                    "supervised_dss2", "modal_supervised_dss2"],
                   NOISE_SETTINGS, 200,
                   lambda m: 32 if m.endswith("dss2") else 64, DEV0)
    # E2 (5 noises, 100 epochs, caps)
    run_experiment(os.path.join(OUT_DIR, "E2_corrected.csv"),
                   "E2_ober70_fair", "ober_sub",
                   ["supervised_ann", "modal_supervised_ann",
                    "supervised_dss2", "modal_supervised_dss2"],
                   ["default", "student_t3", "mixture_010", "erroneous",
                    "missing_erroneous"], 100,
                   lambda m: 16 if m.endswith("dss2") else 64, DEV0,
                   max_train=500, max_test=100)
    # E6A ieee30 (3 noises)
    run_experiment(os.path.join(OUT_DIR, "E6A_ieee30_corrected.csv"),
                   "E6_ieee30", "ieee30",
                   ["supervised_ann", "modal_supervised_ann",
                    "supervised_dss2", "modal_supervised_dss2"],
                   ["default", "mixture_010", "erroneous"], 200,
                   lambda m: 32 if m.endswith("dss2") else 64, DEV0)
    # E6B pantagruel (3 noises, ANN only)
    run_experiment(os.path.join(OUT_DIR, "E6B_pantagruel_corrected.csv"),
                   "E6_pantagruel", "pantagruel_14",
                   ["supervised_ann", "modal_supervised_ann"],
                   ["default", "mixture_010", "erroneous"], 200,
                   lambda m: 64, DEV0)


def chunk_B():
    os.makedirs(OUT_DIR, exist_ok=True)
    # E3 bandwidth (4 bw x 4 noises): bw in {0.5,1.0,2.0} + annealing 8->2
    noises = ["default", "mixture_005", "mixture_010", "mixture_020"]
    csv = os.path.join(OUT_DIR, "E3_corrected.csv")
    if os.path.exists(csv):
        os.remove(csv)
    writer = ResultWriter(csv)
    train_ds, test_ds, xm, xs, pm, ps, ss = load_corrected(
        "cigre14", "E3_cigre14_bandwidth")
    corrupted = corrupt_test_sets(test_ds, noises, xm, xs, pm, ps)
    t0 = time.time()
    for model_id in ["modal_supervised_ann", "modal_supervised_dss2"]:
        batch = 64 if model_id.endswith("ann") else 32
        for bw_key, (bw, anneal) in [("b0.5", (0.5, False)),
                                     ("b1.0", (1.0, False)),
                                     ("b2.0", (2.0, False)),
                                     ("anneal_8_2", (2.0, True))]:
            for rep in range(5):
                try:
                    model, loss, _ = train_learning_model(
                        model_id, train_ds, epochs=200, batch_size=batch,
                        device=DEV1, x_mean=xm, x_std=xs, pflow_mean=pm,
                        pflow_std=ps, state_scale=ss, repeat_id=rep,
                        bandwidth=bw, annealing=anneal, b_start=8.0, b_end=2.0)
                    met = evaluate_learning_model(model, model_id, corrupted,
                                                  xm, xs, ss, DEV1)
                    for noise in noises:
                        row_extra = f"bw_key={bw_key}"
                        m = MODEL_REGISTRY[model_id]
                        writer.append({
                            "experiment_id": "E3_cigre14_bandwidth",
                            "dataset_id": "cigre14",
                            "data_source": "corrected_cigre14",
                            "external_benchmark": False,
                            "model": model_id, "model_name": m["model_name"],
                            "architecture": m["architecture"],
                            "loss": m["loss"], "noise": noise,
                            "bad_data_rate": 0.0, "bandwidth": bw,
                            "bandwidth_schedule":
                                "anneal_8_2" if anneal else "fixed",
                            "lambda_meas": 0.0, "eta": 3e-3,
                            "repeat_id": rep,
                            "training_seed": training_seed(rep),
                            "data_seed": data_seed("E3_cigre14_bandwidth"),
                            "eval_noise_seed": eval_noise_seed(noise),
                            "epochs": 200, "best_epoch": 200,
                            "train_loss": loss, "val_loss": np.nan,
                            "v_rmse": met[noise]["v_rmse"],
                            "th_rmse": met[noise]["th_rmse"],
                            "loading_rmse": met[noise]["loading_rmse"],
                            "modal_score": met[noise]["modal_score"],
                            "hit_rate": met[noise]["hit_rate"],
                            "runtime_seconds": met[noise]["runtime_seconds"],
                            "converged": 1.0, "device": str(DEV1),
                            "gpu_name": gpu_name(),
                            "torch_version": torch.__version__,
                            "cuda_version": torch.version.cuda,
                            "status": STATUS_SUCCESS,
                            "failure_epoch": np.nan, "failure_reason": "",
                            "v_mae": met[noise]["v_mae"],
                            "th_mae": met[noise]["th_mae"],
                            "state_loss": np.nan, "meas_loss": np.nan,
                            "notes": f"preprocessing_version=train_only; {row_extra}",
                        })
                    print(f"[E3 {model_id} {bw_key} rep{rep}] "
                          f"v(def)={met['default']['v_rmse']:.5f} "
                          f"elapsed={time.time()-t0:.0f}s", flush=True)
                except TrainingDivergedError as e:
                    print(f"[E3 {model_id} {bw_key} rep{rep}] FAIL {e}",
                          flush=True)
                except Exception as e:
                    print(f"[E3 {model_id} {bw_key} rep{rep}] ERR "
                          f"{type(e).__name__}: {e}", flush=True)
    # E4 lambda_meas (5 lam x 4 noises)
    csv4 = os.path.join(OUT_DIR, "E4_corrected.csv")
    if os.path.exists(csv4):
        os.remove(csv4)
    writer4 = ResultWriter(csv4)
    train_ds4, test_ds4, xm4, xs4, pm4, ps4, ss4 = load_corrected(
        "cigre14", "E4_cigre14_lambdameas")
    corrupted4 = corrupt_test_sets(test_ds4, noises, xm4, xs4, pm4, ps4)
    for lam in [0.0, 0.05, 0.10, 0.20, 0.50]:
        for rep in range(5):
            try:
                model, loss, _ = train_modal_dss2_with_meas_term(
                    "modal_supervised_dss2", train_ds4, epochs=200,
                    batch_size=32, device=DEV1, x_mean=xm4, x_std=xs4,
                    pflow_mean=pm4, pflow_std=ps4, state_scale=ss4,
                    repeat_id=rep, bandwidth=1.0, lambda_meas=lam,
                    lambda_phys=1.0, annealing=False)
                met = evaluate_learning_model(model, "modal_supervised_dss2",
                                              corrupted4, xm4, xs4, ss4, DEV1)
                for noise in noises:
                    m = MODEL_REGISTRY["modal_supervised_dss2"]
                    writer4.append({
                        "experiment_id": "E4_cigre14_lambdameas",
                        "dataset_id": "cigre14",
                        "data_source": "corrected_cigre14",
                        "external_benchmark": False,
                        "model": "modal_supervised_dss2",
                        "model_name": m["model_name"],
                        "architecture": m["architecture"],
                        "loss": m["loss"], "noise": noise,
                        "bad_data_rate": 0.0, "bandwidth": 1.0,
                        "bandwidth_schedule": "fixed", "lambda_meas": lam,
                        "eta": 3e-3, "repeat_id": rep,
                        "training_seed": training_seed(rep),
                        "data_seed": data_seed("E4_cigre14_lambdameas"),
                        "eval_noise_seed": eval_noise_seed(noise),
                        "epochs": 200, "best_epoch": 200,
                        "train_loss": loss, "val_loss": np.nan,
                        "v_rmse": met[noise]["v_rmse"],
                        "th_rmse": met[noise]["th_rmse"],
                        "loading_rmse": met[noise]["loading_rmse"],
                        "modal_score": met[noise]["modal_score"],
                        "hit_rate": met[noise]["hit_rate"],
                        "runtime_seconds": met[noise]["runtime_seconds"],
                        "converged": 1.0, "device": str(DEV1),
                        "gpu_name": gpu_name(),
                        "torch_version": torch.__version__,
                        "cuda_version": torch.version.cuda,
                        "status": STATUS_SUCCESS,
                        "failure_epoch": np.nan, "failure_reason": "",
                        "v_mae": met[noise]["v_mae"],
                        "th_mae": met[noise]["th_mae"],
                        "state_loss": np.nan, "meas_loss": np.nan,
                        "notes": "preprocessing_version=train_only",
                    })
                print(f"[E4 lam={lam} rep{rep}] "
                      f"v(def)={met['default']['v_rmse']:.5f} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
            except TrainingDivergedError as e:
                print(f"[E4 lam={lam} rep{rep}] FAIL {e}", flush=True)
            except Exception as e:
                print(f"[E4 lam={lam} rep{rep}] ERR {type(e).__name__}: {e}",
                      flush=True)
    print("[chunk B] DONE", flush=True)


def finalize():
    os.makedirs(SUM_DIR, exist_ok=True)
    os.makedirs(MAN_DIR, exist_ok=True)
    write_manifest("2C_corrected_reruns", MAN_DIR,
                   {"preprocessing_version": "train_only",
                    "rerun_rule": "FINAL_RERUN_DECISION_RULE.md",
                    "experiments": ["E1_learning", "E2", "E3", "E4", "E6A",
                                    "E6B"],
                    "not_rerun": ["E1 solvers", "E7 (unaffected)"],
                    "seed_policy": "identical to legacy experiments"},
                   notes="corrected full reruns, Stage 2C")
    print("manifest written; per-experiment CSVs under raw_results/CORRECTED/")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", choices=["A", "B"], default=None)
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    if args.chunk == "A":
        chunk_A()
    elif args.chunk == "B":
        chunk_B()
    elif args.finalize:
        finalize()
    else:
        ap.print_help()
