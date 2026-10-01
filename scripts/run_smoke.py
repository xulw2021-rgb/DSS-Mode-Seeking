"""
Stage 2B-0 smoke tests A-D.

A: cigre14, M-ANN, 1 noise, 1 repeat, 5 epochs
B: cigre14, all 6 models, 1 noise, 1 repeat, 5 epochs (WLS: 1 noise draw)
C: ober_sub, 4 learning models, 1 noise, 1 repeat, 5 epochs
D: PanTaGruEl-14 (200 samples), S-ANN + M-ANN, 1 noise, 1 repeat, 10 epochs

Every smoke writes raw CSV rows, checkpoints, and prints a PASS/FAIL
checklist (NaN/Inf, shapes, CSV, checkpoint).  Nothing else is run.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from framework import (  # noqa: E402
    MODEL_REGISTRY, LEARNING_MODELS, SOLVER_MODELS,
    training_seed, data_seed, eval_noise_seed,
    load_and_split, corrupt_test_sets,
    train_learning_model, train_modal_dss2_with_meas_term,
    evaluate_learning_model, evaluate_solver_model, build_wls_samples,
    ResultWriter, write_manifest, gpu_name, RAW_COLUMNS,
)

DATASETS = {
    "cigre14": dict(folder="data/cigre14/",
                    meas_v=[0, 1, 12, 7, 11, 14], meas_pf=[0, 10]),
    "ober_sub": dict(folder="data/ober_sub/",
                     meas_v=[35, 16, 52, 47, 6, 48, 59, 27, 37, 56],
                     meas_pf=[40, 43, 11, 21, 54, 57]),
    "pantagruel_14": dict(folder=os.path.join(EXP, "data", "pantagruel_14_case1", ""),
                          meas_v=[0, 1, 3, 6, 8, 11], meas_pf=[0, 4, 9]),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_finite(vals, label):
    ok = all(np.isfinite(v) for v in vals if v is not None)
    print(f"    [{'PASS' if ok else 'FAIL'}] {label}: finite={ok} ({vals})")
    return ok


def run_learning_smoke(smoke_id, dataset_id, models, noises, repeat_id,
                       epochs, writer, n_wls_draws=1, max_train=None,
                       max_test=None):
    """One smoke scenario for learning (+optional solver) models."""
    ds = DATASETS[dataset_id]
    print(f"\n=== {smoke_id}: dataset={dataset_id}, models={models}, "
          f"noise={noises}, repeat={repeat_id}, epochs={epochs} ===")
    t0 = time.time()

    train_ds, test_ds, x_mean, x_std, pflow_mean, pflow_std, state_scale = \
        load_and_split(ds["folder"], ds["meas_v"], ds["meas_pf"],
                       experiment_id=smoke_id, max_train=max_train,
                       max_test=max_test)
    print(f"    train={len(train_ds)}, test={len(test_ds)}, "
          f"state_scale={state_scale.tolist()}  [load {time.time()-t0:.1f}s]")

    corrupted = corrupt_test_sets(test_ds, noises, x_mean, x_std,
                                  pflow_mean, pflow_std)
    all_ok = True
    for model_id in models:
        t1 = time.time()
        if not MODEL_REGISTRY[model_id]["trainable"]:
            samples = build_wls_samples(ds["folder"], ds["meas_v"], n_samples=10)
            metrics = evaluate_solver_model(model_id, samples, noises,
                                            n_draws=n_wls_draws)
            loss_val = np.nan
            ckpt = None
        else:
            torch.cuda.reset_peak_memory_stats()
            model, loss_val, _ = train_learning_model(
                model_id, train_ds, epochs=epochs,
                batch_size=32 if model_id.endswith("dss2") else 64,
                device=DEVICE, x_mean=x_mean, x_std=x_std,
                pflow_mean=pflow_mean, pflow_std=pflow_std,
                state_scale=state_scale, repeat_id=repeat_id)
            metrics = evaluate_learning_model(
                model, model_id, corrupted, x_mean, x_std, state_scale,
                DEVICE, batch_size=64)
            ckpt_dir = os.path.join(EXP, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt = os.path.join(ckpt_dir, f"{smoke_id}_{model_id}.pt")
            torch.save(model.state_dict(), ckpt)

        for noise in noises:
            m = metrics[noise]
            writer.append({
                "experiment_id": smoke_id, "dataset_id": dataset_id,
                "model": model_id, "model_name": MODEL_REGISTRY[model_id]["model_name"],
                "architecture": MODEL_REGISTRY[model_id]["architecture"],
                "loss": MODEL_REGISTRY[model_id]["loss"],
                "noise": noise, "bad_data_rate": 0.0, "bandwidth": 1.0,
                "bandwidth_schedule": "fixed", "lambda_meas": 0.0,
                "eta": 3e-3, "repeat_id": repeat_id,
                "training_seed": training_seed(repeat_id),
                "data_seed": data_seed(smoke_id),
                "eval_noise_seed": eval_noise_seed(noise),
                "epochs": epochs, "best_epoch": epochs,
                "train_loss": loss_val, "val_loss": np.nan,
                "v_rmse": m["v_rmse"], "th_rmse": m["th_rmse"],
                "loading_rmse": m["loading_rmse"], "modal_score": m["modal_score"],
                "hit_rate": m["hit_rate"], "runtime_seconds": m["runtime_seconds"],
                "converged": m.get("converged", 1.0), "device": str(DEVICE),
                "gpu_name": gpu_name(), "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "notes": f"smoke run ({epochs} epochs)" if
                         MODEL_REGISTRY[model_id]["trainable"]
                         else f"smoke run ({n_wls_draws} noise draw(s), 10 samples)",
            })
        mem = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
        print(f"    [{model_id}] train_loss={loss_val}, v_rmse(default)="
              f"{metrics[noises[0]]['v_rmse']:.5f}, time={time.time()-t1:.1f}s, "
              f"peak_gpu={mem:.2f}GB, ckpt={ckpt}")
        fin_vals = [metrics[noises[0]]["v_rmse"], metrics[noises[0]]["th_rmse"]]
        if MODEL_REGISTRY[model_id]["trainable"]:
            fin_vals.insert(0, loss_val)   # train_loss only for trained models
        all_ok &= check_finite(fin_vals, f"{model_id} finite")
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", default="A,B,C,D", help="subset, e.g. A,B")
    args = ap.parse_args()
    subset = args.smoke.split(",")

    os.makedirs(os.path.join(EXP, "logs"), exist_ok=True)
    results = {}

    if "A" in subset:
        writer = ResultWriter(os.path.join(EXP, "raw_results", "smoke_A.csv"))
        ok = run_learning_smoke("smoke_A", "cigre14",
                                ["modal_supervised_ann"], ["default"], 0, 5, writer)
        write_manifest("smoke_A", os.path.join(EXP, "raw_results"),
                       {"dataset": "cigre14", "models": ["modal_supervised_ann"],
                        "noise": ["default"], "repeats": 1, "epochs": 5})
        results["A"] = ok

    if "B" in subset:
        writer = ResultWriter(os.path.join(EXP, "raw_results", "smoke_B.csv"))
        ok = run_learning_smoke("smoke_B", "cigre14",
                                ["wls", "modal_wls", "supervised_ann",
                                 "modal_supervised_ann", "supervised_dss2",
                                 "modal_supervised_dss2"],
                                ["default"], 0, 5, writer, n_wls_draws=1)
        write_manifest("smoke_B", os.path.join(EXP, "raw_results"),
                       {"dataset": "cigre14", "models": "all 6",
                        "noise": ["default"], "repeats": 1, "epochs": 5})
        results["B"] = ok

    if "C" in subset:
        writer = ResultWriter(os.path.join(EXP, "raw_results", "smoke_C.csv"))
        ok = run_learning_smoke("smoke_C", "ober_sub",
                                ["supervised_ann", "modal_supervised_ann",
                                 "supervised_dss2", "modal_supervised_dss2"],
                                ["default"], 0, 5, writer,
                                max_train=200, max_test=50)
        write_manifest("smoke_C", os.path.join(EXP, "raw_results"),
                       {"dataset": "ober_sub", "models": "4 learning",
                        "noise": ["default"], "repeats": 1, "epochs": 5})
        results["C"] = ok

    if "D" in subset:
        writer = ResultWriter(os.path.join(EXP, "raw_results", "smoke_D.csv"))
        ok = run_learning_smoke("smoke_D", "pantagruel_14",
                                ["supervised_ann", "modal_supervised_ann"],
                                ["default"], 0, 10, writer,
                                max_train=180, max_test=20)
        write_manifest("smoke_D", os.path.join(EXP, "raw_results"),
                       {"dataset": "pantagruel_14_case1 (Zenodo 5816149)",
                        "models": ["supervised_ann", "modal_supervised_ann"],
                        "noise": ["default"], "repeats": 1, "epochs": 10})
        results["D"] = ok

    print("\n" + "=" * 60)
    for k, v in results.items():
        print(f"Smoke {k}: {'PASS' if v else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
