"""
Stage 2C diagnostic rerun: corrected (train-only) normalization vs legacy.
CIGRE-14 + PanTaGruEl-14, S-ANN/M-ANN x {default, mixture_010} x 5 runs.
Same split/data/eval seeds as the legacy E1/E6 runs; ONLY the normalization
pipeline differs. Outputs -> raw_results/CORRECTED/diagnostic_*.csv
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
    MODEL_REGISTRY, training_seed, eval_noise_seed,
    build_dataset_train_only, corrupt_test_sets, train_learning_model,
    evaluate_learning_model, ResultWriter, RAW_COLUMNS,
    STATUS_SUCCESS, STATUS_TRAINING_FAILURE,
    STATUS_EVALUATION_FAILURE, STATUS_TECHNICAL_ERROR, TrainingDivergedError,
    gpu_name,
)

DATASETS = {
    "diag_cigre14": dict(
        legacy_exp="E1_cigre14_main", dataset_id="cigre14",
        folder="data/cigre14/", meas_v=[0, 1, 12, 7, 11, 14],
        meas_pf=[0, 10], dev=torch.device("cuda:0"),
    ),
    "diag_pantagruel": dict(
        legacy_exp="E6_pantagruel", dataset_id="pantagruel_14",
        folder=os.path.join(EXP, "data", "pantagruel_14_case1", ""),
        meas_v=[0, 1, 3, 6, 8, 11], meas_pf=[0, 4, 9],
        dev=torch.device("cuda:1"),
    ),
}
MODELS = ["supervised_ann", "modal_supervised_ann"]
NOISES = ["default", "mixture_010"]
EPOCHS, N_REPEATS, BANDWIDTH = 200, 5, 1.0
OUT_DIR = os.path.join(EXP, "raw_results", "CORRECTED")


def run_one(key):
    ds = DATASETS[key]
    csv = os.path.join(OUT_DIR, f"diagnostic_{key}.csv")
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(csv):
        os.remove(csv)
    writer = ResultWriter(csv)
    t_start = time.time()

    # corrected pipeline, SAME experiment_id as legacy => same split seed
    train_ds, test_ds, x_mean, x_std, pflow_mean, pflow_std, state_scale = \
        build_dataset_train_only(ds["folder"], ds["meas_v"], ds["meas_pf"],
                                 experiment_id=ds["legacy_exp"])
    corrupted = corrupt_test_sets(test_ds, NOISES, x_mean, x_std,
                                  pflow_mean, pflow_std)
    print(f"[{key}] train={len(train_ds)} test={len(test_ds)} "
          f"device={ds['dev']}", flush=True)

    for mid in MODELS:
        m = MODEL_REGISTRY[mid]
        for rep in range(N_REPEATS):
            t0 = time.time()
            status, model, train_loss = STATUS_SUCCESS, None, np.nan
            fail_epoch, fail_reason = np.nan, ""
            try:
                model, train_loss, _ = train_learning_model(
                    mid, train_ds, epochs=EPOCHS, batch_size=64,
                    device=ds["dev"], x_mean=x_mean, x_std=x_std,
                    pflow_mean=pflow_mean, pflow_std=pflow_std,
                    state_scale=state_scale, repeat_id=rep,
                    bandwidth=BANDWIDTH)
            except TrainingDivergedError as e:
                status, fail_epoch, fail_reason = \
                    STATUS_TRAINING_FAILURE, e.epoch, str(e)
            except Exception as e:
                status, fail_epoch, fail_reason = \
                    STATUS_TECHNICAL_ERROR, np.nan, f"{type(e).__name__}: {e}"
            metrics = None
            if status == STATUS_SUCCESS:
                try:
                    metrics = evaluate_learning_model(
                        model, mid, corrupted, x_mean, x_std, state_scale,
                        ds["dev"])
                except Exception as e:
                    status, fail_reason = \
                        STATUS_EVALUATION_FAILURE, f"{type(e).__name__}: {e}"
            base = {
                "experiment_id": f"2C_diag_{key}", "dataset_id": ds["dataset_id"],
                "data_source": f"corrected_{key}", "external_benchmark":
                    bool("pantagruel" in key),
                "model": mid, "model_name": m["model_name"],
                "architecture": m["architecture"], "loss": m["loss"],
                "bad_data_rate": 0.0, "bandwidth": BANDWIDTH,
                "bandwidth_schedule": "fixed", "lambda_meas": 0.0,
                "eta": 3e-3, "training_seed": training_seed(rep),
                "data_seed": np.nan, "epochs": EPOCHS,
                "device": str(ds["dev"]), "gpu_name": gpu_name(),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            }
            if status == STATUS_SUCCESS:
                for noise in NOISES:
                    met = metrics[noise]
                    row = dict(base)
                    row.update({
                        "noise": noise, "repeat_id": rep,
                        "eval_noise_seed": eval_noise_seed(noise),
                        "best_epoch": EPOCHS, "train_loss": train_loss,
                        "val_loss": np.nan, "v_rmse": met["v_rmse"],
                        "th_rmse": met["th_rmse"],
                        "loading_rmse": met["loading_rmse"],
                        "modal_score": met["modal_score"],
                        "hit_rate": met["hit_rate"],
                        "runtime_seconds": met["runtime_seconds"],
                        "converged": 1.0, "status": STATUS_SUCCESS,
                        "failure_epoch": np.nan, "failure_reason": "",
                        "v_mae": met["v_mae"], "th_mae": met["th_mae"],
                        "state_loss": np.nan, "meas_loss": np.nan,
                        "notes": "preprocessing_version=train_only; "
                                 f"train {time.time()-t0:.0f}s",
                    })
                    writer.append(row)
                print(f"[{key} {mid} rep{rep}] SUCCESS "
                      f"v(def)={metrics['default']['v_rmse']:.5f} "
                      f"v(mix010)={metrics['mixture_010']['v_rmse']:.5f} "
                      f"elapsed={time.time()-t_start:.0f}s", flush=True)
            else:
                for noise in NOISES:
                    row = dict(base)
                    row.update({
                        "noise": noise, "repeat_id": rep,
                        "eval_noise_seed": eval_noise_seed(noise),
                        "best_epoch": np.nan, "train_loss": np.nan,
                        "val_loss": np.nan, "v_rmse": np.nan,
                        "th_rmse": np.nan, "loading_rmse": np.nan,
                        "modal_score": np.nan, "hit_rate": np.nan,
                        "runtime_seconds": np.nan, "converged": 0.0,
                        "status": status, "failure_epoch": fail_epoch,
                        "failure_reason": fail_reason, "v_mae": np.nan,
                        "th_mae": np.nan, "state_loss": np.nan,
                        "meas_loss": np.nan,
                        "notes": "preprocessing_version=train_only",
                    })
                    writer.append(row)
                print(f"[{key} {mid} rep{rep}] {status} ({fail_reason})",
                      flush=True)
            del model
            torch.cuda.empty_cache()
    print(f"[{key}] DONE in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", choices=list(DATASETS), default=None)
    args = ap.parse_args()
    if args.key:
        run_one(args.key)
    else:
        for k in DATASETS:
            run_one(k)
