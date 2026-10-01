"""
E6: cross-system + external-benchmark validation (Stage 2B-E6).

E6-A (chunk ieee30, cuda:0): IEEE-30, 4 learning models (S-ANN, M-ANN,
  S-DSS2, M-DSS2), 3 noise conditions {default, mixture_010, erroneous},
  5 independent training runs, 200 epochs. Main protocol: b=1.0,
  lambda_meas=0 (E4's lambda finding is NOT used to retune E6).

E6-B (chunk pantagruel, cuda:1): PanTaGruEl IEEE-14 external benchmark
  (Zenodo 5816149, Level-3), S-ANN vs M-ANN, same 3 noise conditions,
  5 independent runs, 200 epochs, b=1.0.

Usage:
  python run_e6.py --preflight
  python run_e6.py --smoke
  python run_e6.py --chunk ieee30|pantagruel
  python run_e6.py --finalize
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
    MODEL_REGISTRY, training_seed, data_seed, eval_noise_seed,
    load_and_split, corrupt_test_sets, train_learning_model,
    evaluate_learning_model, ResultWriter, summarize, paired_bootstrap,
    write_manifest, gpu_name, RAW_COLUMNS,
    STATUS_SUCCESS, STATUS_TRAINING_FAILURE,
    STATUS_EVALUATION_FAILURE, STATUS_TECHNICAL_ERROR, TrainingDivergedError,
)

EXPERIMENT_ID = "E6_crosssystem_external"
E6_NOISES = ["default", "mixture_010", "erroneous"]
EPOCHS = 200
N_REPEATS = 5
BANDWIDTH = 1.0
LAMBDA_MEAS = 0.0

DATASETS = {
    "ieee30": dict(
        dataset_id="ieee30",
        data_source="ieee30_internal",
        external_benchmark=False,
        folder=os.path.join(EXP, "data", "ieee30", ""),
        meas_v=[0, 2, 5, 9, 14, 19, 24, 27, 29],
        meas_pf=[0, 4, 8, 12, 16, 20],
        models=["supervised_ann", "modal_supervised_ann",
                "supervised_dss2", "modal_supervised_dss2"],
        batch={"ann": 64, "dss2": 32},
        dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        note="cross-system benchmark, pandapower case30, project-generated data",
    ),
    "pantagruel": dict(
        dataset_id="pantagruel_14",
        data_source="pantagruel_external",
        external_benchmark=True,
        folder=os.path.join(EXP, "data", "pantagruel_14_case1", ""),
        meas_v=[0, 1, 3, 6, 8, 11],
        meas_pf=[0, 4, 9],
        models=["supervised_ann", "modal_supervised_ann"],
        batch={"ann": 64, "dss2": 32},
        dev=torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cpu"),
        note="external benchmark, Zenodo 5816149 (Level-3, CC-BY-4.0)",
    ),
}

RAW_DIR = os.path.join(EXP, "raw_results", "E6")
RAW_CSV = os.path.join(RAW_DIR, "E6_raw_results.csv")
CKPT_DIR = os.path.join(EXP, "checkpoints", "E6")
SUM_DIR = os.path.join(EXP, "summaries", "E6")
TAB_DIR = os.path.join(EXP, "tables", "E6")
FIG_DIR = os.path.join(EXP, "figures", "E6")
MAN_DIR = os.path.join(EXP, "manifests", "E6")


def chunk_csv(c):
    return os.path.join(RAW_DIR, f"E6_raw_{c}.csv")


def preflight():
    print("=" * 70)
    print("E6 PREFLIGHT")
    print("=" * 70)
    ok = True
    for key, ds in DATASETS.items():
        print(f"\n[{key}] dataset={ds['dataset_id']}, "
              f"external={ds['external_benchmark']}")
        train_ds, test_ds, x_mean, x_std, pflow_mean, pflow_std, state_scale = \
            load_and_split(ds["folder"], ds["meas_v"], ds["meas_pf"],
                           experiment_id=f"E6_{key}")
        g0 = train_ds[0]
        n_bus = g0.y.shape[0]
        print(f"    train={len(train_ds)}, test={len(test_ds)}, "
              f"x={tuple(g0.x.shape)}, y={tuple(g0.y.shape)}, "
              f"edges={g0.edge_index.shape[1]}, "
              f"state_scale={[round(float(v),4) for v in state_scale]}")
        print(f"    models: {ds['models']}")
        ok &= g0.x.shape[1] == 11 and g0.y.shape[1] == 2
    print("\n[seeds] training_seed=100+repeat; eval_noise_seed=700+idx "
          "(shared across models per noise)")
    print(f"[frozen] b={BANDWIDTH}, lambda_meas={LAMBDA_MEAS} "
          f"(main protocol; E4's lambda finding NOT used to retune)")
    print(f"[workload] ieee30: 4x3x5=60 rows; pantagruel: 2x3x5=30 rows; "
          f"total 90")
    collide = any(os.path.exists(chunk_csv(c)) for c in DATASETS)
    print(f"[isolation] chunk CSV collision: {collide}")
    ok &= not collide
    print("=" * 70)
    print(f"E6 PREFLIGHT: {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    return ok


def run_smoke():
    print("=" * 70)
    print("E6 SMOKE (each dataset: 1 model family sample x 5 epochs)")
    print("=" * 70)
    ok = True
    for key, ds in DATASETS.items():
        train_ds, test_ds, x_mean, x_std, pflow_mean, pflow_std, state_scale = \
            load_and_split(ds["folder"], ds["meas_v"], ds["meas_pf"],
                           experiment_id=f"E6_{key}",
                           max_train=50, max_test=10)
        for mid in ds["models"]:
            try:
                model, loss_val, _ = train_learning_model(
                    mid, train_ds, epochs=5,
                    batch_size=ds["batch"]["dss2" if mid.endswith("dss2") else "ann"],
                    device=ds["dev"], x_mean=x_mean, x_std=x_std,
                    pflow_mean=pflow_mean, pflow_std=pflow_std,
                    state_scale=state_scale, repeat_id=0,
                    bandwidth=BANDWIDTH, lambda_meas=LAMBDA_MEAS)
                fin = np.isfinite(loss_val)
                print(f"    [{key} {mid}] loss={loss_val:.5f} finite={fin}")
                ok &= fin
            except Exception as e:
                ok = False
                print(f"    [{key} {mid}] FAIL: {type(e).__name__}: {e}")
    print("=" * 70)
    print(f"E6 SMOKE: {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    return ok


def run_chunk(key):
    ds = DATASETS[key]
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    csv = chunk_csv(key)
    if os.path.exists(csv):
        os.remove(csv)
    writer = ResultWriter(csv)
    t_start = time.time()

    train_ds, test_ds, x_mean, x_std, pflow_mean, pflow_std, state_scale = \
        load_and_split(ds["folder"], ds["meas_v"], ds["meas_pf"],
                       experiment_id=f"E6_{key}")
    corrupted = corrupt_test_sets(test_ds, E6_NOISES, x_mean, x_std,
                                  pflow_mean, pflow_std)
    print(f"[{key}] train={len(train_ds)} test={len(test_ds)} "
          f"device={ds['dev']}", flush=True)

    for mid in ds["models"]:
        m = MODEL_REGISTRY[mid]
        for rep in range(N_REPEATS):
            t0 = time.time()
            status, model, train_loss = STATUS_SUCCESS, None, np.nan
            fail_epoch, fail_reason = np.nan, ""
            try:
                model, train_loss, _ = train_learning_model(
                    mid, train_ds, epochs=EPOCHS,
                    batch_size=ds["batch"]["dss2" if mid.endswith("dss2") else "ann"],
                    device=ds["dev"], x_mean=x_mean, x_std=x_std,
                    pflow_mean=pflow_mean, pflow_std=pflow_std,
                    state_scale=state_scale, repeat_id=rep,
                    bandwidth=BANDWIDTH, lambda_meas=LAMBDA_MEAS)
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
                "experiment_id": EXPERIMENT_ID, "dataset_id": ds["dataset_id"],
                "data_source": ds["data_source"],
                "external_benchmark": ds["external_benchmark"],
                "model": mid, "model_name": m["model_name"],
                "architecture": m["architecture"], "loss": m["loss"],
                "bad_data_rate": 0.0, "bandwidth": BANDWIDTH,
                "bandwidth_schedule": "fixed", "lambda_meas": LAMBDA_MEAS,
                "eta": 3e-3, "training_seed": training_seed(rep),
                "data_seed": data_seed(f"E6_{key}"),
                "epochs": EPOCHS, "device": str(ds["dev"]),
                "gpu_name": gpu_name(), "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            }
            if status == STATUS_SUCCESS:
                torch.save(model.state_dict(),
                           os.path.join(CKPT_DIR, f"{key}_{mid}_rep{rep}.pt"))
                for noise in E6_NOISES:
                    met = metrics[noise]
                    row = dict(base)
                    row.update({
                        "noise": noise, "repeat_id": rep,
                        "eval_noise_seed": eval_noise_seed(noise),
                        "best_epoch": EPOCHS, "train_loss": train_loss,
                        "val_loss": np.nan,
                        "v_rmse": met["v_rmse"], "th_rmse": met["th_rmse"],
                        "loading_rmse": met["loading_rmse"],
                        "modal_score": met["modal_score"],
                        "hit_rate": met["hit_rate"],
                        "runtime_seconds": met["runtime_seconds"],
                        "converged": 1.0, "status": STATUS_SUCCESS,
                        "failure_epoch": np.nan, "failure_reason": "",
                        "v_mae": met["v_mae"], "th_mae": met["th_mae"],
                        "state_loss": np.nan, "meas_loss": np.nan,
                        "notes": f"{ds['note']}; train {time.time()-t0:.0f}s",
                    })
                    writer.append(row)
                print(f"[{key} {mid} rep{rep}] SUCCESS "
                      f"train={time.time()-t0:.0f}s "
                      f"v(def)={metrics['default']['v_rmse']:.5f} "
                      f"v(mix010)={metrics['mixture_010']['v_rmse']:.5f} "
                      f"elapsed={time.time()-t_start:.0f}s", flush=True)
            else:
                for noise in E6_NOISES:
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
                        "notes": f"{ds['note']}; wall {time.time()-t0:.0f}s",
                    })
                    writer.append(row)
                print(f"[{key} {mid} rep{rep}] {status} ({fail_reason})",
                      flush=True)
            del model
            torch.cuda.empty_cache()

    print(f"[{key}] DONE in {(time.time()-t_start)/60:.1f} min", flush=True)


def finalize():
    dfs = []
    for c in DATASETS:
        p = chunk_csv(c)
        if not os.path.exists(p):
            print(f"MISSING {p} — run --chunk {c} first")
            return
        dfs.append(pd.read_csv(p))
    df = pd.concat(dfs, ignore_index=True).reindex(columns=RAW_COLUMNS)
    for d in (RAW_DIR, SUM_DIR, TAB_DIR, FIG_DIR, MAN_DIR):
        os.makedirs(d, exist_ok=True)
    if os.path.exists(RAW_CSV):
        os.remove(RAW_CSV)
    df.to_csv(RAW_CSV, index=False)
    print(f"merged {len(df)} rows -> {RAW_CSV}")

    summarize(RAW_CSV, os.path.join(SUM_DIR, "E6_summary.csv"))
    write_manifest(
        EXPERIMENT_ID, MAN_DIR,
        {"epochs": EPOCHS, "n_repeats": N_REPEATS, "noise": E6_NOISES,
         "bandwidth": BANDWIDTH, "lambda_meas": LAMBDA_MEAS,
         "datasets": {k: {"dataset_id": v["dataset_id"],
                          "external_benchmark": v["external_benchmark"],
                          "models": v["models"]} for k, v in DATASETS.items()},
         "seed_policy": "training_seed=100+repeat; data_seed=500+md5(exp); eval_noise_seed=700+idx",
         "post_hoc_tuning": "NONE — E4 lambda finding NOT used to retune E6"},
        notes="E6 cross-system (IEEE-30) + external benchmark (PanTaGruEl Level-3)")
    print("summary + manifest written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--chunk", choices=["ieee30", "pantagruel"], default=None)
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    if args.preflight:
        sys.exit(0 if preflight() else 1)
    if args.smoke:
        sys.exit(0 if run_smoke() else 1)
    if args.chunk:
        run_chunk(args.chunk)
    elif args.finalize:
        finalize()
    else:
        ap.print_help()
