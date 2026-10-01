"""
E1: CIGRE-14 main experiment (Stage 2B-E1).

Design (EXPERIMENT_PROTOCOL.md E1):
  learning models: supervised_ann, modal_supervised_ann,
                   supervised_dss2, modal_supervised_dss2
      10 noise conditions x 5 independent training runs x 200 epochs
  solvers:         wls, modal_wls
      10 noise conditions x 5 noise draws x 10 fixed test samples
      (no training; train_loss is legitimately NaN)

Outputs (isolated from smoke files):
  raw_results/E1/E1_raw_results.csv      (220 rows + header)
  checkpoints/E1/{model}_rep{r}.pt       (20 checkpoints)
  summaries/E1/E1_summary.csv, E1_bootstrap.csv
  manifests/E1/RUN_MANIFEST_E1.json

Usage:
  python run_e1.py --preflight     # checks only, no full run
  python run_e1.py                 # full E1 run
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from framework import (  # noqa: E402
    MODEL_REGISTRY, NOISE_SETTINGS, NOISE_INDEX,
    training_seed, data_seed, eval_noise_seed, seal_environment,
    load_and_split, corrupt_test_sets, make_model,
    train_learning_model, evaluate_learning_model,
    evaluate_solver_model, build_wls_samples,
    ResultWriter, summarize, paired_bootstrap, write_manifest,
    gpu_name, RAW_COLUMNS,
)

EXPERIMENT_ID = "E1_cigre14_main"
DATASET_ID = "cigre14"
FOLDER = "data/cigre14/"
MEAS_V = [0, 1, 12, 7, 11, 14]
MEAS_PF = [0, 10]
EPOCHS = 200
N_REPEATS = 5
LEARNING_MODELS_E1 = ["supervised_ann", "modal_supervised_ann",
                      "supervised_dss2", "modal_supervised_dss2"]
SOLVER_MODELS_E1 = ["wls", "modal_wls"]

RAW_CSV = os.path.join(EXP, "raw_results", "E1", "E1_raw_results.csv")
CKPT_DIR = os.path.join(EXP, "checkpoints", "E1")
SUM_DIR = os.path.join(EXP, "summaries", "E1")
MAN_DIR = os.path.join(EXP, "manifests", "E1")

DEV_ANN = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEV_GNN = torch.device("cuda:1" if torch.cuda.device_count() > 1
                       else (DEV_ANN if torch.cuda.is_available() else "cpu"))


def row_for(model_id, noise, repeat_id, epochs, train_loss, metrics, device,
            notes):
    m = MODEL_REGISTRY[model_id]
    return {
        "experiment_id": EXPERIMENT_ID, "dataset_id": DATASET_ID,
        "model": model_id, "model_name": m["model_name"],
        "architecture": m["architecture"], "loss": m["loss"],
        "noise": noise, "bad_data_rate": 0.0, "bandwidth": 1.0,
        "bandwidth_schedule": "fixed", "lambda_meas": 0.0, "eta": 3e-3,
        "repeat_id": repeat_id, "training_seed": training_seed(repeat_id),
        "data_seed": data_seed(EXPERIMENT_ID),
        "eval_noise_seed": eval_noise_seed(noise),
        "epochs": epochs, "best_epoch": epochs,
        "train_loss": train_loss, "val_loss": np.nan,
        "v_rmse": metrics["v_rmse"], "th_rmse": metrics["th_rmse"],
        "loading_rmse": metrics["loading_rmse"],
        "modal_score": metrics["modal_score"], "hit_rate": metrics["hit_rate"],
        "runtime_seconds": metrics["runtime_seconds"],
        "converged": metrics.get("converged", 1.0),
        "device": str(device), "gpu_name": gpu_name(),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "notes": notes,
    }


def preflight():
    print("=" * 70)
    print("E1 PREFLIGHT (checks only — no full experiment)")
    print("=" * 70)
    ok = True

    # 1. model instantiation
    print("\n[1] Model instantiation from registry")
    for mid in LEARNING_MODELS_E1:
        dev = DEV_ANN if mid.endswith("ann") else DEV_GNN
        try:
            mdl = make_model(mid, dev)
            n = sum(p.numel() for p in mdl.parameters())
            print(f"    [PASS] {mid}: instantiated on {dev}, params={n}")
        except Exception as e:
            ok = False
            print(f"    [FAIL] {mid}: {type(e).__name__}: {e}")

    # 2. short training loops (2 epochs on a 50-graph subset)
    print("\n[2] Training-loop capability (2 epochs, 50-graph subset)")
    np.random.seed(data_seed(EXPERIMENT_ID))
    train_ds, test_ds, x_mean, x_std, pflow_mean, pflow_std, state_scale = \
        load_and_split(FOLDER, MEAS_V, MEAS_PF, EXPERIMENT_ID,
                       max_train=50, max_test=10)
    for mid in LEARNING_MODELS_E1:
        dev = DEV_ANN if mid.endswith("ann") else DEV_GNN
        try:
            mdl, loss_val, hist = train_learning_model(
                mid, train_ds, epochs=2,
                batch_size=32 if mid.endswith("dss2") else 64,
                device=dev, x_mean=x_mean, x_std=x_std,
                pflow_mean=pflow_mean, pflow_std=pflow_std,
                state_scale=state_scale, repeat_id=0)
            finite = np.isfinite(loss_val)
            print(f"    [{'PASS' if finite else 'FAIL'}] {mid}: 2-epoch loss="
                  f"{loss_val:.5f}, finite={finite}")
            ok &= finite
        except Exception as e:
            ok = False
            print(f"    [FAIL] {mid}: {type(e).__name__}: {e}")

    # 3. solver branch
    print("\n[3] Solver branch (1 sample, 1 draw per solver)")
    try:
        samples = build_wls_samples(FOLDER, MEAS_V, n_samples=10)
        for mid in SOLVER_MODELS_E1:
            metrics = evaluate_solver_model(mid, samples, ["default"], n_draws=1)
            print(f"    [PASS] {mid}: v_rmse={metrics['default']['v_rmse']:.5f}, "
                  f"converged={metrics['default'].get('converged', np.nan)}")
    except Exception as e:
        ok = False
        print(f"    [FAIL] solver branch: {type(e).__name__}: {e}")

    # 4. seed independence
    print("\n[4] Seed independence")
    ts = [training_seed(r) for r in range(N_REPEATS)]
    print(f"    training_seeds = {ts} (unique: {len(set(ts)) == len(ts)})")
    ok &= len(set(ts)) == len(ts)
    print(f"    data_seed(E1) = {data_seed(EXPERIMENT_ID)} "
          f"(deterministic: {data_seed(EXPERIMENT_ID) == data_seed(EXPERIMENT_ID)})")
    es = {n: eval_noise_seed(n) for n in NOISE_SETTINGS}
    print(f"    eval_noise_seeds unique per noise: "
          f"{len(set(es.values())) == len(NOISE_SETTINGS)}")

    # 5. shared evaluation noise
    print("\n[5] Shared evaluation noise (paired design)")
    rng1 = np.random.RandomState(eval_noise_seed("mixture_010"))
    rng2 = np.random.RandomState(eval_noise_seed("mixture_010"))
    a = rng1.randn(5); b = rng2.randn(5)
    print(f"    identical draws for same noise seed: {np.allclose(a, b)}")
    ok &= bool(np.allclose(a, b))
    print("    (corrupt_test_sets is called ONCE per noise and the returned "
          "tensors are shared by all models/repeats — structural guarantee)")

    # 6. output isolation + schema
    print("\n[6] Output paths and schema")
    for p in [RAW_CSV, CKPT_DIR, SUM_DIR, MAN_DIR]:
        print(f"    {p}")
    collide = os.path.exists(RAW_CSV)
    print(f"    raw CSV already exists (would append): {collide}")
    if collide:
        ok = False
        print("    [FAIL] E1 raw CSV must start fresh — delete/rename before run")
    else:
        print("    [PASS] fresh E1 raw CSV")
    print(f"    schema columns: {len(RAW_COLUMNS)} "
          f"(contains noise/repeat/training_seed/data_seed/eval_noise_seed/"
          f"lambda_meas/bandwidth/runtime: "
          f"{all(c in RAW_COLUMNS for c in ['noise','repeat_id','training_seed','data_seed','eval_noise_seed','lambda_meas','bandwidth','runtime_seconds'])})")

    # 7. expected size
    print("\n[7] Expected workload")
    n_learn = len(LEARNING_MODELS_E1) * len(NOISE_SETTINGS) * N_REPEATS
    n_solv = len(SOLVER_MODELS_E1) * len(NOISE_SETTINGS)
    print(f"    learning rows: {n_learn}, solver rows: {n_solv}, "
          f"total: {n_learn + n_solv}")
    print(f"    checkpoints: {len(LEARNING_MODELS_E1) * N_REPEATS}")
    print(f"    devices: ANN={DEV_ANN}, GNN={DEV_GNN}")

    print("\n" + "=" * 70)
    print(f"PREFLIGHT: {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    return ok


def run_full():
    t_start = time.time()
    os.makedirs(os.path.dirname(RAW_CSV), exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(SUM_DIR, exist_ok=True)
    os.makedirs(MAN_DIR, exist_ok=True)
    if os.path.exists(RAW_CSV):
        os.remove(RAW_CSV)   # fresh file for a fresh run (preflight checked)

    writer = ResultWriter(RAW_CSV)

    train_ds, test_ds, x_mean, x_std, pflow_mean, pflow_std, state_scale = \
        load_and_split(FOLDER, MEAS_V, MEAS_PF, EXPERIMENT_ID)
    print(f"E1: train={len(train_ds)}, test={len(test_ds)}, "
          f"state_scale={state_scale.tolist()}")

    corrupted = corrupt_test_sets(test_ds, NOISE_SETTINGS, x_mean, x_std,
                                  pflow_mean, pflow_std)

    # ---- learning models: 5 independent training runs each ---------------
    for mid in LEARNING_MODELS_E1:
        dev = DEV_ANN if mid.endswith("ann") else DEV_GNN
        for rep in range(N_REPEATS):
            t0 = time.time()
            model, train_loss, _ = train_learning_model(
                mid, train_ds, epochs=EPOCHS,
                batch_size=32 if mid.endswith("dss2") else 64,
                device=dev, x_mean=x_mean, x_std=x_std,
                pflow_mean=pflow_mean, pflow_std=pflow_std,
                state_scale=state_scale, repeat_id=rep)
            train_t = time.time() - t0
            metrics = evaluate_learning_model(
                model, mid, corrupted, x_mean, x_std, state_scale, dev)
            ckpt = os.path.join(CKPT_DIR, f"{mid}_rep{rep}.pt")
            torch.save(model.state_dict(), ckpt)
            for noise in NOISE_SETTINGS:
                writer.append(row_for(
                    mid, noise, rep, EPOCHS, train_loss, metrics[noise], dev,
                    f"independent training run; train {train_t:.0f}s"))
            print(f"[{mid} rep{rep}] train={train_t:.0f}s, "
                  f"v_rmse(default)={metrics['default']['v_rmse']:.5f}, "
                  f"v_rmse(mixture_010)={metrics['mixture_010']['v_rmse']:.5f}, "
                  f"elapsed={time.time()-t_start:.0f}s")
            del model
            torch.cuda.empty_cache()

    # ---- solvers: 5 draws x 10 samples per noise condition ---------------
    samples = build_wls_samples(FOLDER, MEAS_V, n_samples=10)
    for mid in SOLVER_MODELS_E1:
        t0 = time.time()
        metrics = evaluate_solver_model(mid, samples, NOISE_SETTINGS, n_draws=5)
        for noise in NOISE_SETTINGS:
            writer.append(row_for(
                mid, noise, -1, 0, np.nan, metrics[noise], "cpu",
                "per-sample solver; 5 noise draws x 10 samples"))
        print(f"[{mid}] {time.time()-t0:.0f}s, "
              f"v_rmse(mixture_010)={metrics['mixture_010']['v_rmse']:.5f}, "
              f"conv(default)={metrics['default'].get('converged', np.nan):.2f}")

    # ---- summaries --------------------------------------------------------
    summarize(RAW_CSV, os.path.join(SUM_DIR, "E1_summary.csv"))
    pairs = [
        ("supervised_ann", "modal_supervised_ann", "S-ANN", "M-ANN"),
        ("supervised_dss2", "modal_supervised_dss2", "S-DSS2", "M-DSS2"),
        ("wls", "modal_wls", "WLS", "Modal-WLS"),
    ]
    paired_bootstrap(RAW_CSV, os.path.join(SUM_DIR, "E1_bootstrap.csv"),
                     metric="v_rmse", pairs=pairs)

    write_manifest(
        EXPERIMENT_ID, MAN_DIR,
        {"dataset": DATASET_ID, "folder": FOLDER, "meas_v": MEAS_V,
         "meas_pf": MEAS_PF, "epochs": EPOCHS, "n_repeats": N_REPEATS,
         "noise_settings": NOISE_SETTINGS, "models": LEARNING_MODELS_E1 + SOLVER_MODELS_E1,
         "bandwidth": 1.0, "lambda_meas": 0.0, "lambda_phys": 1.0,
         "batch": {"ann": 64, "dss2": 32},
         "seed_policy": "training_seed=100+repeat; data_seed=500+md5(exp); eval_noise_seed=700+noise_index"},
        notes="E1 CIGRE-14 main experiment; independent training runs; shared eval noise per noise condition")
    print(f"\nE1 DONE in {(time.time()-t_start)/60:.1f} min; raw CSV: {RAW_CSV}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()
    if args.preflight:
        sys.exit(0 if preflight() else 1)
    run_full()
