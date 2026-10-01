"""
E2: Oberrhein-70 fair comparison (Stage 2B-E2).

Design (EXPERIMENT_PROTOCOL.md E2):
  4 learning models x 5 noise conditions x 5 independent training runs
  = 100 experiment units.  100 epochs, fixed b=1.0, lambda_meas=0.
  ANN pair and DSS2 pair share identical architectures; the only intended
  difference is the loss formulation.

Failure classification (E2 protocol §12-13):
  SUCCESS / TRAINING_FAILURE (non-finite loss) / EVALUATION_FAILURE /
  TECHNICAL_ERROR.  A failed training run still writes one row per noise
  condition (NaN metrics, status=TRAINING_FAILURE) so the 100-cell grid
  stays complete; failure rate is a first-class result.

Usage:
  python run_e2.py --preflight
  python run_e2.py --chunk ann     # cuda:0, S-ANN + M-ANN  (parallel-safe)
  python run_e2.py --chunk dss2    # cuda:1, S-DSS2 + M-DSS2
  python run_e2.py --finalize      # merge chunks -> audits, tables, figures
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
    MODEL_REGISTRY, NOISE_SETTINGS,
    training_seed, data_seed, eval_noise_seed,
    load_and_split, corrupt_test_sets, make_model,
    train_learning_model, evaluate_learning_model,
    ResultWriter, summarize, paired_bootstrap, write_manifest,
    gpu_name, RAW_COLUMNS,
    STATUS_SUCCESS, STATUS_TRAINING_FAILURE,
    STATUS_EVALUATION_FAILURE, STATUS_TECHNICAL_ERROR, TrainingDivergedError,
)

EXPERIMENT_ID = "E2_ober70_fair"
DATASET_ID = "ober_sub"
FOLDER = "data/ober_sub/"
MEAS_V = [35, 16, 52, 47, 6, 48, 59, 27, 37, 56]
MEAS_PF = [40, 43, 11, 21, 54, 57]
E2_NOISES = ["default", "student_t3", "mixture_010", "erroneous", "missing_erroneous"]
EPOCHS = 100
N_REPEATS = 5
MAX_TRAIN, MAX_TEST = 500, 100

CHUNK_MODELS = {
    "ann": ["supervised_ann", "modal_supervised_ann"],
    "dss2": ["supervised_dss2", "modal_supervised_dss2"],
}
BATCH = {"ann": 64, "dss2": 16}
DEV = {"ann": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
       "dss2": torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cpu")}

RAW_DIR = os.path.join(EXP, "raw_results", "E2")
RAW_CSV = os.path.join(RAW_DIR, "E2_raw_results.csv")
CKPT_DIR = os.path.join(EXP, "checkpoints", "E2")
SUM_DIR = os.path.join(EXP, "summaries", "E2")
TAB_DIR = os.path.join(EXP, "tables", "E2")
FIG_DIR = os.path.join(EXP, "figures", "E2")
MAN_DIR = os.path.join(EXP, "manifests", "E2")


def chunk_csv(chunk):
    return os.path.join(RAW_DIR, f"E2_raw_{chunk}.csv")


def failure_row(model_id, noise, repeat_id, status, reason, epoch_fail,
                device, notes=""):
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
        "epochs": EPOCHS, "best_epoch": np.nan, "train_loss": np.nan,
        "val_loss": np.nan, "v_rmse": np.nan, "th_rmse": np.nan,
        "loading_rmse": np.nan, "modal_score": np.nan, "hit_rate": np.nan,
        "runtime_seconds": np.nan, "converged": 0.0, "device": str(device),
        "gpu_name": gpu_name(), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "status": status,
        "failure_epoch": epoch_fail, "failure_reason": reason,
        "notes": notes,
    }


def success_row(model_id, noise, repeat_id, train_loss, metrics, device, notes=""):
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
        "epochs": EPOCHS, "best_epoch": EPOCHS, "train_loss": train_loss,
        "val_loss": np.nan,
        "v_rmse": metrics["v_rmse"], "th_rmse": metrics["th_rmse"],
        "loading_rmse": metrics["loading_rmse"],
        "modal_score": metrics["modal_score"], "hit_rate": metrics["hit_rate"],
        "runtime_seconds": metrics["runtime_seconds"],
        "converged": 1.0, "device": str(device), "gpu_name": gpu_name(),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "status": STATUS_SUCCESS, "failure_epoch": np.nan,
        "failure_reason": "", "notes": notes,
    }


def preflight():
    print("=" * 70)
    print("E2 PREFLIGHT")
    print("=" * 70)
    ok = True

    print("\n[A] Dataset")
    train_ds, test_ds, x_mean, x_std, pflow_mean, pflow_std, state_scale = \
        load_and_split(FOLDER, MEAS_V, MEAS_PF, EXPERIMENT_ID,
                       max_train=MAX_TRAIN, max_test=MAX_TEST)
    g0 = train_ds[0]
    print(f"    train={len(train_ds)}, test={len(test_ds)}, "
          f"state_scale={[round(float(v), 4) for v in state_scale]}")
    print(f"    node x shape={tuple(g0.x.shape)} (expect (70,11)), "
          f"y={tuple(g0.y.shape)} (expect (70,2)), "
          f"edge_attr={tuple(g0.edge_attr.shape)}, edges={g0.edge_index.shape[1]}")
    ok &= g0.x.shape[1] == 11 and g0.y.shape[1] == 2

    print("\n[B] Architecture equality (S == M within each pair)")
    for chunk, mids in CHUNK_MODELS.items():
        dev = DEV[chunk]
        p0 = sum(p.numel() for p in make_model(mids[0], dev).parameters())
        p1 = sum(p.numel() for p in make_model(mids[1], dev).parameters())
        same = p0 == p1
        ok &= same
        print(f"    [{chunk}] {mids[0]} params={p0}, {mids[1]} params={p1}, "
              f"equal={same}")

    print("\n[C] Seeds")
    ts = [training_seed(r) for r in range(N_REPEATS)]
    print(f"    training_seeds={ts} unique={len(set(ts)) == len(ts)}")
    es = {n: eval_noise_seed(n) for n in E2_NOISES}
    print(f"    eval_noise_seeds per noise={es} "
          f"(shared across models -> paired)")
    print(f"    data_seed={data_seed(EXPERIMENT_ID)}")

    print("\n[D] Output isolation")
    for p in [chunk_csv("ann"), chunk_csv("dss2"), RAW_CSV, CKPT_DIR,
              SUM_DIR, TAB_DIR, FIG_DIR, MAN_DIR]:
        print(f"    {p}")
    collide = any(os.path.exists(chunk_csv(c)) for c in CHUNK_MODELS)
    print(f"    chunk CSV collision: {collide}")
    if collide:
        ok = False
    print(f"    E1 files untouched (raw_results/E1 separate dir): "
          f"{not RAW_DIR.startswith(os.path.join(EXP, 'raw_results', 'E1'))}")

    print("\n[E] Failure-handling sanity")
    try:
        raise TrainingDivergedError(7, float("nan"))
    except TrainingDivergedError as e:
        print(f"    TrainingDivergedError carries epoch/loss: "
              f"epoch={e.epoch}, loss={e.loss}")

    print("\n[F] Expected workload")
    n = len(CHUNK_MODELS["ann"]) + len(CHUNK_MODELS["dss2"])
    print(f"    {n} models x {len(E2_NOISES)} noises x {N_REPEATS} repeats "
          f"= {n * len(E2_NOISES) * N_REPEATS} rows")
    print(f"    checkpoints: {n * N_REPEATS} (success only)")

    print("\n" + "=" * 70)
    print(f"E2 PREFLIGHT: {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    return ok


def run_chunk(chunk):
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    csv = chunk_csv(chunk)
    if os.path.exists(csv):
        os.remove(csv)
    writer = ResultWriter(csv)
    dev = DEV[chunk]
    t_start = time.time()

    train_ds, test_ds, x_mean, x_std, pflow_mean, pflow_std, state_scale = \
        load_and_split(FOLDER, MEAS_V, MEAS_PF, EXPERIMENT_ID,
                       max_train=MAX_TRAIN, max_test=MAX_TEST)
    corrupted = corrupt_test_sets(test_ds, E2_NOISES, x_mean, x_std,
                                  pflow_mean, pflow_std)
    print(f"[chunk {chunk}] train={len(train_ds)} test={len(test_ds)} "
          f"device={dev}", flush=True)

    for mid in CHUNK_MODELS[chunk]:
        for rep in range(N_REPEATS):
            t0 = time.time()
            status, model, train_loss, fail_epoch, fail_reason = \
                STATUS_SUCCESS, None, np.nan, np.nan, ""
            try:
                model, train_loss, _ = train_learning_model(
                    mid, train_ds, epochs=EPOCHS, batch_size=BATCH[chunk],
                    device=dev, x_mean=x_mean, x_std=x_std,
                    pflow_mean=pflow_mean, pflow_std=pflow_std,
                    state_scale=state_scale, repeat_id=rep)
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
                        model, mid, corrupted, x_mean, x_std, state_scale, dev)
                except Exception as e:
                    status, fail_reason = \
                        STATUS_EVALUATION_FAILURE, f"{type(e).__name__}: {e}"

            if status == STATUS_SUCCESS:
                ckpt = os.path.join(CKPT_DIR, f"{mid}_rep{rep}.pt")
                torch.save(model.state_dict(), ckpt)
                for noise in E2_NOISES:
                    writer.append(success_row(
                        mid, noise, rep, train_loss, metrics[noise], dev,
                        f"independent training run; train {time.time()-t0:.0f}s"))
                print(f"[{mid} rep{rep}] SUCCESS train={time.time()-t0:.0f}s "
                      f"v(default)={metrics['default']['v_rmse']:.5f} "
                      f"v(mix010)={metrics['mixture_010']['v_rmse']:.5f} "
                      f"elapsed={time.time()-t_start:.0f}s", flush=True)
            else:
                for noise in E2_NOISES:
                    writer.append(failure_row(
                        mid, noise, rep, status, fail_reason, fail_epoch, dev,
                        f"epochs_attempted={fail_epoch if not np.isnan(fail_epoch) else 'n/a'}; "
                        f"run wall {time.time()-t0:.0f}s"))
                print(f"[{mid} rep{rep}] {status} ({fail_reason}) "
                      f"elapsed={time.time()-t_start:.0f}s", flush=True)
            del model
            torch.cuda.empty_cache()

    print(f"[chunk {chunk}] DONE in {(time.time()-t_start)/60:.1f} min", flush=True)


def finalize():
    dfs = []
    for c in CHUNK_MODELS:
        p = chunk_csv(c)
        if not os.path.exists(p):
            print(f"MISSING chunk file {p} — run --chunk {c} first")
            return
        dfs.append(pd.read_csv(p))
    df = pd.concat(dfs, ignore_index=True)
    df = df.reindex(columns=RAW_COLUMNS)
    os.makedirs(RAW_DIR, exist_ok=True)
    if os.path.exists(RAW_CSV):
        os.remove(RAW_CSV)
    df.to_csv(RAW_CSV, index=False)
    print(f"merged {len(df)} rows -> {RAW_CSV}")

    os.makedirs(SUM_DIR, exist_ok=True)
    os.makedirs(MAN_DIR, exist_ok=True)
    summarize(RAW_CSV, os.path.join(SUM_DIR, "E2_summary.csv"))
    pairs = [("supervised_ann", "modal_supervised_ann", "S-ANN", "M-ANN"),
             ("supervised_dss2", "modal_supervised_dss2", "S-DSS2", "M-DSS2")]
    paired_bootstrap(RAW_CSV, os.path.join(SUM_DIR, "E2_bootstrap.csv"),
                     metric="v_rmse", pairs=pairs)
    write_manifest(
        EXPERIMENT_ID, MAN_DIR,
        {"dataset": DATASET_ID, "folder": FOLDER, "meas_v": MEAS_V,
         "meas_pf": MEAS_PF, "epochs": EPOCHS, "n_repeats": N_REPEATS,
         "noise_settings": E2_NOISES, "models": CHUNK_MODELS,
         "bandwidth": 1.0, "lambda_meas": 0.0, "lambda_phys": 1.0,
         "batch": BATCH, "max_train": MAX_TRAIN, "max_test": MAX_TEST,
         "seed_policy": "training_seed=100+repeat; data_seed=500+md5(exp); eval_noise_seed=700+noise_index",
         "failure_policy": "non-finite loss -> TRAINING_FAILURE row per noise; no run dropped"},
        notes="E2 Oberrhein-70 fair comparison; architecture-controlled; failure-aware")
    print("summaries + manifest written; table/figure generation follows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--chunk", choices=["ann", "dss2"], default=None)
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    if args.preflight:
        sys.exit(0 if preflight() else 1)
    if args.chunk:
        run_chunk(args.chunk)
    elif args.finalize:
        finalize()
    else:
        ap.print_help()
