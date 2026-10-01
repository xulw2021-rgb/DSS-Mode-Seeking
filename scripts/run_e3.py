"""
E3: Bandwidth sensitivity / annealing experiment (Stage 2B-E3).

Object: 14-bus CIGRE-14 modal estimators, M-ANN and M-DSS2.
Bandwidth conditions: fixed b in {0.5, 1.0, 2.0} and annealing 8 -> 2
(exponential schedule b(t)=b_end+(b_start-b_end)exp(-t/tau), tau=epochs/3,
the existing BandwidthSchedule implementation).

Matrix per model: 4 bandwidth conditions x 4 noise conditions x
5 independent training runs.  Noise does not affect training (training is
on clean data); it only affects evaluation, with eval_noise_seed shared
across bandwidth conditions (paired design).  Training seeds independent
per repeat.

Noise conditions (protocol clarification, logged): default, mixture_005,
mixture_010, mixture_020 — clean baseline + the full mixture ladder where
both robustness and stability phenomena live.

Failure handling: status vocabulary SUCCESS / TRAINING_FAILURE /
EVALUATION_FAILURE / TECHNICAL_ERROR (framework).  Note: E1-style
"collapsed evaluation" (successful training, catastrophic test error) is
NOT a protocol failure; it is analyzed separately as a performance-based
stability criterion (v_rmse > 1.0), clearly labeled.

Usage:
  python run_e3.py --preflight
  python run_e3.py --smoke
  python run_e3.py --chunk ann|dss2
  python run_e3.py --finalize
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
    MODEL_REGISTRY, training_seed, data_seed, eval_noise_seed,
    load_and_split, corrupt_test_sets, train_learning_model,
    evaluate_learning_model, ResultWriter, summarize, paired_bootstrap,
    write_manifest, gpu_name, RAW_COLUMNS,
    STATUS_SUCCESS, STATUS_TRAINING_FAILURE,
    STATUS_EVALUATION_FAILURE, STATUS_TECHNICAL_ERROR, TrainingDivergedError,
    gaussian_kernel_modal_loss, BandwidthSchedule,
)

EXPERIMENT_ID = "E3_cigre14_bandwidth"
DATASET_ID = "cigre14"
FOLDER = "data/cigre14/"
MEAS_V = [0, 1, 12, 7, 11, 14]
MEAS_PF = [0, 10]
E3_NOISES = ["default", "mixture_005", "mixture_010", "mixture_020"]
EPOCHS = 200
N_REPEATS = 5

# bandwidth conditions: key -> (bandwidth, annealing, b_start, b_end)
BANDWIDTH_CONDITIONS = {
    "b0.5": (0.5, False, 8.0, 2.0),
    "b1.0": (1.0, False, 8.0, 2.0),
    "b2.0": (2.0, False, 8.0, 2.0),
    "anneal_8_2": (2.0, True, 8.0, 2.0),
}
CHUNK_MODELS = {"ann": ["modal_supervised_ann"],
                "dss2": ["modal_supervised_dss2"]}
BATCH = {"ann": 64, "dss2": 32}
DEV = {"ann": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
       "dss2": torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cpu")}

RAW_DIR = os.path.join(EXP, "raw_results", "E3")
RAW_CSV = os.path.join(RAW_DIR, "E3_raw_results.csv")
CKPT_DIR = os.path.join(EXP, "checkpoints", "E3")
HIST_DIR = os.path.join(EXP, "training_history", "E3")
SUM_DIR = os.path.join(EXP, "summaries", "E3")
TAB_DIR = os.path.join(EXP, "tables", "E3")
FIG_DIR = os.path.join(EXP, "figures", "E3")
MAN_DIR = os.path.join(EXP, "manifests", "E3")


def chunk_csv(chunk):
    return os.path.join(RAW_DIR, f"E3_raw_{chunk}.csv")


def make_row(model_id, noise, bw_key, bw, anneal, b_start, repeat_id,
             train_loss, metrics, status, fail_epoch, fail_reason, device,
             notes=""):
    m = MODEL_REGISTRY[model_id]
    ok = status == STATUS_SUCCESS
    return {
        "experiment_id": EXPERIMENT_ID, "dataset_id": DATASET_ID,
        "model": model_id, "model_name": m["model_name"],
        "architecture": m["architecture"], "loss": m["loss"],
        "noise": noise, "bad_data_rate": 0.0, "bandwidth": bw,
        "bandwidth_schedule": "anneal_8_2" if anneal else "fixed",
        "lambda_meas": 0.0, "eta": 3e-3, "repeat_id": repeat_id,
        "training_seed": training_seed(repeat_id),
        "data_seed": data_seed(EXPERIMENT_ID),
        "eval_noise_seed": eval_noise_seed(noise),
        "epochs": EPOCHS, "best_epoch": EPOCHS if ok else np.nan,
        "train_loss": train_loss if ok else np.nan, "val_loss": np.nan,
        "v_rmse": metrics["v_rmse"] if ok else np.nan,
        "th_rmse": metrics["th_rmse"] if ok else np.nan,
        "loading_rmse": metrics["loading_rmse"] if ok else np.nan,
        "modal_score": metrics["modal_score"] if ok else np.nan,
        "hit_rate": metrics["hit_rate"] if ok else np.nan,
        "runtime_seconds": metrics["runtime_seconds"] if ok else np.nan,
        "converged": 1.0 if ok else 0.0, "device": str(device),
        "gpu_name": gpu_name(), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "status": status,
        "failure_epoch": fail_epoch, "failure_reason": fail_reason,
        "notes": (f"bw_key={bw_key}, b_start={b_start}, b_end={bw}; " + notes),
    }


def preflight():
    print("=" * 70)
    print("E3 PREFLIGHT")
    print("=" * 70)
    ok = True

    print("\n[A] Dataset (cigre14)")
    train_ds, test_ds, x_mean, x_std, pflow_mean, pflow_std, state_scale = \
        load_and_split(FOLDER, MEAS_V, MEAS_PF, EXPERIMENT_ID)
    g0 = train_ds[0]
    print(f"    train={len(train_ds)}, test={len(test_ds)}, "
          f"node x={tuple(g0.x.shape)}, y={tuple(g0.y.shape)}")
    ok &= g0.x.shape[1] == 11 and g0.y.shape[1] == 2

    print("\n[B] Bandwidth implementation verification")
    r = torch.linspace(-3, 3, 1001)
    L = {b: float(gaussian_kernel_modal_loss(r, b, reduction="mean"))
         for b in (0.5, 1.0, 2.0)}
    print(f"    L(0.5)={L[0.5]:.5f}, L(1.0)={L[1.0]:.5f}, L(2.0)={L[2.0]:.5f}")
    distinct = len(set(round(v, 6) for v in L.values())) == 3
    print(f"    losses distinct: {distinct}")
    ok &= distinct
    sched = BandwidthSchedule(8.0, 2.0, EPOCHS / 3.0)
    b0, bm, be = sched.get(0), sched.get(EPOCHS // 2), sched.get(EPOCHS - 1)
    print(f"    anneal schedule: b(0)={b0:.3f}, b({EPOCHS//2})={bm:.3f}, "
          f"b({EPOCHS-1})={be:.3f}  (target 8 -> ~2, exponential, "
          f"tau={EPOCHS/3:.1f})")
    monotone = b0 > bm > be and abs(b0 - 8.0) < 1e-6
    print(f"    annealing actually decreases 8->~2: {monotone}")
    ok &= monotone

    print("\n[C] Output isolation")
    for p in [chunk_csv("ann"), chunk_csv("dss2"), RAW_CSV, CKPT_DIR,
              HIST_DIR, SUM_DIR, TAB_DIR, FIG_DIR, MAN_DIR]:
        print(f"    {p}")
    collide = any(os.path.exists(chunk_csv(c)) for c in CHUNK_MODELS)
    print(f"    chunk CSV collision: {collide}")
    ok &= not collide

    print("\n[D] Expected workload")
    per_model = len(BANDWIDTH_CONDITIONS) * len(E3_NOISES) * N_REPEATS
    print(f"    per model: 4 bw x 4 noise x 5 repeats = {per_model} rows; "
          f"2 models -> {2 * per_model} rows")
    print(f"    training runs: 2 models x 4 bw x 5 repeats = 40; "
          f"checkpoints 40; history files 40")

    print("\n" + "=" * 70)
    print(f"E3 PREFLIGHT: {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    return ok


def run_smoke():
    print("=" * 70)
    print("E3 SMOKE (4 bandwidth conditions x 1 noise x 1 repeat x 5 epochs)")
    print("=" * 70)
    train_ds, test_ds, x_mean, x_std, pflow_mean, pflow_std, state_scale = \
        load_and_split(FOLDER, MEAS_V, MEAS_PF, EXPERIMENT_ID,
                       max_train=100, max_test=20)
    ok = True
    for bw_key, (bw, anneal, bs, be) in BANDWIDTH_CONDITIONS.items():
        for chunk, mids in CHUNK_MODELS.items():
            mid = mids[0]
            dev = DEV[chunk]
            try:
                model, loss_val, hist = train_learning_model(
                    mid, train_ds, epochs=5,
                    batch_size=BATCH[chunk], device=dev,
                    x_mean=x_mean, x_std=x_std, pflow_mean=pflow_mean,
                    pflow_std=pflow_std, state_scale=state_scale,
                    repeat_id=0, bandwidth=bw, annealing=anneal,
                    b_start=bs, b_end=be)
                b_seq = hist["b"]
                fin = np.isfinite(loss_val) and all(np.isfinite(b_seq))
                span = (max(b_seq) - min(b_seq)) if anneal else 0.0
                print(f"    [{bw_key:11s} {mid:22s}] loss={loss_val:.5f} "
                      f"finite={fin} b_range=[{min(b_seq):.2f},{max(b_seq):.2f}] "
                      f"span={span:.2f}")
                ok &= fin and (span > 0.1 if anneal else abs(span) < 1e-9)
            except Exception as e:
                ok = False
                print(f"    [{bw_key} {mid}] FAIL: {type(e).__name__}: {e}")
    print("=" * 70)
    print(f"E3 SMOKE: {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    return ok


def run_chunk(chunk):
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(HIST_DIR, exist_ok=True)
    csv = chunk_csv(chunk)
    if os.path.exists(csv):
        os.remove(csv)
    writer = ResultWriter(csv)
    dev = DEV[chunk]
    t_start = time.time()

    train_ds, test_ds, x_mean, x_std, pflow_mean, pflow_std, state_scale = \
        load_and_split(FOLDER, MEAS_V, MEAS_PF, EXPERIMENT_ID)
    corrupted = corrupt_test_sets(test_ds, E3_NOISES, x_mean, x_std,
                                  pflow_mean, pflow_std)
    print(f"[chunk {chunk}] train={len(train_ds)} test={len(test_ds)} "
          f"device={dev}", flush=True)

    for mid in CHUNK_MODELS[chunk]:
        for bw_key, (bw, anneal, bs, be) in BANDWIDTH_CONDITIONS.items():
            for rep in range(N_REPEATS):
                t0 = time.time()
                status, model, train_loss, hist = STATUS_SUCCESS, None, np.nan, None
                fail_epoch, fail_reason = np.nan, ""
                try:
                    model, train_loss, hist = train_learning_model(
                        mid, train_ds, epochs=EPOCHS, batch_size=BATCH[chunk],
                        device=dev, x_mean=x_mean, x_std=x_std,
                        pflow_mean=pflow_mean, pflow_std=pflow_std,
                        state_scale=state_scale, repeat_id=rep,
                        bandwidth=bw, annealing=anneal, b_start=bs, b_end=be)
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
                    torch.save(model.state_dict(),
                               os.path.join(CKPT_DIR, f"{mid}_{bw_key}_rep{rep}.pt"))
                    hdf = pd.DataFrame({"epoch": range(EPOCHS),
                                        "train_loss": hist["train_loss"],
                                        "b": hist["b"]})
                    hdf.to_csv(os.path.join(
                        HIST_DIR, f"{mid}_{bw_key}_rep{rep}.csv"), index=False)
                    for noise in E3_NOISES:
                        writer.append(make_row(
                            mid, noise, bw_key, bw, anneal, bs, rep, train_loss,
                            metrics[noise], STATUS_SUCCESS, np.nan, "", dev,
                            f"train {time.time()-t0:.0f}s"))
                    print(f"[{mid} {bw_key} rep{rep}] SUCCESS "
                          f"train={time.time()-t0:.0f}s "
                          f"v(def)={metrics['default']['v_rmse']:.5f} "
                          f"v(mix010)={metrics['mixture_010']['v_rmse']:.5f} "
                          f"elapsed={time.time()-t_start:.0f}s", flush=True)
                else:
                    for noise in E3_NOISES:
                        writer.append(make_row(
                            mid, noise, bw_key, bw, anneal, bs, rep, np.nan,
                            None, status, fail_epoch, fail_reason, dev,
                            f"wall {time.time()-t0:.0f}s"))
                    print(f"[{mid} {bw_key} rep{rep}] {status} ({fail_reason})",
                          flush=True)
                del model
                torch.cuda.empty_cache()

    print(f"[chunk {chunk}] DONE in {(time.time()-t_start)/60:.1f} min", flush=True)


def finalize():
    dfs = []
    for c in CHUNK_MODELS:
        p = chunk_csv(c)
        if not os.path.exists(p):
            print(f"MISSING {p} — run --chunk {c} first")
            return
        dfs.append(pd.read_csv(p))
    df = pd.concat(dfs, ignore_index=True).reindex(columns=RAW_COLUMNS)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(SUM_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(MAN_DIR, exist_ok=True)
    if os.path.exists(RAW_CSV):
        os.remove(RAW_CSV)
    df.to_csv(RAW_CSV, index=False)
    print(f"merged {len(df)} rows -> {RAW_CSV}")

    summarize(RAW_CSV, os.path.join(SUM_DIR, "E3_summary.csv"))
    # pairwise bandwidth comparisons (paired on repeat_id within bw keys)
    pairs = []
    for a in BANDWIDTH_CONDITIONS:
        for b in BANDWIDTH_CONDITIONS:
            if a < b:
                pairs.append((a, b, a, b))
    bw_pairs = []
    for a, b, la, lb in pairs:
        for mid in ["modal_supervised_ann", "modal_supervised_dss2"]:
            bw_pairs.append((f"{mid}_{a}", f"{mid}_{b}", f"{mid} {la}", f"{mid} {lb}"))
    write_manifest(
        EXPERIMENT_ID, MAN_DIR,
        {"dataset": DATASET_ID, "epochs": EPOCHS, "n_repeats": N_REPEATS,
         "noise_settings": E3_NOISES,
         "bandwidth_conditions": {k: {"bandwidth": v[0], "annealing": v[1],
                                      "b_start": v[2], "b_end": v[3]}
                                  for k, v in BANDWIDTH_CONDITIONS.items()},
         "models": CHUNK_MODELS, "lambda_meas": 0.0, "lambda_phys": 1.0,
         "batch": BATCH,
         "seed_policy": "training_seed=100+repeat; data_seed=500+md5(exp); eval_noise_seed=700+noise_index",
         "noise_choice_note": ("4 noise conditions per protocol: default, "
                               "mixture_005, mixture_010, mixture_020 (clean + "
                               "mixture ladder)")},
        notes="E3 bandwidth sensitivity; paired eval noise across bw conditions")
    print("summary + manifest written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--chunk", choices=["ann", "dss2"], default=None)
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
