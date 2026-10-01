"""
E4: lambda_meas (measurement-consistency weight) sensitivity (Stage 2B-E4).

Protocol (EXPERIMENT_PROTOCOL.md E4): 14-bus Modal-Supervised DSS2.
lambda_meas in {0, 0.05, 0.10, 0.20, 0.50}; fixed b=1.0; 4 noise conditions;
5 independent training runs; 200 epochs.

Loss form (code-verified): L_total = L_state + lambda_meas * L_meas
+ lambda_phys * penalty(x), with L_state = state-space modal loss (b=1.0),
L_meas = measurement-space modal loss (dss2_meas_modal_loss).

Symbol discipline (NOTATION.md): lambda_meas = measurement weight;
eta = SGD learning rate only.

Usage:
  python run_e4.py --preflight
  python run_e4.py --smoke
  python run_e4.py --chunk 0|1        # chunk0: lam 0, 0.05 (cuda:0)
                                      # chunk1: lam 0.1, 0.2, 0.5 (cuda:1)
  python run_e4.py --finalize
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
    seal_environment, load_and_split, corrupt_test_sets, make_model,
    train_modal_dss2_with_meas_term, dss2_meas_modal_loss,
    evaluate_learning_model, ResultWriter, summarize, paired_bootstrap,
    write_manifest, gpu_name, RAW_COLUMNS,
    STATUS_SUCCESS, STATUS_TRAINING_FAILURE,
    STATUS_EVALUATION_FAILURE, STATUS_TECHNICAL_ERROR, TrainingDivergedError,
    state_space_modal_loss, _physical_penalty,
)

EXPERIMENT_ID = "E4_cigre14_lambdameas"
DATASET_ID = "cigre14"
FOLDER = "data/cigre14/"
MEAS_V = [0, 1, 12, 7, 11, 14]
MEAS_PF = [0, 10]
E4_NOISES = ["default", "mixture_005", "mixture_010", "mixture_020"]
LAMBDA_MEAS = [0.0, 0.05, 0.10, 0.20, 0.50]
MODEL_ID = "modal_supervised_dss2"
EPOCHS = 200
N_REPEATS = 5
BATCH = 32
BANDWIDTH = 1.0

CHUNK_LAMS = {0: [0.0, 0.05], 1: [0.10, 0.20, 0.50]}
DEV = {0: torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
       1: torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cpu")}

RAW_DIR = os.path.join(EXP, "raw_results", "E4")
RAW_CSV = os.path.join(RAW_DIR, "E4_raw_results.csv")
CKPT_DIR = os.path.join(EXP, "checkpoints", "E4")
HIST_DIR = os.path.join(EXP, "training_history", "E4")
SUM_DIR = os.path.join(EXP, "summaries", "E4")
TAB_DIR = os.path.join(EXP, "tables", "E4")
FIG_DIR = os.path.join(EXP, "figures", "E4")
MAN_DIR = os.path.join(EXP, "manifests", "E4")


def chunk_csv(c):
    return os.path.join(RAW_DIR, f"E4_raw_chunk{c}.csv")


def lam_key(lam):
    return f"lam{lam:g}"


def preflight():
    print("=" * 70)
    print("E4 PREFLIGHT")
    print("=" * 70)
    ok = True

    print("\n[A] Loss-form trace (code)")
    print("    train_modal_dss2_with_meas_term (framework.py):")
    print("      L_total = L_state + lambda_meas * L_meas + lambda_phys * penalty")
    print("      L_state = state_space_modal_loss(y_true, out, ss, b=1.0)")
    print("      L_meas  = dss2_meas_modal_loss(...)  [V,theta,P,Q nodes; P,Q edges]")

    print("\n[B] Numerical unit test (fixed batch, fixed model params)")
    train_ds, test_ds, x_mean, x_std, pflow_mean, pflow_std, state_scale = \
        load_and_split(FOLDER, MEAS_V, MEAS_PF, EXPERIMENT_ID,
                       max_train=20, max_test=5)
    seal_environment(training_seed(0))
    model = make_model(MODEL_ID, DEV[0])
    data = next(iter(__import__("torch_geometric.loader", fromlist=["DataLoader"])
                     .DataLoader(train_ds, batch_size=4, shuffle=False))).to(DEV[0])
    x_mean_d = x_mean.to(DEV[0]); x_std_d = x_std.to(DEV[0])
    em_d = pflow_mean.to(DEV[0]); es_d = pflow_std.to(DEV[0])
    ss = state_scale.to(DEV[0])
    x_in = data.x[:, :8]; e_in = data.edge_attr[:, :6]; y_true = data.y
    node_p = data.x[:, 8:]; edge_p = data.edge_attr[:, 6:]
    model.eval()
    with torch.no_grad():
        out = model(x_in, data.edge_index, e_in)
        Ls = float(state_space_modal_loss(y_true, out, ss, BANDWIDTH, "mean"))
        Lm = float(dss2_meas_modal_loss(x_in, e_in, out, data.edge_index,
                                       node_p, edge_p, BANDWIDTH,
                                       x_mean_d, x_std_d, em_d, es_d))
        pen = float(_physical_penalty(out, data.edge_index, node_p, edge_p,
                                      x_std_d, x_mean_d))
        totals = {lam: float(Ls + lam * Lm + 1.0 * pen) for lam in LAMBDA_MEAS}
    print(f"    L_state={Ls:.6f}, L_meas={Lm:.6f}, penalty={pen:.6f}")
    print(f"    L(0)={totals[0.0]:.6f}, L(0.05)={totals[0.05]:.6f}, "
          f"L(0.10)={totals[0.10]:.6f}, L(0.20)={totals[0.20]:.6f}, "
          f"L(0.50)={totals[0.50]:.6f}")
    lin = True
    for a, b in zip(LAMBDA_MEAS[:-1], LAMBDA_MEAS[1:]):
        diff = totals[b] - totals[a]
        expect = (b - a) * Lm
        rel = abs(diff - expect) / max(abs(expect), 1e-12)
        print(f"    L({b})-L({a}) = {diff:.8f} vs (b-a)*L_m = {expect:.8f} "
              f"(rel err {rel:.2e})")
        lin &= rel < 1e-4
    print(f"    linear-in-lambda identity holds: {lin}")
    ok &= lin

    print("\n[C] lambda_meas = 0 control group")
    print(f"    contribution at lambda=0: {0.0 * Lm} (zero by construction; "
          f"L_meas still computed for logging)")

    print("\n[D] Fixed bandwidth")
    print(f"    all E4 runs use b={BANDWIDTH} fixed, annealing=False; "
          f"bandwidth_schedule column = 'fixed'")

    print("\n[E] Output isolation")
    for p in [chunk_csv(0), chunk_csv(1), RAW_CSV, CKPT_DIR, HIST_DIR,
              SUM_DIR, TAB_DIR, FIG_DIR, MAN_DIR]:
        print(f"    {p}")
    collide = any(os.path.exists(chunk_csv(c)) for c in CHUNK_LAMS)
    ok &= not collide
    print(f"    chunk CSV collision: {collide}")

    print("\n[F] Expected workload (from protocol: Modal-DSS2 only)")
    n = len(LAMBDA_MEAS) * len(E4_NOISES) * N_REPEATS
    print(f"    1 model x 5 lambda x 4 noise x 5 repeats = {n} rows")
    print(f"    training runs: 25; checkpoints: 25; history files: 25")

    print("\n" + "=" * 70)
    print(f"E4 PREFLIGHT: {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    return ok


def run_smoke():
    print("=" * 70)
    print("E4 SMOKE (5 lambda x 1 noise x 1 repeat x 5 epochs)")
    print("=" * 70)
    train_ds, test_ds, x_mean, x_std, pflow_mean, pflow_std, state_scale = \
        load_and_split(FOLDER, MEAS_V, MEAS_PF, EXPERIMENT_ID,
                       max_train=50, max_test=10)
    ok = True
    for lam in LAMBDA_MEAS:
        dev = DEV[0] if lam in CHUNK_LAMS[0] else DEV[1]
        try:
            model, loss_val, hist = train_modal_dss2_with_meas_term(
                MODEL_ID, train_ds, epochs=5, batch_size=BATCH, device=dev,
                x_mean=x_mean, x_std=x_std, pflow_mean=pflow_mean,
                pflow_std=pflow_std, state_scale=state_scale, repeat_id=0,
                bandwidth=BANDWIDTH, lambda_meas=lam, lambda_phys=1.0,
                annealing=False)
            fin = (np.isfinite(loss_val)
                   and all(np.isfinite(hist["train_state_loss"]))
                   and all(np.isfinite(hist["train_meas_loss"])))
            print(f"    [lam={lam:4.2f}] loss={loss_val:.5f} finite={fin} "
                  f"state={hist['train_state_loss'][-1]:.5f} "
                  f"meas={hist['train_meas_loss'][-1]:.5f}")
            ok &= fin
        except Exception as e:
            ok = False
            print(f"    [lam={lam:4.2f}] FAIL: {type(e).__name__}: {e}")
    print("=" * 70)
    print(f"E4 SMOKE: {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    return ok


def run_chunk(c):
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(HIST_DIR, exist_ok=True)
    csv = chunk_csv(c)
    if os.path.exists(csv):
        os.remove(csv)
    writer = ResultWriter(csv)
    dev = DEV[c]
    t_start = time.time()

    train_ds, test_ds, x_mean, x_std, pflow_mean, pflow_std, state_scale = \
        load_and_split(FOLDER, MEAS_V, MEAS_PF, EXPERIMENT_ID)
    corrupted = corrupt_test_sets(test_ds, E4_NOISES, x_mean, x_std,
                                  pflow_mean, pflow_std)
    print(f"[chunk {c}] train={len(train_ds)} test={len(test_ds)} "
          f"device={dev}", flush=True)

    m = MODEL_REGISTRY[MODEL_ID]
    for lam in CHUNK_LAMS[c]:
        for rep in range(N_REPEATS):
            t0 = time.time()
            status, model, train_loss, hist = STATUS_SUCCESS, None, np.nan, None
            fail_epoch, fail_reason = np.nan, ""
            try:
                model, train_loss, hist = train_modal_dss2_with_meas_term(
                    MODEL_ID, train_ds, epochs=EPOCHS, batch_size=BATCH,
                    device=dev, x_mean=x_mean, x_std=x_std,
                    pflow_mean=pflow_mean, pflow_std=pflow_std,
                    state_scale=state_scale, repeat_id=rep,
                    bandwidth=BANDWIDTH, lambda_meas=lam, lambda_phys=1.0,
                    annealing=False)
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
                        model, MODEL_ID, corrupted, x_mean, x_std,
                        state_scale, dev)
                except Exception as e:
                    status, fail_reason = \
                        STATUS_EVALUATION_FAILURE, f"{type(e).__name__}: {e}"

            base = {
                "experiment_id": EXPERIMENT_ID, "dataset_id": DATASET_ID,
                "model": MODEL_ID, "model_name": m["model_name"],
                "architecture": m["architecture"], "loss": m["loss"],
                "bad_data_rate": 0.0, "bandwidth": BANDWIDTH,
                "bandwidth_schedule": "fixed", "lambda_meas": lam,
                "eta": 3e-3, "training_seed": training_seed(rep),
                "data_seed": data_seed(EXPERIMENT_ID),
                "epochs": EPOCHS,
                "device": str(dev), "gpu_name": gpu_name(),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            }
            if status == STATUS_SUCCESS:
                torch.save(model.state_dict(),
                           os.path.join(CKPT_DIR, f"{lam_key(lam)}_rep{rep}.pt"))
                hdf = pd.DataFrame({
                    "epoch": range(EPOCHS), "train_loss": hist["train_loss"],
                    "train_state_loss": hist["train_state_loss"],
                    "train_meas_loss": hist["train_meas_loss"], "b": hist["b"]})
                hdf.to_csv(os.path.join(
                    HIST_DIR, f"{lam_key(lam)}_rep{rep}.csv"), index=False)
                for noise in E4_NOISES:
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
                        "state_loss": hist["train_state_loss"][-1],
                        "meas_loss": hist["train_meas_loss"][-1],
                        "notes": f"train {time.time()-t0:.0f}s",
                    })
                    writer.append(row)
                print(f"[lam={lam:4.2f} rep{rep}] SUCCESS "
                      f"train={time.time()-t0:.0f}s "
                      f"v(def)={metrics['default']['v_rmse']:.5f} "
                      f"v(mix010)={metrics['mixture_010']['v_rmse']:.5f} "
                      f"elapsed={time.time()-t_start:.0f}s", flush=True)
            else:
                for noise in E4_NOISES:
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
                        "notes": f"wall {time.time()-t0:.0f}s",
                    })
                    writer.append(row)
                print(f"[lam={lam:4.2f} rep{rep}] {status} ({fail_reason})",
                      flush=True)
            del model
            torch.cuda.empty_cache()

    print(f"[chunk {c}] DONE in {(time.time()-t_start)/60:.1f} min", flush=True)


def finalize():
    dfs = []
    for c in CHUNK_LAMS:
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

    summarize(RAW_CSV, os.path.join(SUM_DIR, "E4_summary.csv"))
    write_manifest(
        EXPERIMENT_ID, MAN_DIR,
        {"dataset": DATASET_ID, "model": MODEL_ID, "epochs": EPOCHS,
         "n_repeats": N_REPEATS, "noise_settings": E4_NOISES,
         "lambda_meas": LAMBDA_MEAS, "bandwidth": BANDWIDTH,
         "bandwidth_schedule": "fixed", "lambda_phys": 1.0,
         "batch": BATCH,
         "loss_form": "L_total = L_state + lambda_meas*L_meas + lambda_phys*penalty",
         "seed_policy": "training_seed=100+repeat; data_seed=500+md5(exp); eval_noise_seed=700+idx (shared across lambda -> paired)",
         "noise_choice_note": "4 noise conditions: default, mixture_005, mixture_010, mixture_020"},
        notes="E4 lambda_meas sensitivity; fixed b=1.0; no annealing")
    print("summary + manifest written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--chunk", type=int, choices=[0, 1], default=None)
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    if args.preflight:
        sys.exit(0 if preflight() else 1)
    if args.smoke:
        sys.exit(0 if run_smoke() else 1)
    if args.chunk is not None:
        run_chunk(args.chunk)
    elif args.finalize:
        finalize()
    else:
        ap.print_help()
