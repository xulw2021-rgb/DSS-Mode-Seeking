"""
E5: robust-loss family comparison (Stage 2D-E5).

Frozen design (E5_NECESSITY_DECISION.md + E5_ROBUST_LOSS_PARAMETER_PROTOCOL.md):
  CIGRE-14, corrected train-only pipeline (preprocessing_version=train_only),
  ANN_DSSE(8->64->64->2, dropout 0.1) for ALL losses,
  6 losses x 4 noises {default, mixture_010, mixture_020, erroneous}
  x 5 independent training runs, 200 epochs, Adamax 3e-3, batch 64.
  Losses: MSE, Modal b=1.0, Modal b=2.0, Huber d=1.0, Tukey c=2.0,
  Quantile tau=0.5 — all on standardized state residuals
  r_j = (y_j - f_j)/s_j with train-derived s_j (state_scale).

Usage:
  python run_e5.py --preflight
  python run_e5.py --smoke
  python run_e5.py --run
  python run_e5.py --finalize
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from framework import (  # noqa: E402
    MODEL_REGISTRY, training_seed, data_seed, eval_noise_seed,
    seal_environment, build_dataset_train_only, corrupt_test_sets,
    make_model, evaluate_learning_model, ResultWriter, summarize,
    paired_bootstrap, write_manifest, gpu_name, RAW_COLUMNS,
    STATUS_SUCCESS, STATUS_TRAINING_FAILURE,
    STATUS_EVALUATION_FAILURE, STATUS_TECHNICAL_ERROR, TrainingDivergedError,
    gaussian_kernel_modal_loss,
)
from modal_losses import huber_loss, tukey_biweight_loss, quantile_loss

EXPERIMENT_ID = "E5_cigre14_robustloss"
FOLDER = "data/cigre14/"
MEAS_V = [0, 1, 12, 7, 11, 14]
MEAS_PF = [0, 10]
E5_NOISES = ["default", "mixture_010", "mixture_020", "erroneous"]
EPOCHS, N_REPEATS = 200, 5
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LOSS_GRID = [
    ("mse", "MSE", None),
    ("modal_b1.0", "Modal b=1.0", ("modal", 1.0)),
    ("modal_b2.0", "Modal b=2.0", ("modal", 2.0)),
    ("huber_d1.0", "Huber d=1.0", ("huber", 1.0)),
    ("tukey_c2.0", "Tukey c=2.0", ("tukey", 2.0)),
    ("quantile_t0.5", "Quantile t=0.5", ("quantile", 0.5)),
]
ARCH = "ANN_DSSE(8->64->64->2, LeakyReLU, dropout=0.1)"

OUT_DIR = os.path.join(EXP, "raw_results", "CORRECTED")
RAW_CSV = os.path.join(OUT_DIR, "E5_corrected.csv")
SUM_DIR = os.path.join(EXP, "summaries", "CORRECTED")
TAB_DIR = os.path.join(EXP, "tables", "CORRECTED")
FIG_DIR = os.path.join(EXP, "figures", "CORRECTED")
MAN_DIR = os.path.join(EXP, "manifests", "CORRECTED")


def loss_fn(key, r, ss, out, y):
    """Loss on standardized residuals r=(y-out)/ss (mean reduction)."""
    if key == "mse":
        return (r ** 2).mean()
    if key.startswith("modal"):
        return gaussian_kernel_modal_loss(r, float(key.split("modal_b")[1]),
                                          reduction="mean")
    if key.startswith("huber"):
        return huber_loss(r, delta=float(key.split("huber_d")[1]),
                          reduction="mean")
    if key.startswith("tukey"):
        return tukey_biweight_loss(r, c=float(key.split("tukey_c")[1]),
                                   reduction="mean")
    if key.startswith("quantile"):
        return quantile_loss(r, tau=float(key.split("quantile_t")[1]),
                             reduction="mean")
    raise ValueError(key)


def train_one(key, train_ds, ss, rep):
    seal_environment(training_seed(rep))
    model = make_model("supervised_ann", DEV)
    opt = torch.optim.Adamax(model.parameters(), lr=3e-3)
    loader = __import__("torch_geometric.loader",
                        fromlist=["DataLoader"]).DataLoader(
        train_ds, batch_size=64, shuffle=True)
    ss_d = ss.to(DEV)
    for epoch in range(EPOCHS):
        model.train()
        tot, nb = 0.0, 0
        for data in loader:
            data = data.to(DEV)
            x_in = data.x[:, :8]
            y = data.y
            opt.zero_grad()
            out = model(x_in)
            r = (y - out) / ss_d
            loss = loss_fn(key, r, ss_d, out, y)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        el = tot / max(nb, 1)
        if not np.isfinite(el):
            raise TrainingDivergedError(epoch, el)
    return model, el


def preflight():
    print("=" * 60)
    print("E5 PREFLIGHT")
    print("=" * 60)
    ok = True
    train_ds, test_ds, xm, xs, pm, ps, ss = build_dataset_train_only(
        FOLDER, MEAS_V, MEAS_PF, experiment_id=EXPERIMENT_ID,
        max_train=50, max_test=10)
    print(f"[data] train={len(train_ds)} test={len(test_ds)} "
          f"state_scale={[round(float(v),4) for v in ss]}")
    # verify losses differ on fixed residuals
    r = torch.linspace(-3, 3, 1001)
    vals = {k: float(loss_fn(k, r, None, None, None)) for k, _, _ in LOSS_GRID}
    for k, _, _ in LOSS_GRID:
        print(f"  {k:14s} mean loss on r grid = {vals[k]:.5f}")
    distinct = len(set(round(v, 6) for v in vals.values())) == len(LOSS_GRID)
    print(f"[losses distinct] {distinct}")
    ok &= distinct
    collide = os.path.exists(RAW_CSV)
    print(f"[isolation] E5 raw CSV exists (must be fresh): {collide}")
    ok &= not collide
    print(f"[workload] {len(LOSS_GRID)} losses x {len(E5_NOISES)} noises x "
          f"{N_REPEATS} runs = {len(LOSS_GRID)*len(E5_NOISES)*N_REPEATS} rows; "
          f"{len(LOSS_GRID)*N_REPEATS} training runs")
    print("E5 PREFLIGHT:", "PASS" if ok else "FAIL")
    return ok


def smoke():
    print("E5 smoke: 1 rep x 3 epochs per loss on 50-graph subset")
    train_ds, _, _, _, _, _, ss = build_dataset_train_only(
        FOLDER, MEAS_V, MEAS_PF, experiment_id=EXPERIMENT_ID,
        max_train=50, max_test=5)
    ok = True
    for key, _, _ in LOSS_GRID:
        model, loss = train_one(key, train_ds, ss, 0)
        print(f"  {key:14s} loss={loss:.5f} finite={np.isfinite(loss)}")
        ok &= np.isfinite(loss)
    print("E5 SMOKE:", "PASS" if ok else "FAIL")
    return ok


def run_full():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(RAW_CSV):
        os.remove(RAW_CSV)
    writer = ResultWriter(RAW_CSV)
    train_ds, test_ds, xm, xs, pm, ps, ss = build_dataset_train_only(
        FOLDER, MEAS_V, MEAS_PF, experiment_id=EXPERIMENT_ID)
    corrupted = corrupt_test_sets(test_ds, E5_NOISES, xm, xs, pm, ps)
    print(f"[E5] train={len(train_ds)} test={len(test_ds)}", flush=True)
    t0 = time.time()
    m = MODEL_REGISTRY["supervised_ann"]
    for key, label, _ in LOSS_GRID:
        for rep in range(N_REPEATS):
            try:
                model, train_loss = train_one(key, train_ds, ss, rep)
                met = evaluate_learning_model(
                    model, "supervised_ann", corrupted, xm, xs, ss, DEV)
                for noise in E5_NOISES:
                    mm = met[noise]
                    writer.append({
                        "experiment_id": EXPERIMENT_ID, "dataset_id": "cigre14",
                        "data_source": "corrected_cigre14",
                        "external_benchmark": False,
                        "model": key, "model_name": label,
                        "architecture": ARCH, "loss": f"{label} (state-space)",
                        "noise": noise, "bad_data_rate": 0.0, "bandwidth": 1.0,
                        "bandwidth_schedule": "fixed", "lambda_meas": 0.0,
                        "eta": 3e-3, "repeat_id": rep,
                        "training_seed": training_seed(rep),
                        "data_seed": data_seed(EXPERIMENT_ID),
                        "eval_noise_seed": eval_noise_seed(noise),
                        "epochs": EPOCHS, "best_epoch": EPOCHS,
                        "train_loss": train_loss, "val_loss": np.nan,
                        "v_rmse": mm["v_rmse"], "th_rmse": mm["th_rmse"],
                        "loading_rmse": mm["loading_rmse"],
                        "modal_score": mm["modal_score"],
                        "hit_rate": mm["hit_rate"],
                        "runtime_seconds": mm["runtime_seconds"],
                        "converged": 1.0, "device": str(DEV),
                        "gpu_name": gpu_name(),
                        "torch_version": torch.__version__,
                        "cuda_version": torch.version.cuda,
                        "status": STATUS_SUCCESS,
                        "failure_epoch": np.nan, "failure_reason": "",
                        "v_mae": mm["v_mae"], "th_mae": mm["th_mae"],
                        "state_loss": np.nan, "meas_loss": np.nan,
                        "notes": "preprocessing_version=train_only",
                    })
                print(f"[{key} rep{rep}] "
                      f"v(def)={met['default']['v_rmse']:.5f} "
                      f"v(mix010)={met['mixture_010']['v_rmse']:.5f} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
                del model
                torch.cuda.empty_cache()
            except TrainingDivergedError as e:
                print(f"[{key} rep{rep}] TRAINING_FAILURE {e}", flush=True)
            except Exception as e:
                print(f"[{key} rep{rep}] ERR {type(e).__name__}: {e}",
                      flush=True)
    print(f"[E5] DONE in {(time.time()-t0)/60:.1f} min", flush=True)


def finalize():
    os.makedirs(SUM_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(MAN_DIR, exist_ok=True)
    summarize(RAW_CSV, os.path.join(SUM_DIR, "E5_summary.csv"))
    pairs = [(k, "modal_b1.0", l, "Modal b=1.0")
             for k, l, _ in LOSS_GRID if k != "modal_b1.0"]
    # paired bootstrap per noise vs Modal b=1.0
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(RAW_CSV)
    rows = []
    rng = np.random.RandomState(0)
    for key, label, _ in LOSS_GRID:
        if key == "modal_b1.0":
            continue
        for noise in E5_NOISES:
            a = df[(df.model == key) & (df.noise == noise)].sort_values(
                "repeat_id").v_rmse.values
            b = df[(df.model == "modal_b1.0") & (df.noise == noise)].sort_values(
                "repeat_id").v_rmse.values
            n = min(len(a), len(b))
            if n < 2:
                continue
            d = a[:n] - b[:n]
            boot = np.array([rng.choice(d, n, replace=True).mean()
                             for _ in range(10000)])
            rows.append({"loss": label, "noise": noise, "n": n,
                         "mean_diff": d.mean(),
                         "ci_low": np.percentile(boot, 2.5),
                         "ci_high": np.percentile(boot, 97.5)})
    bt = pd.DataFrame(rows)
    bt.to_csv(os.path.join(TAB_DIR, "Table_E5_comparison.csv"), index=False)
    with open(os.path.join(TAB_DIR, "Table_E5_comparison.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n"
                "\\caption{Robust-loss comparison, 14-bus ANN (corrected "
                "pipeline, 5 independent training runs, 200 epochs). Paired "
                "bootstrap (10k): V\\_RMSE difference of each loss vs Modal "
                "b=1.0 (positive = worse than Modal b=1.0).}\n"
                "\\label{tab:E5_comparison}\n\\footnotesize\n"
                "\\begin{tabular}{llrrrr}\n\\toprule\n"
                "Loss & Noise & n & MeanDiff & CI low & CI high \\\\\n\\midrule\n")
        for _, r in bt.iterrows():
            f.write(f"{r['loss']} & {r['noise']} & {r['n']} & "
                    f"{r['mean_diff']:.4f} & {r['ci_low']:.4f} & "
                    f"{r['ci_high']:.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    # table E5 main
    rows2 = []
    for key, label, _ in LOSS_GRID:
        for noise in E5_NOISES:
            g = df[(df.model == key) & (df.noise == noise)]
            rows2.append({"Loss": label, "Noise": noise,
                          "n": len(g),
                          "v_mean": round(g.v_rmse.mean(), 5),
                          "v_sd": round(g.v_rmse.std(), 5),
                          "v_median": round(g.v_rmse.median(), 5),
                          "th_mean": round(g.th_rmse.mean(), 5),
                          "th_sd": round(g.th_rmse.std(), 5)})
    t2 = pd.DataFrame(rows2)
    t2.to_csv(os.path.join(TAB_DIR, "Table_E5_main.csv"), index=False)
    with open(os.path.join(TAB_DIR, "Table_E5_main.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n"
                "\\caption{Robust-loss family comparison (14-bus ANN, "
                "corrected pipeline). Mean$\\pm$std over 5 independent "
                "training runs.}\n\\label{tab:E5_main}\n\\footnotesize\n"
                "\\begin{tabular}{llrrrrr}\n\\toprule\n"
                "Loss & Noise & n & V mean & V sd & $\\theta$ mean & "
                "$\\theta$ sd \\\\\n\\midrule\n")
        for _, r in t2.iterrows():
            f.write(f"{r['Loss']} & {r['Noise']} & {r['n']} & "
                    f"{r['v_mean']:.4f} & {r['v_sd']:.4f} & "
                    f"{r['th_mean']:.4f} & {r['th_sd']:.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    # figure: median V_RMSE by loss per noise
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(E5_NOISES)); w = 0.13
    for i, (key, label, _) in enumerate(LOSS_GRID):
        vals = []
        for noise in E5_NOISES:
            g = df[(df.model == key) & (df.noise == noise)]
            vals.append(g.v_rmse.median())
        ax.bar(xs + (i - 2.5) * w, vals, w, label=label)
    ax.set_xticks(xs); ax.set_xticklabels(E5_NOISES, rotation=15)
    ax.set_ylabel("V RMSE (pu, median)"); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_E5_main.pdf")); plt.close(fig)
    write_manifest(EXPERIMENT_ID, MAN_DIR,
                   {"pipeline": "train_only", "losses": [l for _, l, _ in LOSS_GRID],
                    "noises": E5_NOISES, "n_repeats": N_REPEATS,
                    "epochs": EPOCHS, "arch": ARCH,
                    "parameter_protocol": "E5_ROBUST_LOSS_PARAMETER_PROTOCOL.md"},
                   notes="E5 robust-loss family comparison, corrected pipeline")
    print("E5 summaries/tables/figures/manifest written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    if args.preflight:
        sys.exit(0 if preflight() else 1)
    if args.smoke:
        sys.exit(0 if smoke() else 1)
    if args.run:
        run_full()
    elif args.finalize:
        finalize()
    else:
        ap.print_help()
