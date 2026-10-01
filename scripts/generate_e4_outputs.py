"""
E4 post-processing: lambda_meas tables + figures + paired bootstrap.
All outputs derived programmatically from E4_raw_results.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

RAW_CSV = os.path.join(EXP, "raw_results", "E4", "E4_raw_results.csv")
TAB_DIR = os.path.join(EXP, "tables", "E4")
FIG_DIR = os.path.join(EXP, "figures", "E4")

LAMS = [0.0, 0.05, 0.10, 0.20, 0.50]
LAM_LABEL = {0.0: "lam=0", 0.05: "lam=0.05", 0.10: "lam=0.10",
             0.20: "lam=0.20", 0.50: "lam=0.50"}
NOISE_ORDER = ["default", "mixture_005", "mixture_010", "mixture_020"]
NOISE_LABEL = {"default": "Default", "mixture_005": "5% Mixture",
               "mixture_010": "10% Mixture", "mixture_020": "20% Mixture"}


def main():
    os.makedirs(TAB_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(RAW_CSV)

    # ---------------- Table_E4_lambda ----------------
    rows = []
    for lam in LAMS:
        for noise in NOISE_ORDER:
            g = df[(df.lambda_meas == lam) & (df.noise == noise)]
            succ = g[g.status == "SUCCESS"]
            rows.append({
                "lambda_meas": lam, "noise": NOISE_LABEL[noise],
                "attempted": len(g), "success": len(succ),
                "training_failure": (g.status == "TRAINING_FAILURE").sum(),
                "failure_rate": round((len(g) - len(succ)) / len(g), 3),
                "v_rmse_mean": round(succ.v_rmse.mean(), 5) if len(succ) else np.nan,
                "v_rmse_sd": round(succ.v_rmse.std(), 5) if len(succ) else np.nan,
                "v_rmse_median": round(succ.v_rmse.median(), 5) if len(succ) else np.nan,
                "v_rmse_min": round(succ.v_rmse.min(), 5) if len(succ) else np.nan,
                "v_rmse_max": round(succ.v_rmse.max(), 5) if len(succ) else np.nan,
                "th_rmse_mean": round(succ.th_rmse.mean(), 5) if len(succ) else np.nan,
                "th_rmse_sd": round(succ.th_rmse.std(), 5) if len(succ) else np.nan,
                "v_mae_mean": round(succ.v_mae.mean(), 5) if len(succ) else np.nan,
                "th_mae_mean": round(succ.th_mae.mean(), 5) if len(succ) else np.nan,
                "state_loss_mean": round(succ.state_loss.mean(), 5) if len(succ) else np.nan,
                "meas_loss_mean": round(succ.meas_loss.mean(), 5) if len(succ) else np.nan,
                "n_collapsed": (succ.v_rmse > 1.0).sum() if len(succ) else np.nan,
            })
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TAB_DIR, "Table_E4_lambda.csv"), index=False)
    with open(os.path.join(TAB_DIR, "Table_E4_lambda.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n"
                "\\caption{$\\lambda_{\\mathrm{meas}}$ sensitivity, "
                "Modal-Supervised DSS2, 14-bus (5 independent training runs, "
                "200 epochs, fixed b=1.0).}\n\\label{tab:E4_lambda}\n"
                "\\footnotesize\n\\begin{tabular}{rrrrrrrrrr}\n\\toprule\n"
                "$\\lambda$ & Noise & Att. & Succ. & FailRate & V mean & "
                "V sd & V med & $\\theta$ mean & V-MAE & Coll. \\\\\n\\midrule\n")
        for _, r in t.iterrows():
            f.write(f"{r['lambda_meas']:.2f} & {r['noise']} & {r['attempted']} & "
                    f"{r['success']} & {r['failure_rate']:.2f} & "
                    f"{r['v_rmse_mean']:.4f} & {r['v_rmse_sd']:.4f} & "
                    f"{r['v_rmse_median']:.4f} & {r['th_rmse_mean']:.4f} & "
                    f"{r['v_mae_mean']:.4f} & {r['n_collapsed']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # ---------------- paired bootstrap vs lam=0 ----------------
    boot_rows = []
    rng = np.random.RandomState(0)
    for lam in LAMS[1:]:
        for noise in NOISE_ORDER:
            a = df[(df.lambda_meas == lam) & (df.noise == noise)
                   & (df.status == "SUCCESS")].sort_values("repeat_id").v_rmse.values
            b = df[(df.lambda_meas == 0.0) & (df.noise == noise)
                   & (df.status == "SUCCESS")].sort_values("repeat_id").v_rmse.values
            n = min(len(a), len(b))
            if n < 2:
                continue
            d = a[:n] - b[:n]
            boot = np.array([rng.choice(d, n, replace=True).mean()
                             for _ in range(10000)])
            boot_rows.append({
                "lambda_meas": lam, "noise": NOISE_LABEL[noise], "n_pairs": n,
                "mean_diff": d.mean(), "ci_low": np.percentile(boot, 2.5),
                "ci_high": np.percentile(boot, 97.5),
            })
    bt = pd.DataFrame(boot_rows)
    bt.to_csv(os.path.join(TAB_DIR, "Table_E4_comparison.csv"), index=False)
    with open(os.path.join(TAB_DIR, "Table_E4_comparison.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n"
                "\\caption{Paired bootstrap (10k resamples): V\\_RMSE "
                "difference of each $\\lambda_{\\mathrm{meas}}$ vs "
                "$\\lambda_{\\mathrm{meas}}=0$ (positive = worse).}\n"
                "\\label{tab:E4_comparison}\n\\footnotesize\n"
                "\\begin{tabular}{rlrrrr}\n\\toprule\n"
                "$\\lambda$ & Noise & n & MeanDiff & CI low & CI high \\\\\n"
                "\\midrule\n")
        for _, r in bt.iterrows():
            f.write(f"{r['lambda_meas']:.2f} & {r['noise']} & {r['n_pairs']} & "
                    f"{r['mean_diff']:.4f} & {r['ci_low']:.4f} & "
                    f"{r['ci_high']:.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # ---------------- figures ----------------
    xs = np.arange(len(NOISE_ORDER))
    fig, ax = plt.subplots(figsize=(7, 4))
    for lam in LAMS:
        vals, errs = [], []
        for noise in NOISE_ORDER:
            succ = df[(df.lambda_meas == lam) & (df.noise == noise)
                      & (df.status == "SUCCESS")]
            vals.append(succ.v_rmse.median() if len(succ) else np.nan)
            errs.append(succ.v_rmse.std() if len(succ) else 0)
        ax.plot(xs, vals, marker="o", label=LAM_LABEL[lam])
    ax.set_xticks(xs); ax.set_xticklabels([NOISE_LABEL[n] for n in NOISE_ORDER])
    ax.set_ylabel("V RMSE (pu, median)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_E4_vrmse.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for lam in LAMS:
        vals = []
        for noise in NOISE_ORDER:
            succ = df[(df.lambda_meas == lam) & (df.noise == noise)
                      & (df.status == "SUCCESS")]
            vals.append(succ.v_rmse.mean() if len(succ) else np.nan)
        ax.plot(xs, vals, marker="o", label=LAM_LABEL[lam])
    ax.set_xticks(xs); ax.set_xticklabels([NOISE_LABEL[n] for n in NOISE_ORDER])
    ax.set_ylabel("V RMSE (pu, mean)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_E4_vrmse_mean.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for lam in LAMS:
        vals = []
        for noise in NOISE_ORDER:
            succ = df[(df.lambda_meas == lam) & (df.noise == noise)
                      & (df.status == "SUCCESS")]
            vals.append(succ.th_rmse.mean() if len(succ) else np.nan)
        ax.plot(xs, vals, marker="o", label=LAM_LABEL[lam])
    ax.set_xticks(xs); ax.set_xticklabels([NOISE_LABEL[n] for n in NOISE_ORDER])
    ax.set_ylabel("theta RMSE (rad, mean)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_E4_thrmse.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for lam in LAMS:
        rates = []
        for noise in NOISE_ORDER:
            succ = df[(df.lambda_meas == lam) & (df.noise == noise)
                      & (df.status == "SUCCESS")]
            rates.append((succ.v_rmse > 1.0).mean() if len(succ) else np.nan)
        ax.plot(xs, rates, marker="o", label=LAM_LABEL[lam])
    ax.set_xticks(xs); ax.set_xticklabels([NOISE_LABEL[n] for n in NOISE_ORDER])
    ax.set_ylabel("Collapse rate (V_RMSE>1.0)"); ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_E4_collapse_rate.pdf"))
    plt.close(fig)

    print("E4 tables + figures written")


if __name__ == "__main__":
    main()
