"""
E3 post-processing: bandwidth tables + figures + pairwise bootstrap.
All outputs derived programmatically from E3_raw_results.csv.
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

RAW_CSV = os.path.join(EXP, "raw_results", "E3", "E3_raw_results.csv")
TAB_DIR = os.path.join(EXP, "tables", "E3")
FIG_DIR = os.path.join(EXP, "figures", "E3")
SUM_DIR = os.path.join(EXP, "summaries", "E3")
HIST_DIR = os.path.join(EXP, "training_history", "E3")

BW_ORDER = ["b0.5", "b1.0", "b2.0", "anneal_8_2"]
BW_LABEL = {"b0.5": "b=0.5", "b1.0": "b=1.0", "b2.0": "b=2.0",
            "anneal_8_2": "anneal 8->2"}
NOISE_ORDER = ["default", "mixture_005", "mixture_010", "mixture_020"]
NOISE_LABEL = {"default": "Default", "mixture_005": "5% Mixture",
               "mixture_010": "10% Mixture", "mixture_020": "20% Mixture"}
MODELS = ["modal_supervised_ann", "modal_supervised_dss2"]
MODEL_LABEL = {"modal_supervised_ann": "M-ANN", "modal_supervised_dss2": "M-DSS2"}


def main():
    os.makedirs(TAB_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(RAW_CSV)
    _bwmap = {0.5: "b0.5", 1.0: "b1.0", 2.0: "b2.0"}
    df["bw_key"] = df.apply(
        lambda r: ("anneal_8_2" if r["bandwidth_schedule"] == "anneal_8_2"
                   else _bwmap.get(float(r["bandwidth"]), "unknown")), axis=1)

    # ---------------- Table_E3_bandwidth ----------------
    rows = []
    for mid in MODELS:
        for bw in BW_ORDER:
            for noise in NOISE_ORDER:
                g = df[(df.model == mid) & (df.bw_key == bw)
                       & (df.noise == noise)]
                succ = g[g.status == "SUCCESS"]
                rows.append({
                    "model": MODEL_LABEL[mid], "bandwidth": BW_LABEL[bw],
                    "noise": NOISE_LABEL[noise], "attempted": len(g),
                    "success": len(succ),
                    "training_failure": (g.status == "TRAINING_FAILURE").sum(),
                    "failure_rate": round((len(g) - len(succ)) / len(g), 3),
                    "v_rmse_mean": round(succ.v_rmse.mean(), 5) if len(succ) else np.nan,
                    "v_rmse_sd": round(succ.v_rmse.std(), 5) if len(succ) else np.nan,
                    "v_rmse_median": round(succ.v_rmse.median(), 5) if len(succ) else np.nan,
                    "v_rmse_min": round(succ.v_rmse.min(), 5) if len(succ) else np.nan,
                    "v_rmse_max": round(succ.v_rmse.max(), 5) if len(succ) else np.nan,
                    "th_rmse_mean": round(succ.th_rmse.mean(), 5) if len(succ) else np.nan,
                    "th_rmse_sd": round(succ.th_rmse.std(), 5) if len(succ) else np.nan,
                    "n_collapsed": (succ.v_rmse > 1.0).sum() if len(succ) else np.nan,
                })
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TAB_DIR, "Table_E3_bandwidth.csv"), index=False)
    with open(os.path.join(TAB_DIR, "Table_E3_bandwidth.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n"
                "\\caption{Bandwidth sensitivity, 14-bus (5 independent training "
                "runs, 200 epochs). Failure-aware; collapsed = V\\_RMSE > 1.0 "
                "(performance criterion, not a protocol failure).}\n"
                "\\label{tab:E3_bandwidth}\n\\footnotesize\n"
                "\\begin{tabular}{lllrrrrrrrr}\n\\toprule\n"
                "Model & Bandwidth & Noise & Att. & Succ. & FailRate & "
                "V mean & V sd & V med & $\\theta$ mean & Coll. \\\\\n\\midrule\n")
        for _, r in t.iterrows():
            f.write(f"{r['model']} & {r['bandwidth']} & {r['noise']} & "
                    f"{r['attempted']} & {r['success']} & {r['failure_rate']:.2f} & "
                    f"{r['v_rmse_mean']:.4f} & {r['v_rmse_sd']:.4f} & "
                    f"{r['v_rmse_median']:.4f} & {r['th_rmse_mean']:.4f} & "
                    f"{r['n_collapsed']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # ---------------- Table_E3_comparison (paired bootstrap bw pairs) -------
    from framework import paired_bootstrap
    pairs = []
    for mid in MODELS:
        base = "b1.0"
        for bw in BW_ORDER:
            if bw != base:
                pairs.append((f"{mid}_{bw}", f"{mid}_{base}",
                              f"{MODEL_LABEL[mid]} {BW_LABEL[bw]}",
                              f"{MODEL_LABEL[mid]} {BW_LABEL[base]}"))
    # paired_bootstrap compares within model+noise using repeat_id; we need
    # bw as an additional grouping dim — do it manually here
    boot_rows = []
    rng = np.random.RandomState(0)
    for mid in MODELS:
        for bw in BW_ORDER:
            if bw == "b1.0":
                continue
            for noise in NOISE_ORDER:
                a = df[(df.model == mid) & (df.bw_key == bw)
                       & (df.noise == noise) & (df.status == "SUCCESS")]
                b = df[(df.model == mid) & (df.bw_key == "b1.0")
                       & (df.noise == noise) & (df.status == "SUCCESS")]
                a = a.sort_values("repeat_id").v_rmse.values
                b = b.sort_values("repeat_id").v_rmse.values
                n = min(len(a), len(b))
                if n < 2:
                    continue
                d = a[:n] - b[:n]
                boot = np.array([rng.choice(d, n, replace=True).mean()
                                 for _ in range(10000)])
                boot_rows.append({
                    "model": MODEL_LABEL[mid], "bw": BW_LABEL[bw],
                    "noise": NOISE_LABEL[noise], "n_pairs": n,
                    "mean_diff": d.mean(),
                    "ci_low": np.percentile(boot, 2.5),
                    "ci_high": np.percentile(boot, 97.5),
                })
    bt = pd.DataFrame(boot_rows)
    bt.to_csv(os.path.join(TAB_DIR, "Table_E3_comparison.csv"), index=False)
    with open(os.path.join(TAB_DIR, "Table_E3_comparison.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n"
                "\\caption{Paired bootstrap (10k resamples): V\\_RMSE difference "
                "of each bandwidth condition vs b=1.0 (positive = worse than "
                "b=1.0).}\n\\label{tab:E3_comparison}\n\\footnotesize\n"
                "\\begin{tabular}{lllrrrr}\n\\toprule\n"
                "Model & Bandwidth & Noise & n & MeanDiff & CI low & CI high "
                "\\\\\n\\midrule\n")
        for _, r in bt.iterrows():
            f.write(f"{r['model']} & {r['bw']} & {r['noise']} & {r['n_pairs']} & "
                    f"{r['mean_diff']:.4f} & {r['ci_low']:.4f} & "
                    f"{r['ci_high']:.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # ---------------- Table_E3_stability ----------------
    srows = []
    for mid in MODELS:
        for bw in BW_ORDER:
            g = df[(df.model == mid) & (df.bw_key == bw)]
            succ = g[g.status == "SUCCESS"]
            srows.append({
                "model": MODEL_LABEL[mid], "bandwidth": BW_LABEL[bw],
                "attempted": len(g),
                "training_failure": (g.status == "TRAINING_FAILURE").sum(),
                "failure_rate": round((g.status != "SUCCESS").mean(), 3),
                "collapsed_runs": (succ.v_rmse > 1.0).sum(),
                "collapsed_rate": round((succ.v_rmse > 1.0).mean(), 3),
                "mean_runtime_s": round(succ.runtime_seconds.mean(), 1) if len(succ) else np.nan,
            })
    st = pd.DataFrame(srows)
    st.to_csv(os.path.join(TAB_DIR, "Table_E3_stability.csv"), index=False)
    with open(os.path.join(TAB_DIR, "Table_E3_stability.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n"
                "\\caption{Training stability by bandwidth (14-bus, all 4 noise "
                "conditions pooled; collapsed = any successful run with "
                "V\\_RMSE > 1.0 on some noise condition).}\n"
                "\\label{tab:E3_stability}\n\\footnotesize\n"
                "\\begin{tabular}{llrrrrr}\n\\toprule\n"
                "Model & Bandwidth & Attempted & TrainF & FailRate & Collapsed "
                "& Runtime(s) \\\\\n\\midrule\n")
        for _, r in st.iterrows():
            f.write(f"{r['model']} & {r['bandwidth']} & {r['attempted']} & "
                    f"{r['training_failure']} & {r['failure_rate']:.2f} & "
                    f"{r['collapsed_runs']} & {r['mean_runtime_s']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # ---------------- figures ----------------
    xs = np.arange(len(NOISE_ORDER)); w = 0.2
    for metric, ylab, fname in [("v_rmse", "V RMSE (pu)", "fig_E3_vrmse.pdf"),
                                ("th_rmse", "theta RMSE (rad)", "fig_E3_thrmse.pdf")]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
        for ax, mid in zip(axes, MODELS):
            for i, bw in enumerate(BW_ORDER):
                vals, errs = [], []
                for noise in NOISE_ORDER:
                    succ = df[(df.model == mid) & (df.bw_key == bw)
                              & (df.noise == noise) & (df.status == "SUCCESS")]
                    vals.append(succ[metric].median() if len(succ) else np.nan)
                    errs.append(0)
                ax.bar(xs + (i - 1.5) * w, vals, w, label=BW_LABEL[bw])
            ax.set_xticks(xs); ax.set_xticklabels(
                [NOISE_LABEL[n] for n in NOISE_ORDER], rotation=20)
            ax.set_ylabel(ylab); ax.set_title(MODEL_LABEL[mid])
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, fname)); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, mid in zip(axes, MODELS):
        for i, bw in enumerate(BW_ORDER):
            rates = []
            for noise in NOISE_ORDER:
                g = df[(df.model == mid) & (df.bw_key == bw)
                       & (df.noise == noise)]
                succ = g[g.status == "SUCCESS"]
                rates.append((succ.v_rmse > 1.0).mean() if len(succ) else np.nan)
            ax.plot(xs, rates, marker="o", label=BW_LABEL[bw])
        ax.set_xticks(xs); ax.set_xticklabels(
            [NOISE_LABEL[n] for n in NOISE_ORDER], rotation=20)
        ax.set_ylabel("Collapse rate (V_RMSE>1.0)"); ax.set_ylim(-0.05, 1.05)
        ax.set_title(MODEL_LABEL[mid]); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_E3_collapse_rate.pdf")); plt.close(fig)

    # annealing trajectory (b_t vs epoch) from history files
    fig, ax = plt.subplots(figsize=(6, 4))
    for mid in MODELS:
        p = os.path.join(HIST_DIR, f"{mid}_anneal_8_2_rep0.csv")
        if os.path.exists(p):
            h = pd.read_csv(p)
            ax.plot(h.epoch, h.b, label=MODEL_LABEL[mid])
    ax.set_xlabel("epoch"); ax.set_ylabel("bandwidth b(epoch)")
    ax.set_title("Annealing trajectory (rep 0)")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_E3_anneal_trajectory.pdf"))
    plt.close(fig)

    print("E3 tables + figures written")


if __name__ == "__main__":
    main()
