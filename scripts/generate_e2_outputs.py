"""
E2 post-processing: failure-aware tables + figures.

Produces (all derived programmatically from E2_raw_results.csv):
  tables/E2/Table_E2_70bus.csv / .tex   (per model x noise, failure-aware)
  tables/E2/Table_2_REBUILT.csv / .tex (paper-style V_RMSE table)
  figures/E2/fig_E2_vrmse.pdf, fig_E2_thrmse.pdf, fig_E2_failure_rate.pdf
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

RAW_CSV = os.path.join(EXP, "raw_results", "E2", "E2_raw_results.csv")
TAB_DIR = os.path.join(EXP, "tables", "E2")
FIG_DIR = os.path.join(EXP, "figures", "E2")
E2_NOISES = ["default", "student_t3", "mixture_010", "erroneous",
             "missing_erroneous"]
MODEL_ORDER = ["supervised_ann", "modal_supervised_ann",
               "supervised_dss2", "modal_supervised_dss2"]
MODEL_LABEL = {"supervised_ann": "S-ANN", "modal_supervised_ann": "M-ANN",
               "supervised_dss2": "S-DSS2", "modal_supervised_dss2": "M-DSS2"}
NOISE_LABEL = {"default": "Default", "student_t3": "Student-t3",
               "mixture_010": "10% Mixture", "erroneous": "Erroneous",
               "missing_erroneous": "Miss+Err"}


def main():
    os.makedirs(TAB_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(RAW_CSV)

    # ---------------- failure-aware table ----------------
    rows = []
    for mid in MODEL_ORDER:
        for noise in E2_NOISES:
            g = df[(df.model == mid) & (df.noise == noise)]
            attempted = len(g)
            succ = g[g.status == "SUCCESS"]
            n_succ = len(succ)
            n_train_fail = (g.status == "TRAINING_FAILURE").sum()
            n_eval_fail = (g.status == "EVALUATION_FAILURE").sum()
            n_tech = (g.status == "TECHNICAL_ERROR").sum()
            rec = {
                "model": MODEL_LABEL[mid], "noise": NOISE_LABEL[noise],
                "attempted": attempted, "success": n_succ,
                "training_failure": n_train_fail,
                "evaluation_failure": n_eval_fail,
                "technical_error": n_tech,
                "failure_rate": round((attempted - n_succ) / attempted, 3),
                "v_rmse_mean": round(succ.v_rmse.mean(), 5) if n_succ else np.nan,
                "v_rmse_sd": round(succ.v_rmse.std(), 5) if n_succ else np.nan,
                "v_rmse_median": round(succ.v_rmse.median(), 5) if n_succ else np.nan,
                "v_rmse_min": round(succ.v_rmse.min(), 5) if n_succ else np.nan,
                "v_rmse_max": round(succ.v_rmse.max(), 5) if n_succ else np.nan,
                "th_rmse_mean": round(succ.th_rmse.mean(), 5) if n_succ else np.nan,
                "th_rmse_sd": round(succ.th_rmse.std(), 5) if n_succ else np.nan,
            }
            rows.append(rec)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TAB_DIR, "Table_E2_70bus.csv"), index=False)
    with open(os.path.join(TAB_DIR, "Table_E2_70bus.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n"
                "\\caption{70-Bus fair comparison (5 independent training runs, "
                "100 epochs). Failure-aware: metrics over successful runs; "
                "failure rate reported per cell.}\n\\label{tab:E2_70bus}\n"
                "\\footnotesize\n\\begin{tabular}{llrrrrrrrr}\n\\toprule\n"
                "Model & Noise & Att. & Succ. & TrainF & FailRate & "
                "V mean & V sd & $\\theta$ mean & $\\theta$ sd \\\\\n\\midrule\n")
        for _, r in t.iterrows():
            f.write(f"{r['model']} & {r['noise']} & {r['attempted']} & "
                    f"{r['success']} & {r['training_failure']} & "
                    f"{r['failure_rate']:.2f} & {r['v_rmse_mean']:.4f} & "
                    f"{r['v_rmse_sd']:.4f} & {r['th_rmse_mean']:.4f} & "
                    f"{r['th_rmse_sd']:.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # ---------------- rebuilt Table 2 (paper style) ----------------
    t2 = pd.DataFrame(columns=["Noise"] + [MODEL_LABEL[m] for m in MODEL_ORDER])
    t2_rows = []
    for noise in E2_NOISES:
        rec = {"Noise": NOISE_LABEL[noise]}
        for mid in MODEL_ORDER:
            g = df[(df.model == mid) & (df.noise == noise)]
            succ = g[g.status == "SUCCESS"]
            n_succ = len(succ)
            n_fail = (g.status != "SUCCESS").sum()
            if n_succ == 0:
                rec[MODEL_LABEL[mid]] = "FAILED"
            elif n_fail > 0:
                rec[MODEL_LABEL[mid]] = (f"{succ.v_rmse.mean():.4f}"
                                         f"$\\pm${succ.v_rmse.std():.4f}"
                                         f" ({n_fail}f)")
            else:
                rec[MODEL_LABEL[mid]] = (f"{succ.v_rmse.mean():.4f}"
                                         f"$\\pm${succ.v_rmse.std():.4f}")
        t2_rows.append(rec)
    t2 = pd.DataFrame(t2_rows)
    t2.to_csv(os.path.join(TAB_DIR, "Table_2_REBUILT.csv"), index=False)
    with open(os.path.join(TAB_DIR, "Table_2_REBUILT.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n"
                "\\caption{70-Bus V\\_RMSE (learning models, 5 independent "
                "training runs, 100 epochs). Mean$\\pm$std over successful "
                "runs; failed runs marked. Old Table 2 values are superseded "
                "(see TABLE2_REBUILD_AUDIT.md).}\n\\label{tab:70bus_rebuilt}\n"
                "\\footnotesize\n\\begin{tabular}{lcccc}\n\\toprule\n"
                "\\textbf{Noise} & \\textbf{S-ANN} & \\textbf{M-ANN} & "
                "\\textbf{S-DSS2} & \\textbf{M-DSS2} \\\\\n\\midrule\n")
        for _, r in t2.iterrows():
            f.write(f"{r['Noise']} & {r['S-ANN']} & {r['M-ANN']} & "
                    f"{r['S-DSS2']} & {r['M-DSS2']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # ---------------- figures ----------------
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(E2_NOISES)); w = 0.2
    for i, mid in enumerate(MODEL_ORDER):
        vals, errs = [], []
        for noise in E2_NOISES:
            succ = df[(df.model == mid) & (df.noise == noise)
                      & (df.status == "SUCCESS")]
            vals.append(succ.v_rmse.mean() if len(succ) else np.nan)
            errs.append(succ.v_rmse.std() if len(succ) else 0)
        ax.bar(xs + (i - 1.5) * w, vals, w, yerr=errs, capsize=3,
               label=MODEL_LABEL[mid])
    ax.set_xticks(xs); ax.set_xticklabels([NOISE_LABEL[n] for n in E2_NOISES],
                                          rotation=20)
    ax.set_ylabel("V RMSE (pu)"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_E2_vrmse.pdf")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, mid in enumerate(MODEL_ORDER):
        vals, errs = [], []
        for noise in E2_NOISES:
            succ = df[(df.model == mid) & (df.noise == noise)
                      & (df.status == "SUCCESS")]
            vals.append(succ.th_rmse.mean() if len(succ) else np.nan)
            errs.append(succ.th_rmse.std() if len(succ) else 0)
        ax.bar(xs + (i - 1.5) * w, vals, w, yerr=errs, capsize=3,
               label=MODEL_LABEL[mid])
    ax.set_xticks(xs); ax.set_xticklabels([NOISE_LABEL[n] for n in E2_NOISES],
                                          rotation=20)
    ax.set_ylabel("theta RMSE (rad)"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_E2_thrmse.pdf")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, mid in enumerate(MODEL_ORDER):
        rates = []
        for noise in E2_NOISES:
            g = df[(df.model == mid) & (df.noise == noise)]
            rates.append((g.status != "SUCCESS").mean() if len(g) else np.nan)
        ax.plot(xs, rates, marker="o", label=MODEL_LABEL[mid])
    ax.set_xticks(xs); ax.set_xticklabels([NOISE_LABEL[n] for n in E2_NOISES],
                                          rotation=20)
    ax.set_ylabel("Training-failure rate"); ax.set_ylim(-0.05, 1.05)
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_E2_failure_rate.pdf")); plt.close(fig)

    print("E2 tables + figures written:")
    for p in ["Table_E2_70bus.csv", "Table_E2_70bus.tex",
              "Table_2_REBUILT.csv", "Table_2_REBUILT.tex"]:
        print("  tables/E2/" + p)
    for p in ["fig_E2_vrmse.pdf", "fig_E2_thrmse.pdf",
              "fig_E2_failure_rate.pdf"]:
        print("  figures/E2/" + p)


if __name__ == "__main__":
    main()
