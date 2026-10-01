"""
E6 post-processing: cross-system tables + figures + paired bootstrap.
All outputs derived programmatically from E6_raw_results.csv.
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

RAW_CSV = os.path.join(EXP, "raw_results", "E6", "E6_raw_results.csv")
TAB_DIR = os.path.join(EXP, "tables", "E6")
FIG_DIR = os.path.join(EXP, "figures", "E6")

NOISE_ORDER = ["default", "mixture_010", "erroneous"]
NOISE_LABEL = {"default": "Default", "mixture_010": "10% Mixture",
               "erroneous": "Erroneous"}
MODEL_LABEL = {"supervised_ann": "S-ANN", "modal_supervised_ann": "M-ANN",
               "supervised_dss2": "S-DSS2", "modal_supervised_dss2": "M-DSS2"}


def paired_diff(df, dataset, ma, mb, metric="v_rmse"):
    rows = []
    rng = np.random.RandomState(0)
    for noise in NOISE_ORDER:
        a = df[(df.dataset_id == dataset) & (df.model == ma)
               & (df.noise == noise) & (df.status == "SUCCESS")]
        b = df[(df.dataset_id == dataset) & (df.model == mb)
               & (df.noise == noise) & (df.status == "SUCCESS")]
        a = a.sort_values("repeat_id")[metric].values
        b = b.sort_values("repeat_id")[metric].values
        n = min(len(a), len(b))
        if n < 2:
            continue
        d = a[:n] - b[:n]
        boot = np.array([rng.choice(d, n, replace=True).mean()
                         for _ in range(10000)])
        rows.append({"noise": noise, "n_pairs": n, "mean_diff": d.mean(),
                     "ci_low": np.percentile(boot, 2.5),
                     "ci_high": np.percentile(boot, 97.5)})
    return pd.DataFrame(rows)


def main():
    os.makedirs(TAB_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(RAW_CSV)

    # ---------------- Table E6-A (IEEE-30) ----------------
    rows = []
    for noise in NOISE_ORDER:
        rec = {"Noise": NOISE_LABEL[noise]}
        for mid in ["supervised_ann", "modal_supervised_ann",
                    "supervised_dss2", "modal_supervised_dss2"]:
            g = df[(df.dataset_id == "ieee30") & (df.model == mid)
                   & (df.noise == noise)]
            succ = g[g.status == "SUCCESS"]
            n_fail = (g.status != "SUCCESS").sum()
            cell = (f"{succ.v_rmse.mean():.4f}$\\pm${succ.v_rmse.std():.4f}"
                    + (f" ({n_fail}f)" if n_fail else ""))
            rec[MODEL_LABEL[mid]] = cell
        rows.append(rec)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TAB_DIR, "Table_E6_IEEE30.csv"), index=False)
    with open(os.path.join(TAB_DIR, "Table_E6_IEEE30.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n"
                "\\caption{IEEE-30 V\\_RMSE (4 learning models, 5 independent "
                "training runs, 200 epochs, b=1.0, $\\lambda_{\\mathrm{meas}}$=0). "
                "Mean$\\pm$std.}\n\\label{tab:E6_ieee30}\n\\footnotesize\n"
                "\\begin{tabular}{lcccc}\n\\toprule\n"
                "\\textbf{Noise} & \\textbf{S-ANN} & \\textbf{M-ANN} & "
                "\\textbf{S-DSS2} & \\textbf{M-DSS2} \\\\\n\\midrule\n")
        for _, r in t.iterrows():
            f.write(f"{r['Noise']} & {r['S-ANN']} & {r['M-ANN']} & "
                    f"{r['S-DSS2']} & {r['M-DSS2']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # ---------------- Table E6-B (PanTaGruEl) ----------------
    pb = paired_diff(df, "pantagruel_14", "supervised_ann",
                     "modal_supervised_ann")
    rows = []
    for noise in NOISE_ORDER:
        for mid, lab in [("supervised_ann", "S-ANN (squared)"),
                         ("modal_supervised_ann", "M-ANN (modal)")]:
            g = df[(df.dataset_id == "pantagruel_14") & (df.model == mid)
                   & (df.noise == noise)]
            succ = g[g.status == "SUCCESS"]
            rows.append({"Noise": NOISE_LABEL[noise], "Estimator": lab,
                         "V_RMSE": f"{succ.v_rmse.mean():.4f}$\\pm$"
                                   f"{succ.v_rmse.std():.4f}",
                         "theta_RMSE": f"{succ.th_rmse.mean():.4f}$\\pm$"
                                       f"{succ.th_rmse.std():.4f}"})
    t2 = pd.DataFrame(rows)
    t2.to_csv(os.path.join(TAB_DIR, "Table_E6_PanTaGruEl.csv"), index=False)
    with open(os.path.join(TAB_DIR, "Table_E6_PanTaGruEl.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n"
                "\\caption{PanTaGruEl IEEE-14 external benchmark (Level-3, "
                "Zenodo 5816149). 5 independent training runs, 200 epochs.}\n"
                "\\label{tab:E6_pantagruel}\n\\footnotesize\n"
                "\\begin{tabular}{llll}\n\\toprule\n"
                "\\textbf{Noise} & \\textbf{Estimator} & \\textbf{V\\_RMSE} & "
                "\\textbf{$\\theta$\\_RMSE} \\\\\n\\midrule\n")
        for _, r in t2.iterrows():
            f.write(f"{r['Noise']} & {r['Estimator']} & {r['V_RMSE']} & "
                    f"{r['theta_RMSE']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    pb.to_csv(os.path.join(TAB_DIR, "Table_E6_PanTaGruEl_diff.csv"), index=False)

    # ---------------- Table E6-C (cross-system) ----------------
    rows = []
    for dataset, dsname in [("cigre14", "CIGRE-14"), ("ieee30", "IEEE-30"),
                            ("ober_sub", "Oberrhein-70"),
                            ("pantagruel_14", "PanTaGruEl-14")]:
        src = None
        if dataset == "cigre14":
            src = os.path.join(EXP, "raw_results", "E1", "E1_raw_results.csv")
        elif dataset == "ober_sub":
            src = os.path.join(EXP, "raw_results", "E2", "E2_raw_results.csv")
        else:
            src = RAW_CSV
        d = pd.read_csv(src)
        has_status = "status" in d.columns
        for noise in NOISE_ORDER:
            for mid in ["supervised_ann", "modal_supervised_ann"]:
                base = d[(d.model == mid) & (d.noise == noise)]
                g = base[base.status == "SUCCESS"] if has_status else base
                if len(g) == 0:
                    continue
                fail_rate = ((base.status != "SUCCESS").mean()
                             if has_status else 0.0)
                rows.append({"Dataset": dsname, "Noise": NOISE_LABEL[noise],
                             "Model": MODEL_LABEL[mid], "n": len(g),
                             "V_RMSE": round(g.v_rmse.mean(), 5),
                             "theta_RMSE": round(g.th_rmse.mean(), 5),
                             "FailureRate": round(fail_rate, 3)})
    t3 = pd.DataFrame(rows)
    t3.to_csv(os.path.join(TAB_DIR, "Table_E6_cross_system.csv"), index=False)

    # ---------------- figures ----------------
    # E6-1/E6-2: IEEE-30 V/theta RMSE
    for metric, ylab, fname in [("v_rmse", "V RMSE (pu)", "fig_E6_ieee30_vrmse.pdf"),
                                ("th_rmse", "theta RMSE (rad)", "fig_E6_ieee30_thrmse.pdf")]:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        xs = np.arange(len(NOISE_ORDER)); w = 0.2
        for i, mid in enumerate(["supervised_ann", "modal_supervised_ann",
                                 "supervised_dss2", "modal_supervised_dss2"]):
            vals, errs = [], []
            for noise in NOISE_ORDER:
                succ = df[(df.dataset_id == "ieee30") & (df.model == mid)
                          & (df.noise == noise) & (df.status == "SUCCESS")]
                vals.append(succ[metric].mean() if len(succ) else np.nan)
                errs.append(succ[metric].std() if len(succ) else 0)
            ax.bar(xs + (i - 1.5) * w, vals, w, yerr=errs, capsize=3,
                   label=MODEL_LABEL[mid])
        ax.set_xticks(xs); ax.set_xticklabels([NOISE_LABEL[n] for n in NOISE_ORDER])
        ax.set_ylabel(ylab); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, fname))
        plt.close(fig)

    # E6-3: PanTaGruEl V RMSE
    fig, ax = plt.subplots(figsize=(6.5, 4))
    xs = np.arange(len(NOISE_ORDER)); w = 0.35
    for i, mid in enumerate(["supervised_ann", "modal_supervised_ann"]):
        vals, errs = [], []
        for noise in NOISE_ORDER:
            succ = df[(df.dataset_id == "pantagruel_14") & (df.model == mid)
                      & (df.noise == noise) & (df.status == "SUCCESS")]
            vals.append(succ.v_rmse.mean() if len(succ) else np.nan)
            errs.append(succ.v_rmse.std() if len(succ) else 0)
        ax.bar(xs + (i - 0.5) * w, vals, w, yerr=errs, capsize=3,
               label=MODEL_LABEL[mid])
    ax.set_xticks(xs); ax.set_xticklabels([NOISE_LABEL[n] for n in NOISE_ORDER])
    ax.set_ylabel("V RMSE (pu)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_E6_pantagruel_vrmse.pdf"))
    plt.close(fig)

    # E6-4: cross-system effect size (S - M, V_RMSE) with 95% CI
    fig, ax = plt.subplots(figsize=(8, 4))
    datasets = [("cigre14", "CIGRE-14", os.path.join(EXP, "raw_results", "E1", "E1_raw_results.csv")),
                ("ieee30", "IEEE-30", RAW_CSV),
                ("ober_sub", "Oberrhein-70", os.path.join(EXP, "raw_results", "E2", "E2_raw_results.csv")),
                ("pantagruel_14", "PanTaGruEl-14", RAW_CSV)]
    xpos, labels = [], []
    for k, (dsid, dsname, src) in enumerate(datasets):
        d = pd.read_csv(src)
        for j, noise in enumerate(["default", "mixture_010", "erroneous"]):
            has_status = "status" in d.columns
            a = d[(d.model == "supervised_ann") & (d.noise == noise)]
            b = d[(d.model == "modal_supervised_ann") & (d.noise == noise)]
            if has_status:
                a = a[a.status == "SUCCESS"]
                b = b[b.status == "SUCCESS"]
            a = a.sort_values("repeat_id").v_rmse.values
            b = b.sort_values("repeat_id").v_rmse.values
            n = min(len(a), len(b))
            if n < 2:
                continue
            dvec = a[:n] - b[:n]
            rng = np.random.RandomState(0)
            boot = np.array([rng.choice(dvec, n, replace=True).mean()
                             for _ in range(10000)])
            x = k * 3 + j
            xpos.append(x)
            labels.append(f"{dsname}\n{NOISE_LABEL[noise]}")
            ax.errorbar(x, dvec.mean(),
                        yerr=[[dvec.mean() - np.percentile(boot, 2.5)],
                              [np.percentile(boot, 97.5) - dvec.mean()]],
                        fmt="o", capsize=3)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("S-ANN - M-ANN V_RMSE (positive = modal better)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_E6_cross_system_effect.pdf"))
    plt.close(fig)

    print("E6 tables + figures written")


if __name__ == "__main__":
    main()
