"""
E7 post-processing: multi-start tables + figures + paired bootstrap.
All outputs derived programmatically from e7_start_level.csv and
e7_selected_solution.csv.
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

RAW_DIR = os.path.join(EXP, "raw_results", "E7")
START_CSV = os.path.join(RAW_DIR, "e7_start_level.csv")
SEL_CSV = os.path.join(RAW_DIR, "e7_selected_solution.csv")
TAB_DIR = os.path.join(EXP, "tables", "E7")
FIG_DIR = os.path.join(EXP, "figures", "E7")

NOISE_ORDER = ["default", "mixture_010", "mixture_020", "erroneous"]
NOISE_LABEL = {"default": "Default", "mixture_010": "10% Mixture",
               "mixture_020": "20% Mixture", "erroneous": "Erroneous"}
SOLVER_LABEL = {"wls": "WLS", "modal_wls": "Modal-WLS"}
K_VALUES = [1, 3, 5, 10]


def main():
    os.makedirs(TAB_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    st = pd.read_csv(START_CSV)
    sel = pd.read_csv(SEL_CSV)

    # derive K in {1,3,5,10} selected solutions from the recorded start-level
    # rows (protocol: first K starts; objective-based selection among
    # converged starts).  sel already holds K=10 rows from the runner.
    derived = []
    for (solver, noise, si, di), g in st.groupby(
            ["solver", "noise", "sample_idx", "draw_idx"]):
        for K in [1, 3, 5]:
            sub = g[g.start_id <= K]
            conv = sub[(sub.converged == 1) & sub.objective.notna()]
            if len(conv) == 0:
                derived.append(dict(solver=solver, noise=noise, sample_idx=si,
                                    draw_idx=di, K=K,
                                    multistart_status="MULTISTART_FAILURE",
                                    selected_start=-1, v_rmse=np.nan,
                                    th_rmse=np.nan,
                                    runtime_s_total=sub.runtime_s.sum()))
                continue
            best = conv.sort_values("objective").iloc[0]
            derived.append(dict(solver=solver, noise=noise, sample_idx=si,
                                draw_idx=di, K=K,
                                multistart_status="MULTISTART_SUCCESS",
                                selected_start=int(best.start_id),
                                v_rmse=best.v_rmse, th_rmse=best.th_rmse,
                                runtime_s_total=sub.runtime_s.sum()))
    sel = pd.concat([sel, pd.DataFrame(derived)], ignore_index=True)

    # ---------------- Table E7-1: convergence ----------------
    rows = []
    for solver in ["wls", "modal_wls"]:
        for noise in NOISE_ORDER:
            s = st[(st.solver == solver) & (st.noise == noise)]
            n_units = s.groupby(["sample_idx", "draw_idx"]).ngroups
            per_start = s.converged.mean()
            num_fail = (s.status == "NUMERICAL_FAILURE").mean()
            for K in K_VALUES:
                sub = s[s.start_id <= K]
                ms_succ = sub.groupby(["sample_idx", "draw_idx"]).converged.max()
                rows.append({
                    "Noise": NOISE_LABEL[noise], "Solver": SOLVER_LABEL[solver],
                    "K": K, "per_start_conv": round(per_start, 3),
                    "multistart_success": round(float(ms_succ.mean()), 3),
                    "numerical_failure_rate": round(num_fail, 3),
                    "theoretical_1minus(1-p)^K": round(
                        1 - (1 - per_start) ** K, 3),
                })
    t1 = pd.DataFrame(rows)
    t1.to_csv(os.path.join(TAB_DIR, "Table_E7_convergence.csv"), index=False)
    with open(os.path.join(TAB_DIR, "Table_E7_convergence.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n"
                "\\caption{Multi-start convergence (10 samples x 5 draws). "
                "Per-start rate over all starts; multi-start success = at "
                "least one converged start among the first K. Theoretical "
                "column assumes independent starts.}\n"
                "\\label{tab:E7_convergence}\n\\footnotesize\n"
                "\\begin{tabular}{llrrrrr}\n\\toprule\n"
                "Noise & Solver & K & PerStart & MultiSucc & NumFail & "
                "$1-(1-p)^K$ \\\\\n\\midrule\n")
        for _, r in t1.iterrows():
            f.write(f"{r['Noise']} & {r['Solver']} & {r['K']} & "
                    f"{r['per_start_conv']:.2f} & {r['multistart_success']:.2f} & "
                    f"{r['numerical_failure_rate']:.2f} & "
                    f"{r['theoretical_1minus(1-p)^K']:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # ---------------- Table E7-2: accuracy ----------------
    rows = []
    for solver in ["wls", "modal_wls"]:
        for noise in NOISE_ORDER:
            s = sel[(sel.solver == solver) & (sel.noise == noise)]
            for K in K_VALUES:
                g = s[s.K == K].dropna(subset=["v_rmse"])
                if len(g) == 0:
                    continue
                rows.append({
                    "Noise": NOISE_LABEL[noise], "Solver": SOLVER_LABEL[solver],
                    "K": K, "n_success": len(g),
                    "v_mean": round(g.v_rmse.mean(), 5),
                    "v_sd": round(g.v_rmse.std(), 5),
                    "v_median": round(g.v_rmse.median(), 5),
                    "th_mean": round(g.th_rmse.mean(), 5),
                    "th_sd": round(g.th_rmse.std(), 5),
                    "th_median": round(g.th_rmse.median(), 5),
                })
    t2 = pd.DataFrame(rows)
    t2.to_csv(os.path.join(TAB_DIR, "Table_E7_accuracy.csv"), index=False)
    with open(os.path.join(TAB_DIR, "Table_E7_accuracy.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n"
                "\\caption{Selected-solution accuracy by K (objective-based "
                "selection among converged starts).}\n"
                "\\label{tab:E7_accuracy}\n\\footnotesize\n"
                "\\begin{tabular}{llrrrrrrr}\n\\toprule\n"
                "Noise & Solver & K & n & V mean & V sd & V med & "
                "$\\theta$ mean & $\\theta$ sd \\\\\n\\midrule\n")
        for _, r in t2.iterrows():
            f.write(f"{r['Noise']} & {r['Solver']} & {r['K']} & {r['n_success']} & "
                    f"{r['v_mean']:.4f} & {r['v_sd']:.4f} & {r['v_median']:.4f} & "
                    f"{r['th_mean']:.4f} & {r['th_sd']:.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # ---------------- Table E7-3: runtime ----------------
    rows = []
    for solver in ["wls", "modal_wls"]:
        for noise in NOISE_ORDER:
            s = st[(st.solver == solver) & (st.noise == noise)]
            for K in K_VALUES:
                sub = s[s.start_id <= K]
                rt = sel[(sel.solver == solver) & (sel.noise == noise)
                         & (sel.K == K)]
                rows.append({
                    "Noise": NOISE_LABEL[noise], "Solver": SOLVER_LABEL[solver],
                    "K": K, "mean_start_s": round(sub.runtime_s.mean(), 3),
                    "median_start_s": round(sub.runtime_s.median(), 3),
                    "mean_iter": round(sub.n_iter.mean(), 1),
                    "unit_total_s": round(rt.runtime_s_total.mean(), 2),
                })
    t3 = pd.DataFrame(rows)
    t3.to_csv(os.path.join(TAB_DIR, "Table_E7_runtime.csv"), index=False)
    with open(os.path.join(TAB_DIR, "Table_E7_runtime.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n"
                "\\caption{Runtime by K.}\n\\label{tab:E7_runtime}\n"
                "\\footnotesize\n\\begin{tabular}{llrrrrr}\n\\toprule\n"
                "Noise & Solver & K & MeanStart(s) & MedStart(s) & MeanIter & "
                "UnitTotal(s) \\\\\n\\midrule\n")
        for _, r in t3.iterrows():
            f.write(f"{r['Noise']} & {r['Solver']} & {r['K']} & "
                    f"{r['mean_start_s']:.3f} & {r['median_start_s']:.3f} & "
                    f"{r['mean_iter']:.1f} & {r['unit_total_s']:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # ---------------- Table E7-4: paired WLS vs Modal-WLS ----------------
    rows = []
    rng = np.random.RandomState(0)
    for noise in NOISE_ORDER:
        for K in K_VALUES:
            a = sel[(sel.solver == "wls") & (sel.noise == noise) & (sel.K == K)]
            b = sel[(sel.solver == "modal_wls") & (sel.noise == noise) & (sel.K == K)]
            merged = a.merge(b, on=["sample_idx", "draw_idx"],
                             suffixes=("_wls", "_modal"))
            merged = merged.dropna(subset=["v_rmse_wls", "v_rmse_modal"])
            if len(merged) < 2:
                continue
            d = (merged.v_rmse_wls - merged.v_rmse_modal).values
            boot = np.array([rng.choice(d, len(d), replace=True).mean()
                             for _ in range(10000)])
            rows.append({
                "Noise": NOISE_LABEL[noise], "K": K, "n_pairs": len(d),
                "mean_diff": d.mean(),
                "ci_low": np.percentile(boot, 2.5),
                "ci_high": np.percentile(boot, 97.5),
            })
    t4 = pd.DataFrame(rows)
    t4.to_csv(os.path.join(TAB_DIR, "Table_E7_paired_comparison.csv"), index=False)
    with open(os.path.join(TAB_DIR, "Table_E7_paired_comparison.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n"
                "\\caption{Paired bootstrap (10k): V\\_RMSE(WLS) - "
                "V\\_RMSE(Modal-WLS) on shared draws; positive = Modal-WLS "
                "better.}\n\\label{tab:E7_paired}\n\\footnotesize\n"
                "\\begin{tabular}{lrrrrr}\n\\toprule\n"
                "Noise & K & n & MeanDiff & CI low & CI high \\\\\n\\midrule\n")
        for _, r in t4.iterrows():
            f.write(f"{r['Noise']} & {r['K']} & {r['n_pairs']} & "
                    f"{r['mean_diff']:.4f} & {r['ci_low']:.4f} & "
                    f"{r['ci_high']:.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # ---------------- figures ----------------
    xs = np.arange(len(K_VALUES))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, solver in zip(axes, ["wls", "modal_wls"]):
        for noise in NOISE_ORDER:
            vals = []
            for K in K_VALUES:
                s = st[(st.solver == solver) & (st.noise == noise)]
                sub = s[s.start_id <= K]
                vals.append(sub.groupby(["sample_idx", "draw_idx"])
                           .converged.max().mean())
            ax.plot(xs, vals, marker="o", label=NOISE_LABEL[noise])
        ax.set_xticks(xs); ax.set_xticklabels([f"K={k}" for k in K_VALUES])
        ax.set_ylabel("multi-start success rate")
        ax.set_title(SOLVER_LABEL[solver]); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_E7_convergence.pdf")); plt.close(fig)

    for metric, ylab, fname in [("v_rmse", "V RMSE (pu)", "fig_E7_vrmse.pdf"),
                                ("th_rmse", "theta RMSE (rad)", "fig_E7_thrmse.pdf")]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, solver in zip(axes, ["wls", "modal_wls"]):
            for noise in NOISE_ORDER:
                vals = []
                for K in K_VALUES:
                    g = sel[(sel.solver == solver) & (sel.noise == noise)
                            & (sel.K == K)].dropna(subset=[metric])
                    vals.append(g[metric].median() if len(g) else np.nan)
                ax.plot(xs, vals, marker="o", label=NOISE_LABEL[noise])
            ax.set_xticks(xs); ax.set_xticklabels([f"K={k}" for k in K_VALUES])
            ax.set_ylabel(ylab + " (median)")
            ax.set_title(SOLVER_LABEL[solver]); ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, fname)); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, solver in zip(axes, ["wls", "modal_wls"]):
        for noise in NOISE_ORDER:
            vals = []
            for K in K_VALUES:
                g = sel[(sel.solver == solver) & (sel.noise == noise)
                        & (sel.K == K)]
                vals.append(g.runtime_s_total.median())
            ax.plot(xs, vals, marker="o", label=NOISE_LABEL[noise])
        ax.set_xticks(xs); ax.set_xticklabels([f"K={k}" for k in K_VALUES])
        ax.set_ylabel("unit runtime (s, median)")
        ax.set_title(SOLVER_LABEL[solver]); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_E7_runtime.pdf")); plt.close(fig)

    # E7-5: severe contamination — convergence benefit vs cost
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, solver in zip(axes, ["wls", "modal_wls"]):
        noise = "mixture_020"
        conv, cost = [], []
        for K in K_VALUES:
            s = st[(st.solver == solver) & (st.noise == noise)]
            sub = s[s.start_id <= K]
            conv.append(sub.groupby(["sample_idx", "draw_idx"])
                        .converged.max().mean())
            g = sel[(sel.solver == solver) & (sel.noise == noise)
                    & (sel.K == K)]
            cost.append(g.runtime_s_total.median())
        ax.plot(cost, conv, marker="o")
        for K, c, v in zip(K_VALUES, cost, conv):
            ax.annotate(f"K={K}", (c, v), fontsize=8)
        ax.set_xlabel("median runtime (s)"); ax.set_ylabel("multi-start success")
        ax.set_title(f"{SOLVER_LABEL[solver]}, 20% mixture")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_E7_benefit_cost.pdf")); plt.close(fig)

    print("E7 tables + figures written")


if __name__ == "__main__":
    main()
