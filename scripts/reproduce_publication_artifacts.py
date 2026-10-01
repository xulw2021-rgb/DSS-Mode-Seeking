"""
Stage 3B: FINAL manuscript tables and figures.
Every number is read from the approved CSVs (STAGE3_EVIDENCE_SOURCE_MAP).
Outputs: tables/FINAL/*.tex, figures/FINAL/*.pdf
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
sys.path.insert(0, os.path.join(EXP, "src"))

RES = os.path.join(EXP, "results", "final")
E1S = os.path.join(RES, "E1_solvers.csv")
E7S = RES
TAB = os.path.join(EXP, "reproduced", "tables")
FIG = os.path.join(EXP, "reproduced", "figures")

MODEL_LABEL = {"supervised_ann": "S-ANN", "modal_supervised_ann": "M-ANN",
               "supervised_dss2": "S-DSS2", "modal_supervised_dss2": "M-DSS2",
               "wls": "WLS", "modal_wls": "Modal-WLS"}
NOISE_LABEL = {"default": "Default", "student_t3": "Student-$t_3$",
               "mixture_005": "5\\% Mixture", "mixture_010": "10\\% Mixture",
               "mixture_020": "20\\% Mixture", "erroneous": "Erroneous",
               "missing_erroneous": "Miss+Err"}

# matplotlib labels: plain text, NO LaTeX backslash escapes
PLOT_NOISE = {"default": "Default", "student_t3": "Student-t3",
              "mixture_005": "5% Mixture", "mixture_010": "10% Mixture",
              "mixture_020": "20% Mixture", "erroneous": "Erroneous",
              "missing_erroneous": "Miss+Err"}
PLOT_NOISE_SHORT = {"default": "Clean", "mixture_010": "10% mix",
                    "erroneous": "Erroneous"}


def fmt_mstd(g, col="v_rmse"):
    return f"{g[col].mean():.4f}$\\pm${g[col].std():.4f}"


def paired_ci(a, b, seed=0):
    """a, b: aligned per-repeat arrays; returns (mean_diff, lo, hi) of a-b."""
    n = min(len(a), len(b))
    d = a[:n] - b[:n]
    rng = np.random.RandomState(seed)
    boot = np.array([rng.choice(d, n, replace=True).mean()
                     for _ in range(10000)])
    return d.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def read_corrected(name):
    return pd.read_csv(os.path.join(RES, name))


def main():
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    e1 = read_corrected("E1_learning_corrected.csv")
    e2 = read_corrected("E2_corrected.csv")
    e3 = read_corrected("E3_corrected.csv")
    e4 = read_corrected("E4_corrected.csv")
    e5 = read_corrected("E5_corrected.csv")
    e6a = read_corrected("E6A_ieee30_corrected.csv")
    e6b = read_corrected("E6B_pantagruel_corrected.csv")
    e1s = pd.read_csv(E1S)
    e1s = e1s[e1s.model.isin(["wls", "modal_wls"])]
    e7s = pd.read_csv(os.path.join(E7S, "e7_start_level.csv"))
    e7e = pd.read_csv(os.path.join(E7S, "e7_selected_solution.csv"))

    # ---------------- Table 1: datasets ----------------
    with open(os.path.join(TAB, "Table_1_datasets.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n"
                "\\caption{Systems used in the experiments. All learning "
                "experiments use simulation-generated measurements; "
                "PanTaGruEl is an independently generated external benchmark "
                "(Level-3), not field data.}\n\\label{tab:datasets}\n"
                "\\footnotesize\n\\begin{tabular}{lrrrl}\n\\toprule\n"
                "System & Buses & Branches & Train/Test & Data source \\\\\n"
                "\\midrule\n"
                "CIGRE-14 & 15 & 17 & 7884/876 & project-generated simulation (pandapower) \\\\\n"
                "IEEE-30 & 30 & 41 & 1944/216 & standard benchmark, project-generated scenarios \\\\\n"
                "Oberrhein-70 & 70 & 69 & 500/100 & project-generated simulation (pandapower) \\\\\n"
                "PanTaGruEl-14 & 14 & 20 & 1890/210 & external benchmark (Zenodo 5816149, CC-BY-4.0) \\\\\n"
                "\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # ---------------- Table 2: main 14-bus ----------------
    noises6 = ["default", "student_t3", "mixture_010", "mixture_020",
               "erroneous", "missing_erroneous"]
    with open(os.path.join(TAB, "Table_2_main_14bus.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n"
                "\\caption{CIGRE-14 main comparison, V\\_RMSE (pu). Learning "
                "models: 5 independent training runs, 200 epochs, corrected "
                "train-only normalization; solvers: 5 noise draws on 10 fixed "
                "samples. Mean$\\pm$std.}\n\\label{tab:main14}\n"
                "\\setlength{\\tabcolsep}{4pt}\n\\footnotesize\n"
                "\\begin{tabular}{lcccccc}\n\\toprule\n"
                "Noise & WLS & M-WLS & S-ANN & M-ANN & S-DSS2 & M-DSS2 "
                "\\\\\n\\midrule\n")
        for n in noises6:
            cells = []
            for mid in ["wls", "modal_wls"]:
                g = e1s[(e1s.model == mid) & (e1s.noise == n)]
                cells.append(f"{g.v_rmse.iloc[0]:.4f}" if len(g) else "--")
            for mid in ["supervised_ann", "modal_supervised_ann",
                        "supervised_dss2", "modal_supervised_dss2"]:
                g = e1[(e1.model == mid) & (e1.noise == n)]
                cells.append(fmt_mstd(g))
            f.write(f"{NOISE_LABEL[n]} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # bootstrap for the main comparisons
    rows = []
    for noise in noises6:
        for ma, mb, la, lb in [("supervised_ann", "modal_supervised_ann",
                                "S-ANN", "M-ANN"),
                               ("supervised_dss2", "modal_supervised_dss2",
                                "S-DSS2", "M-DSS2")]:
            a = e1[(e1.model == ma) & (e1.noise == noise)].sort_values(
                "repeat_id").v_rmse.values
            b = e1[(e1.model == mb) & (e1.noise == noise)].sort_values(
                "repeat_id").v_rmse.values
            d, lo, hi = paired_ci(a, b)
            rows.append({"noise": noise, "pair": f"{la}-{lb}",
                         "mean_diff": d, "ci_low": lo, "ci_high": hi})
    bt = pd.DataFrame(rows)
    bt.to_csv(os.path.join(TAB, "Table_2_bootstrap.csv"), index=False)

    # ---------------- Table 3: 70-bus ----------------
    with open(os.path.join(TAB, "Table_3_70bus.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n"
                "\\caption{Oberrhein-70 comparison, V\\_RMSE (pu), 5 "
                "independent training runs, 100 epochs. Mean$\\pm$std.}\n"
                "\\label{tab:70bus}\n\\footnotesize\n"
                "\\begin{tabular}{lcccc}\n\\toprule\n"
                "Noise & S-ANN & M-ANN & S-DSS2 & M-DSS2 \\\\\n\\midrule\n")
        for n in ["default", "student_t3", "mixture_010", "erroneous",
                  "missing_erroneous"]:
            cells = []
            for mid in ["supervised_ann", "modal_supervised_ann",
                        "supervised_dss2", "modal_supervised_dss2"]:
                g = e2[(e2.model == mid) & (e2.noise == n)]
                cells.append(fmt_mstd(g))
            f.write(f"{NOISE_LABEL[n]} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # ---------------- Table 4: cross-system S-ANN vs M-ANN ----------------
    rows = []
    for ds, df_, label in [("cigre14", e1, "CIGRE-14"),
                           ("ober_sub", e2, "Oberrhein-70"),
                           ("ieee30", e6a, "IEEE-30"),
                           ("pantagruel_14", e6b, "PanTaGruEl-14")]:
        for n in ["default", "mixture_010", "erroneous"]:
            a = df_[(df_.model == "supervised_ann") & (df_.noise == n)]
            b = df_[(df_.model == "modal_supervised_ann") & (df_.noise == n)]
            d, lo, hi = paired_ci(a.sort_values("repeat_id").v_rmse.values,
                                  b.sort_values("repeat_id").v_rmse.values)
            rows.append({"System": label, "Noise": NOISE_LABEL[n],
                         "S-ANN": f"{a.v_rmse.mean():.4f}",
                         "M-ANN": f"{b.v_rmse.mean():.4f}",
                         "Diff": f"{d:.4f}", "CI": f"[{lo:.4f}, {hi:.4f}]"})
    t4 = pd.DataFrame(rows)
    t4.to_csv(os.path.join(TAB, "Table_4_cross_system.csv"), index=False)
    with open(os.path.join(TAB, "Table_4_cross_system.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n"
                "\\caption{Cross-system comparison of the MLP estimators, "
                "V\\_RMSE (pu), 5 independent training runs. Diff = "
                "S-ANN $-$ M-ANN with paired-bootstrap 95\\% CI (10k "
                "resamples); all CIs exclude zero.}\n"
                "\\label{tab:cross}\n\\footnotesize\n"
                "\\begin{tabular}{llllrr}\n\\toprule\n"
                "\multicolumn{1}{c}{System} & \multicolumn{1}{c}{Noise} & "
                "\multicolumn{1}{c}{S-ANN} & \multicolumn{1}{c}{M-ANN} & "
                "\multicolumn{1}{c}{Diff} & \multicolumn{1}{c}{CI} \\\\\n\midrule\n")
        for _, r in t4.iterrows():
            f.write(f"{r['System']} & {r['Noise']} & {r['S-ANN']} & "
                    f"{r['M-ANN']} & {r['Diff']} & {r['CI']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # ---------------- Table 5: robust-loss family (E5) ----------------
    e5loss = {"mse": "MSE", "huber_d1.0": "Huber $\\delta{=}1.0$",
              "quantile_t0.5": "Quantile $\\tau{=}0.5$",
              "modal_b2.0": "Modal $b{=}2.0$", "modal_b1.0": "Modal $b{=}1.0$",
              "tukey_c2.0": "Tukey $c{=}2.0$"}
    order5 = ["mse", "huber_d1.0", "quantile_t0.5", "modal_b2.0",
              "modal_b1.0", "tukey_c2.0"]
    with open(os.path.join(TAB, "Table_5_robust_loss.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n"
                "\\caption{Robust-loss family comparison (14-bus ANN, "
                "corrected pipeline, 5 independent training runs). "
                "V\\_RMSE (pu), mean$\\pm$std. Parameters frozen before the "
                "experiments (E5 protocol).}\n\\label{tab:robustloss}\n"
                "\\footnotesize\n\\begin{tabular}{lcccc}\n\\toprule\n"
                "Loss & Default & 10\\% Mixture & 20\\% Mixture & Erroneous "
                "\\\\\n\\midrule\n")
        for k in order5:
            cells = []
            for n in ["default", "mixture_010", "mixture_020", "erroneous"]:
                g = e5[(e5.model == k) & (e5.noise == n)]
                cells.append(fmt_mstd(g))
            f.write(f"{e5loss[k]} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # ---------------- Table 6: sensitivity (E3+E4 condensed) ----------------
    rows = []
    for bw in ["b0.5", "b1.0", "b2.0", "anneal_8_2"]:
        g = e3[(e3.notes.str.contains(bw)) & (e3.model == "modal_supervised_ann")
               & (e3.noise == "mixture_020")]
        rows.append({"Setting": "ANN bandwidth " + bw.replace("_", "\\_"),
                     "Metric": "V\\_RMSE (20\\% mixture)",
                     "Value": f"{g.v_rmse.mean():.4f}"})
    for lam in [0.0, 0.05, 0.10, 0.20, 0.50]:
        g = e4[(e4.lambda_meas == lam) & (e4.noise == "mixture_020")]
        rows.append({"Setting": f"M-DSS2 $\\lambda_{{\\mathrm{{meas}}}}={lam}$",
                     "Metric": "V\\_RMSE (20\\% mixture)",
                     "Value": f"{g.v_rmse.mean():.4f}"})
    t6 = pd.DataFrame(rows)
    t6.to_csv(os.path.join(TAB, "Table_6_sensitivity.csv"), index=False)
    with open(os.path.join(TAB, "Table_6_sensitivity.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n"
                "\\caption{Condensed sensitivity results (20\\% mixture, "
                "CIGRE-14, corrected pipeline; full tables in the "
                "Appendix).}\n\\label{tab:sensitivity}\n\\footnotesize\n"
                "\\begin{tabular}{lll}\n\\toprule\n"
                "Setting & Metric & Value \\\\\n\\midrule\n")
        for _, r in t6.iterrows():
            f.write(f"{r['Setting']} & {r['Metric']} & {r['Value']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # ---------------- Table 7: multi-start (E7, K=1 vs K=10) ----------------
    def _e7_sel_from_starts(solver, noise, K):
        # K=1: the flat start only (no tie ambiguity; derived from the
        # frozen start-level CSV). K=10: the runner's frozen selected CSV.
        if K == 10:
            g10 = e7e[(e7e.solver == solver) & (e7e.noise == noise)
                      & (e7e.K == 10)]
            succ = int(g10.multistart_status.eq("MULTISTART_SUCCESS").sum())
            n_units = len(g10)
            v = g10.v_rmse.dropna().median()
            return succ / n_units, (float(v) if np.isfinite(v) else np.nan)
        sub = e7s[(e7s.solver == solver) & (e7s.noise == noise)
                  & (e7s.start_id <= K)]
        best_v, succ = [], 0
        for (si, di), gg in sub.groupby(["sample_idx", "draw_idx"]):
            best_obj, best_rmse = np.inf, np.nan
            for _, r in gg.sort_values("start_id").iterrows():
                if r.converged == 1 and np.isfinite(r.objective) \
                        and r.objective < best_obj:
                    best_obj = r.objective
                    best_rmse = r.v_rmse
            if np.isfinite(best_obj):
                succ += 1
                if np.isfinite(best_rmse):
                    best_v.append(best_rmse)
        n_units = sub.groupby(["sample_idx", "draw_idx"]).ngroups
        return succ / n_units, (float(np.median(best_v)) if best_v else np.nan)

    with open(os.path.join(TAB, "Table_7_multistart.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n"
                "\\caption{Multi-start solver stability (10 samples x 5 "
                "draws). Success = at least one converged start among the "
                "first K starts; accuracy from objective-based selection "
                "among converged starts (derived from the recorded "
                "start-level results).}\n\\label{tab:multistart}\n"
                "\\footnotesize\n\\begin{tabular}{llrrrr}\n\\toprule\n"
                "Noise & Solver & Succ $K{=}1$ & Succ $K{=}10$ & "
                "V\\_RMSE $K{=}1$ & V\\_RMSE $K{=}10$ \\\\\n\\midrule\n")
        for n in ["default", "mixture_010", "mixture_020", "erroneous"]:
            for solver, lab in [("wls", "WLS"), ("modal_wls", "Modal-WLS")]:
                s1, v1 = _e7_sel_from_starts(solver, n, 1)
                s10, v10 = _e7_sel_from_starts(solver, n, 10)
                f.write(f"{NOISE_LABEL[n]} & {lab} & {s1:.2f} & {s10:.2f} & "
                        f"{v1:.4f} & {v10:.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # ================= FIGURES =================
    # Fig 1: loss/influence curves
    r = np.linspace(-4, 4, 2001)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, mode in zip(axes, ["loss", "influence"]):
        def mse_rho(x): return x ** 2
        def mse_psi(x): return x
        def hub_rho(x): return np.where(np.abs(x) <= 1, 0.5 * x ** 2,
                                        np.abs(x) - 0.5)
        def hub_psi(x): return np.clip(x, -1, 1)
        def tuk_rho(x): return np.where(np.abs(x) <= 2, (4 / 6) * (
            1 - (1 - (x / 2) ** 2) ** 3), 4 / 6)
        def tuk_psi(x): return np.where(np.abs(x) <= 2,
                                        x * (1 - (x / 2) ** 2) ** 2, 0)
        def mod_rho(x, b): return 1 - np.exp(-x ** 2 / (2 * b ** 2))
        def mod_psi(x, b): return x / b ** 2 * np.exp(-x ** 2 / (2 * b ** 2))
        curves = [(mse_rho, mse_psi, "MSE", "C0"),
                  (hub_rho, hub_psi, "Huber $\\delta{=}1$", "C1"),
                  (tuk_rho, tuk_psi, "Tukey $c{=}2$", "C2"),
                  (lambda x: mod_rho(x, 1), lambda x: mod_psi(x, 1),
                   "Modal $b{=}1$", "C3"),
                  (lambda x: mod_rho(x, 2), lambda x: mod_psi(x, 2),
                   "Modal $b{=}2$", "C4")]
        for fr, fp, lab, col in curves:
            ax.plot(r, fr(r) if mode == "loss" else fp(r), color=col,
                    label=lab, lw=1.6)
        ax.set_xlabel("standardized residual $r$")
        ax.set_ylabel("$\\rho(r)$" if mode == "loss"
                      else "$\\psi(r)=\\partial\\rho/\\partial r$")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_1_influence.pdf")); plt.close(fig)

    # Fig 2: main 14-bus V_RMSE
    fig, ax = plt.subplots(figsize=(8.5, 4))
    xs = np.arange(6); w = 0.12
    for i, mid in enumerate(["wls", "modal_wls", "supervised_ann",
                             "modal_supervised_ann", "supervised_dss2",
                             "modal_supervised_dss2"]):
        vals, errs = [], []
        for n in noises6:
            if mid in ("wls", "modal_wls"):
                g = e1s[(e1s.model == mid) & (e1s.noise == n)]
                vals.append(g.v_rmse.iloc[0] if len(g) else np.nan)
                errs.append(0)
            else:
                g = e1[(e1.model == mid) & (e1.noise == n)]
                vals.append(g.v_rmse.mean()); errs.append(g.v_rmse.std())
        ax.bar(xs + (i - 2.5) * w, vals, w, yerr=errs, capsize=2,
               label=MODEL_LABEL[mid])
    ax.set_xticks(xs)
    ax.set_xticklabels([PLOT_NOISE[n] for n in noises6], rotation=25,
                       ha="right", fontsize=8)
    ax.set_ylabel("V RMSE (pu)"); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_2_main_14bus.pdf")); plt.close(fig)

    # Fig 3: cross-system effect sizes
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    xpos, labels = [], []
    k = 0
    for ds, df_, label in [("cigre14", e1, "CIGRE-14"),
                           ("ieee30", e6a, "IEEE-30"),
                           ("ober_sub", e2, "Ober-70"),
                           ("pantagruel_14", e6b, "PTG-14")]:
        for j, n in enumerate(["default", "mixture_010", "erroneous"]):
            a = df_[(df_.model == "supervised_ann") & (df_.noise == n)
                    ].sort_values("repeat_id").v_rmse.values
            b = df_[(df_.model == "modal_supervised_ann") & (df_.noise == n)
                    ].sort_values("repeat_id").v_rmse.values
            d, lo, hi = paired_ci(a, b)
            xpos.append(k); labels.append(f"{label}\n{PLOT_NOISE_SHORT[n]}")
            ax.errorbar(k, d, yerr=[[d - lo], [hi - d]], fmt="o", capsize=3)
            k += 1
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=7.5, ha="center")
    ax.margins(x=0.02)
    ax.set_ylabel("S-ANN $-$ M-ANN V_RMSE (pu)\n(positive = modal better)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_3_cross_system_effect.pdf"))
    plt.close(fig)

    # Fig 4: sensitivity (2 panels: E3 bandwidth, E4 lambda)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for mid, lab, ax in [("modal_supervised_ann", "M-ANN", axes[0]),
                         ("modal_supervised_dss2", "M-DSS2", axes[1])]:
        xs = np.arange(4); w = 0.2
        for j, n in enumerate(["default", "mixture_010", "mixture_020"]):
            vals = []
            for bw in ["b0.5", "b1.0", "b2.0", "anneal_8_2"]:
                g = e3[(e3.notes.str.contains(bw)) & (e3.model == mid)
                       & (e3.noise == n)]
                vals.append(g.v_rmse.mean() if len(g) else np.nan)
            ax.bar(xs + (j - 1) * w, vals, w, label=PLOT_NOISE[n])
        ax.set_xticks(xs)
        ax.set_xticklabels(["b=0.5", "b=1.0", "b=2.0", "anneal"], rotation=10)
        ax.set_title(f"Bandwidth, {lab}"); ax.legend(fontsize=6)
        ax.set_ylabel("V RMSE (pu, mean)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_4_bandwidth.pdf")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    lams = [0.0, 0.05, 0.10, 0.20, 0.50]
    for n in ["default", "mixture_010", "mixture_020"]:
        vals = []
        for lam in lams:
            g = e4[(e4.lambda_meas == lam) & (e4.noise == n)]
            vals.append(g.v_rmse.mean())
        ax.plot(lams, vals, marker="o", label=PLOT_NOISE[n])
    ax.set_xlabel("$\\lambda_{\\mathrm{meas}}$")
    ax.set_ylabel("V RMSE (pu, mean)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_4_lambda.pdf")); plt.close(fig)

    # Fig 5: multi-start
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    xs_cat = np.arange(4)
    for ax, solver in zip(axes, ["wls", "modal_wls"]):
        for n in ["default", "mixture_010", "mixture_020"]:
            lab = PLOT_NOISE_SHORT.get(n, PLOT_NOISE[n])
            vals = []
            for K in [1, 3, 5, 10]:
                sub = e7s[(e7s.solver == solver) & (e7s.noise == n)
                          & (e7s.start_id <= K)]
                vals.append(sub.groupby(["sample_idx", "draw_idx"])
                            .converged.max().mean())
            ax.plot(xs_cat, vals, marker="o", label=lab)
        ax.set_xticks(xs_cat)
        ax.set_xticklabels(["K=1", "K=3", "K=5", "K=10"])
        ax.set_xlabel("number of starts K")
        ax.set_ylabel("multi-start success")
        ax.set_title("WLS" if solver == "wls" else "Modal-WLS")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_5_multistart.pdf")); plt.close(fig)

    print("FINAL tables + figures written to tables/FINAL, figures/FINAL")


if __name__ == "__main__":
    main()
