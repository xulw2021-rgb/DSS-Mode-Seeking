# Experiment map (internal IDs → publication-facing names)

Internal identifiers were used during development for provenance only; the manuscript and this README use scientific names.

| Internal ID | Publication-facing experiment | Frozen results |
|---|---|---|
| E1 | CIGRE-14 main comparison | results/final/E1_learning_corrected.csv (+ E1_solvers.csv) |
| E2 | Oberrhein-70 architecture-matched comparison | results/final/E2_corrected.csv |
| E3 | Bandwidth sensitivity | results/final/E3_corrected.csv |
| E4 | Measurement-consistency weight sensitivity | results/final/E4_corrected.csv |
| E5 | Robust-loss family comparison | results/final/E5_corrected.csv |
| E6 | IEEE-30 and PanTaGruEl external benchmark | results/final/E6A_ieee30_corrected.csv, E6B_pantagruel_corrected.csv |
| E7 | WLS / Modal-WLS multi-start stability | results/final/e7_start_level.csv, e7_selected_solution.csv |

Notes: E1's optimization-based solver rows (WLS, Modal-WLS) operate on raw measurement vectors and are unaffected by the train-only preprocessing correction; all neural results use preprocessing_version='train_only'. There is no E8 in the paper (an optional diagnostic was judged unnecessary during review).
