# Datasets

## Classification

| Dataset | Classification | Source |
|---|---|---|
| CIGRE-14 (15 buses) | project-generated simulation benchmark | pandapower Monte-Carlo pipeline (`src/toy_network.py`) |
| IEEE-30 | standard benchmark, project-generated scenarios | `scripts/generate_ieee30_data.py` |
| Oberrhein-70 | project-generated simulation benchmark | pandapower pipeline (`src/toy_network.py`, mv_oberrhein) |
| PanTaGruEl IEEE-14 | independently generated external benchmark (Level 3; NOT field data) | Zenodo 10.5281/zenodo.5816149, CC-BY-4.0 |

## PanTaGruEl (external download required)

The PanTaGruEl dataset is **not bundled** with this repository.

1. Download: `python scripts/download_pantagruel.py --out external/pantagruel`
   (or manually from https://zenodo.org/records/5816149 — the archive
   `data_sets.tar.xz` contains `dataset_14_case_1.h5`).
2. Convert (bus reindexing 1→0, per-unit to MW convention, shunt splitting):
   `python scripts/convert_pantagruel.py --src external/pantagruel/dataset_14_case_1.h5 --out-dir data/pantagruel_14_case1`
3. The converter writes the project's pickle format (`nodes`, `edges`, `labels`,
   `bus_param`, `edge_param`, `noise_param`, plus `dataset_meta.json` documenting
   the unit conventions).

Only IEEE-14 case 1 is needed for the paper's external-benchmark experiment.

## Generated data

- IEEE-30: `python scripts/generate_ieee30_data.py --out-dir data/ieee30` (2160 scenarios, ext-grid slack handling, Monte-Carlo load sampling; see the script header for the documented adaptations).
- CIGRE-14 / Oberrhein-70: `src/toy_network.py` implements the original pandapower Monte-Carlo scenario pipeline (run from the repository root with the grids' default parameters; results land under `data/<grid>/`).

## Train/test split and preprocessing

The split is seeded per experiment (`data_seed`), 90/10 (70-bus: capped at 500 train / 100 test). All normalization statistics for the learning experiments are fitted on the training partition only (`train_only` preprocessing), matching the corrected manuscript protocol.
