# Reproducibility

## Validated environment

- OS: Linux (kernel 5.15, glibc 2.31)
- Python 3.10.0
- PyTorch 2.1.2 (CUDA 12.1 builds)
- torch_geometric 2.6.1
- pandapower 2.14.10
- numpy 1.26.4, scipy 1.13.1, pandas 2.2.3, matplotlib 3.10.1, seaborn 0.13.2
- Hardware used for training: 2× NVIDIA RTX 4090 (24 GB)

Minimum requirements (CPU-only, tables/figures/tests): Python 3.10, numpy, scipy, pandas, matplotlib, torch (CPU build), torch_geometric, pandapower. Full neural training: NVIDIA GPU with ≥ 8 GB memory recommended.

## Reproducibility modes

- **Mode A (recommended, minutes):** regenerate all manuscript tables and figures from the frozen results:
  `python scripts/reproduce_publication_artifacts.py` then `python scripts/verify_publication_artifacts.py`.
- **Mode B (training, hours):** rerun the experiments with the commands below.

## Seed protocol (all experiments)

- `training_seed = 100 + repeat_id` (model init, optimizer, dataloader shuffling; repeat_id 0–4)
- `data_seed` per experiment: `500 + md5(experiment_id) % 1000` (dataset noise draws + train/test split)
- `eval_noise_seed = 700 + noise_index` (test-corruption draws; shared across models → paired comparisons)
- The paper reports five independent training runs per cell; 10,000 bootstrap resamples do not increase this replication.
- GPU training is statistically reproducible under this protocol but not bitwise deterministic across GPU architectures.

## Experiment commands (run from repository root; add `--device cuda:0` / `--device cpu`)

| Experiment | Command | Approx. cost |
|---|---|---|
| CIGRE-14 main (learning) | `python scripts/run_e1.py --chunk 0` (or without chunk on one GPU; solvers are CPU) | ~2 h GPU |
| Oberrhein-70 | `python scripts/run_e2.py --chunk ann` + `--chunk dss2` (parallel GPUs) | ~0.2 h |
| Bandwidth sensitivity | `python scripts/run_e3.py --chunk ann` / `--chunk dss2` | ~3 h |
| Measurement-weight sensitivity | `python scripts/run_e4.py --chunk 0` / `--chunk 1` | ~5 h (2 GPUs) |
| Robust-loss family | `python scripts/run_e5.py --run` | ~0.7 h |
| IEEE-30 + PanTaGruEl | `python scripts/run_e6.py --chunk ieee30` / `--chunk pantagruel` | ~0.5 h |
| WLS / Modal-WLS multi-start | `python scripts/run_e7.py --chunk wls` / `--chunk modal` | WLS ~0.2 h; Modal-WLS ~6 h CPU |

Each runner has `--preflight`, `--smoke`, and `--finalize` modes; finalize merges chunk CSVs and writes summaries/manifests. Frozen hyperparameters: bandwidth b=1.0 (neural modal losses), Modal-WLS solver bandwidth b=2.0, λ_meas=0 (main protocol), λ_phys=1.0 (M-DSS2), Adamax η=3e-3, 200 epochs (100 for 70-bus), batch 64 (ANN) / 32–16 (DSS2).

## Known caveats

- The WLS / Modal-WLS multi-start sweep over severe contamination is CPU-bound and slow (documented in the paper).
- PanTaGruEl must be downloaded separately (see docs/DATASETS.md).
- The toy_network.py pipeline assumes pandapower's CIGRE/Oberrhein network definitions; IEEE-30 uses the dedicated generator with documented adaptations (ext-grid slack, line-only edge parameters).
