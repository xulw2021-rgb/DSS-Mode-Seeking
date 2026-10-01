# Troubleshooting

- **`ModuleNotFoundError: torch_geometric`**: install from the PyTorch index or via `pip install torch_geometric` (2.6.1 works with torch 2.1.2 without the old torch_scatter/torch_sparse wheels).
- **`KeyError: 'PM_NOISE'`**: the data pickles use the uppercase noise-parameter column; regenerate data with the bundled pipeline (older pickles may predate this fix).
- **OOM when generating corrupted test copies**: the framework uses the memory-safe `corrupt_graph_data_lite`; if you copy the corruption code elsewhere, avoid deepcopying graph objects built from shared storages.
- **Modal-WLS slow under 20% mixture**: expected (documented numerical difficulty); reduce `n_draws` for a quick smoke.
- **Figures/tables differ from the paper**: run `verify_publication_artifacts.py`; any mismatch means the frozen CSVs were modified — restore them from the release tag.
- **No GPU**: use `--device cpu` for smoke runs; full neural training is GPU-recommended.
