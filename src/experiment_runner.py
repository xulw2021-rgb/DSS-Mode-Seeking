"""
Experiment runner for the modal-loss DSSE study.

Orchestrates training and evaluation across:
  - 7 models:  WLS, Modal-WLS, Supervised-ANN, Modal-Supervised-ANN,
               Supervised-DSS2, Modal-Supervised-DSS2, Weakly-Supervised-DSS2
  - 8+ noise settings: default Gaussian, high Gaussian, Student-t, Laplace,
               mixture outliers, missing, erroneous, missing+erroneous
  - Repeated runs with mean / std collection

Output: CSV tables under results_modal/tables/, figures under
        results_modal/figures/, summary under results_modal/summary.md

Usage:
    python experiment_runner.py          # full run (CIGRE 14)
    python experiment_runner.py --smoke  # quick smoke test
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import time
import os
import sys
import json
from copy import deepcopy

# Work around transformers/numpy metadata bug
import importlib.metadata as _im
_orig_v = _im.version
_im.version = lambda pkg: np.__version__ if pkg == 'numpy' else _orig_v(pkg)

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from networks import GAT_DSSE, ANN_DSSE, SkipPFN
from data import data_from_pickles, get_pflow, gsp_wls_edge
from modal_losses import (
    state_space_modal_loss, modal_score, hit_rate,
    gaussian_kernel_modal_loss,
)
from modal_wls import (
    solve_wls, solve_modal_wls, build_measurement_structure,
    load_sample, get_pflow_np, compute_injections,
    measurement_function,
)

# ======================================================================
# Configuration
# ======================================================================

RESULTS_DIR = "results_modal"
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

DEFAULT_NOISE_SETTINGS = [
    "default",
    "high_gaussian",
    "student_t3",
    "laplace",
    "mixture_005",
    "mixture_010",
    "mixture_020",
    "missing",
    "erroneous",
    "missing_erroneous",
]

SMOKE_NOISE_SETTINGS = ["default", "mixture_010", "missing"]  # representative subset
SMOKE_MODELS = [
    "wls", "modal_wls",
    "supervised_ann", "modal_supervised_ann",
    "supervised_dss2", "modal_supervised_dss2",
]

# ======================================================================
# Noise injection
# ======================================================================


def inject_noise_to_measurements(z, sigma, noise_type, rng=None):
    """
    Apply noise corruption to a measurement vector.

    Args:
        z:          Clean measurement vector (numpy array).
        sigma:      Measurement standard deviations (numpy array).
        noise_type: One of the noise setting keys.
        rng:        numpy RandomState (optional).

    Returns:
        z_corrupted, sigma_corrupted
    """
    if rng is None:
        rng = np.random.RandomState()

    z = np.asarray(z, dtype=np.float64).copy()
    sigma = np.asarray(sigma, dtype=np.float64).copy()
    n = len(z)

    if noise_type == "default":
        z += rng.randn(n) * sigma

    elif noise_type == "high_gaussian":
        z += rng.randn(n) * sigma * 3.0

    elif noise_type == "student_t3":
        z += rng.standard_t(df=3, size=n) * sigma

    elif noise_type == "laplace":
        z += rng.laplace(loc=0.0, scale=sigma / np.sqrt(2.0), size=n)

    elif noise_type.startswith("mixture"):
        pi = float(noise_type.split("_")[1]) / 100.0  # e.g. "mixture_005" -> 0.05
        is_outlier = rng.rand(n) < pi
        z += rng.randn(n) * sigma  # base Gaussian noise
        outlier_scale = 20.0 * np.abs(z[is_outlier]) + 5.0
        z[is_outlier] += rng.randn(is_outlier.sum()) * outlier_scale

    elif noise_type == "missing":
        # Randomly drop 20% of measurements (set to 0 → masked out)
        drop_mask = rng.rand(n) < 0.20
        z[drop_mask] = 0.0
        sigma[drop_mask] = 1e10  # effectively infinite uncertainty

    elif noise_type == "erroneous":
        # Flip sign on 10% of measurements
        flip_mask = rng.rand(n) < 0.10
        z[flip_mask] *= -1.0

    elif noise_type == "missing_erroneous":
        drop_mask = rng.rand(n) < 0.10
        z[drop_mask] = 0.0
        sigma[drop_mask] = 1e10
        flip_mask = rng.rand(n) < 0.10
        flip_mask[drop_mask] = False  # don't flip dropped measurements
        z[flip_mask] *= -1.0

    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return z, sigma


def corrupt_graph_data(data, noise_type, x_mean, x_std, edge_mean, edge_std, rng):
    """
    Apply noise corruption to a PyG Data object's measurements.

    The data.x and data.edge_attr contain normalized measurements.
    We un-normalize, inject noise, and re-normalize.
    """
    num_nfeat = 8
    num_efeat = 6

    data = deepcopy(data)

    # --- Node measurements ---
    x_in = data.x[:, :num_nfeat].numpy()
    # Un-normalize
    z_nodes = x_in[:, ::2] * x_std[::2].numpy() + x_mean[::2].numpy()
    sigma_nodes = 1.0 / np.sqrt(
        np.abs(x_in[:, 1::2] * x_std[1::2].numpy() + x_mean[1::2].numpy()) + 1e-10
    )

    # Corrupt
    z_corr = z_nodes.copy()
    sigma_corr = sigma_nodes.copy()
    for col in range(4):
        mask = z_nodes[:, col] != 0.0
        if mask.sum() == 0:
            continue
        z_corr[mask, col], sigma_corr[mask, col] = inject_noise_to_measurements(
            z_nodes[mask, col], sigma_nodes[mask, col], noise_type, rng
        )

    # Re-normalize
    x_corr = np.zeros_like(x_in)
    for col in range(4):
        x_corr[:, 2 * col] = (z_corr[:, col] - x_mean[2 * col].item()) / max(
            x_std[2 * col].item(), 1e-10
        )
        x_corr[:, 2 * col + 1] = x_in[:, 2 * col + 1]  # keep original cov

    data.x[:, :num_nfeat] = torch.tensor(x_corr, dtype=torch.float32)

    # --- Edge measurements ---
    e_in = data.edge_attr[:, :num_efeat].numpy()
    z_edges = e_in[:, :4:2] * edge_std[:4:2].numpy() + edge_mean[:4:2].numpy()
    sigma_edges = 1.0 / np.sqrt(
        np.abs(e_in[:, 1:4:2] * edge_std[1:4:2].numpy() + edge_mean[1:4:2].numpy()) + 1e-10
    )

    z_ec = z_edges.copy()
    sigma_ec = sigma_edges.copy()
    for col in range(2):
        mask = z_edges[:, col] != 0.0
        if mask.sum() == 0:
            continue
        z_ec[mask, col], sigma_ec[mask, col] = inject_noise_to_measurements(
            z_edges[mask, col], sigma_edges[mask, col], noise_type, rng
        )

    e_corr = np.zeros_like(e_in)
    for col in range(2):
        e_corr[:, 2 * col] = (z_ec[:, col] - edge_mean[2 * col].item()) / max(
            edge_std[2 * col].item(), 1e-10
        )
        e_corr[:, 2 * col + 1] = e_in[:, 2 * col + 1]

    data.edge_attr[:, :num_efeat] = torch.tensor(e_corr, dtype=torch.float32)

    return data


# ======================================================================
# Missing baseline models: training functions
# ======================================================================


def train_supervised_ann(folder, device, epochs=600, lr=3e-3, dim_hid=64,
                         num_layers=3, verbose=False):
    """Train a standard Supervised ANN (MSE loss)."""
    num_nfeat = 8; num_efeat = 6; num_nmeas = 4; num_emeas = 2
    meas_v = np.array([0, 1, 12, 7, 11, 14])
    meas_pf = np.array([0, 10])

    dataset, x_mean, x_std, pflow_mean, pflow_std = data_from_pickles(
        folder, num_nfeat, num_efeat, num_nmeas, num_emeas, meas_v, meas_pf
    )
    random.shuffle(dataset)
    n_train = int(0.9 * len(dataset))
    train_ds = dataset[:n_train]

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    model = ANN_DSSE(dim_in=num_nfeat, dim_hid=dim_hid, dim_out=2,
                     num_layers=num_layers).to(device)
    opt = torch.optim.Adamax(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            x_in = data.x[:, :num_nfeat]
            y = data.y
            opt.zero_grad()
            out = model(x_in)
            loss = F.mse_loss(out, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if verbose and epoch % 100 == 0:
            print(f"  ANN epoch {epoch}: loss={total_loss/len(train_loader):.6f}")

    return model, x_mean, x_std, pflow_mean, pflow_std


def train_supervised_dss2(folder, device, epochs=600, lr=3e-3, lam_sup=1.0,
                          verbose=False):
    """Train a Supervised DSS2 (GNN with WLS + supervised MSE).

    NOTE: Uses CPU for gsp_wls_edge compatibility (internal tensor creation).
    """
    num_nfeat = 8; num_efeat = 6; num_nmeas = 4; num_emeas = 2
    meas_v = np.array([0, 1, 12, 7, 11, 14])
    meas_pf = np.array([0, 10])

    dataset, x_mean, x_std, pflow_mean, pflow_std = data_from_pickles(
        folder, num_nfeat, num_efeat, num_nmeas, num_emeas, meas_v, meas_pf
    )
    random.shuffle(dataset)
    n_train = int(0.9 * len(dataset))
    train_ds = dataset[:n_train]

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = GAT_DSSE(dim_feat=num_nfeat, dim_dense=32, dim_out=2, heads=1,
                     num_layers=8, edge_dim=num_efeat)
    opt = torch.optim.Adamax(model.parameters(), lr=lr)

    reg_coefs = {'lam_v': 1e-4, 'lam_p': 1e-8, 'lam_pf': 1e-6, 'lam_reg': 1e2}

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for data in train_loader:
            opt.zero_grad()
            out = model(data.x[:, :num_nfeat], data.edge_index,
                        data.edge_attr[:, :num_efeat])
            loss_wls = gsp_wls_edge(
                input=data.x[:, :num_nfeat],
                edge_input=data.edge_attr[:, :num_efeat],
                output=out, x_mean=x_mean, x_std=x_std,
                edge_mean=pflow_mean, edge_std=pflow_std,
                edge_index=data.edge_index, reg_coefs=reg_coefs,
                num_samples=data.batch[-1] + 1,
                node_param=data.x[:, num_nfeat:],
                edge_param=data.edge_attr[:, num_efeat:],
            )
            loss_sup = F.mse_loss(out, data.y)
            loss = loss_wls + lam_sup * loss_sup
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if verbose and epoch % 100 == 0:
            print(f"  SupDSS2 epoch {epoch}: loss={total_loss/len(train_loader):.6f}")

    return model, x_mean, x_std, pflow_mean, pflow_std


def train_weakly_supervised_dss2(folder, device, epochs=600, lr=3e-3,
                                 verbose=False):
    """Train Weakly-Supervised DSS2 (original gsp_wls_edge only).

    NOTE: Uses CPU for gsp_wls_edge compatibility.
    """
    num_nfeat = 8; num_efeat = 6; num_nmeas = 4; num_emeas = 2
    meas_v = np.array([0, 1, 12, 7, 11, 14])
    meas_pf = np.array([0, 10])

    dataset, x_mean, x_std, pflow_mean, pflow_std = data_from_pickles(
        folder, num_nfeat, num_efeat, num_nmeas, num_emeas, meas_v, meas_pf
    )
    random.shuffle(dataset)
    n_train = int(0.9 * len(dataset))
    train_ds = dataset[:n_train]

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = GAT_DSSE(dim_feat=num_nfeat, dim_dense=32, dim_out=2, heads=1,
                     num_layers=8, edge_dim=num_efeat)
    opt = torch.optim.Adamax(model.parameters(), lr=lr)

    reg_coefs = {'lam_v': 1e-4, 'lam_p': 1e-8, 'lam_pf': 1e-6, 'lam_reg': 1e2}

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for data in train_loader:
            opt.zero_grad()
            out = model(data.x[:, :num_nfeat], data.edge_index,
                        data.edge_attr[:, :num_efeat])
            loss = gsp_wls_edge(
                input=data.x[:, :num_nfeat],
                edge_input=data.edge_attr[:, :num_efeat],
                output=out, x_mean=x_mean, x_std=x_std,
                edge_mean=pflow_mean, edge_std=pflow_std,
                edge_index=data.edge_index, reg_coefs=reg_coefs,
                num_samples=data.batch[-1] + 1,
                node_param=data.x[:, num_nfeat:],
                edge_param=data.edge_attr[:, num_efeat:],
            )
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if verbose and epoch % 100 == 0:
            print(f"  WeakDSS2 epoch {epoch}: loss={total_loss/len(train_loader):.6f}")

    return model, x_mean, x_std, pflow_mean, pflow_std


# ======================================================================
# Evaluation
# ======================================================================


def evaluate_wls_model(sample_loader, model_fn, device=None):
    """
    Evaluate a per-sample solver (WLS / Modal-WLS) on a loader of samples.

    model_fn(sample_dict) -> result_dict with V_est, theta_est.

    Returns metrics dict.
    """
    metrics = {
        'v_rmse': [], 'th_rmse': [], 'loading_rmse': [],
        'modal_score': [], 'hit_rate': [], 'runtime': [], 'converged': [],
    }

    for sample in sample_loader:
        t0 = time.time()
        result = model_fn(sample)
        elapsed = time.time() - t0

        true_V = sample['true_V']
        true_theta = sample['true_theta']
        V_est = result['V_est']
        theta_est = result['theta_est']

        metrics['v_rmse'].append(np.sqrt(np.mean((V_est - true_V) ** 2)))
        metrics['th_rmse'].append(np.sqrt(np.mean((theta_est - true_theta) ** 2)))
        metrics['runtime'].append(elapsed)
        metrics['converged'].append(float(result.get('converged', False)))

        # ModalScore
        x_true = np.concatenate([true_V, true_theta])
        x_est = np.concatenate([V_est, theta_est])
        state_scale = np.concatenate([
            np.full(len(true_V), max(np.std(true_V), 0.005)),
            np.full(len(true_theta), max(np.std(true_theta), 0.005)),
        ])
        r = (x_true - x_est) / (state_scale + 1e-8)
        bw = 2.0
        metrics['modal_score'].append(
            float(np.mean(np.exp(-0.5 * (r / bw) ** 2)))
        )
        metrics['hit_rate'].append(
            float(np.mean(np.abs(r) <= 3.0))
        )

    return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}


def evaluate_learning_model(test_loader, model, x_mean, x_std, device):
    """Evaluate a trained learning model on a DataLoader."""
    num_nfeat = 8; num_efeat = 6

    v_rmse = []; th_rmse = []; load_rmse = []; ms = []; hr = []; runtimes = []

    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            x_in = data.x[:, :num_nfeat]
            y_true = data.y

            t0 = time.time()
            out = model(x_in)
            if hasattr(model, 'edge_aggr'):  # GNN model
                out = model(x_in, data.edge_index, data.edge_attr[:, :num_efeat])
            elapsed = time.time() - t0
            runtimes.append(elapsed)

            node_p = data.x[:, num_nfeat:]
            edge_p = data.edge_attr[:, num_efeat:]

            # Un-normalize
            out_un = out.clone()
            out_un[:, 0] = out[:, 0] * x_mean[0] + x_std[0]  # Hmm, this looks wrong.

    # Actually un-normalize correctly
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            x_in = data.x[:, :num_nfeat]
            y_true = data.y
            node_p = data.x[:, num_nfeat:]
            edge_p = data.edge_attr[:, num_efeat:]

            t0 = time.time()
            if isinstance(model, ANN_DSSE):
                out = model(x_in)
            else:
                out = model(x_in, data.edge_index, data.edge_attr[:, :num_efeat])
            elapsed = time.time() - t0
            runtimes.append(elapsed)

            out_un = out.clone()
            out_un[:, 0] = out[:, 0] * x_std[0] + x_mean[0]
            out_un[:, 1] *= (1.0 - node_p[:, 1])

            v_rmse.append(
                torch.sqrt(F.mse_loss(out_un[:, :1], y_true[:, :1])).item()
            )
            th_rmse.append(
                torch.sqrt(F.mse_loss(out_un[:, 1:], y_true[:, 1:])).item()
            )

            # Line loading RMSE
            true_ll, true_lt, _, _, _, _, _, _ = get_pflow(
                y_true, data.edge_index, node_p, edge_p
            )
            out_ll, out_lt, _, _, _, _, _, _ = get_pflow(
                out_un, data.edge_index, node_p, edge_p
            )
            mask = true_ll.nonzero(as_tuple=True)[0]
            if mask.numel() > 0:
                load_rmse.append(
                    torch.sqrt(F.mse_loss(out_ll[mask], true_ll[mask])).item()
                )

            # ModalScore
            state_scale = torch.tensor([
                max(float(y_true[:, 0].std()), 0.005),
                max(float(y_true[:, 1].std()), 0.005),
            ], device=device)
            ms.append(modal_score(y_true, out_un, state_scale, 2.0).item())
            hr.append(hit_rate(y_true, out_un, state_scale, 3.0).item())

    return {
        'v_rmse': (np.mean(v_rmse), np.std(v_rmse)),
        'th_rmse': (np.mean(th_rmse), np.std(th_rmse)),
        'loading_rmse': (np.mean(load_rmse), np.std(load_rmse)) if load_rmse else (np.nan, np.nan),
        'modal_score': (np.mean(ms), np.std(ms)),
        'hit_rate': (np.mean(hr), np.std(hr)),
        'runtime': (np.mean(runtimes), np.std(runtimes)),
        'converged': (1.0, 0.0),  # learning models always "converge"
    }


def evaluate_wls_on_corrupted_sample(meas, node_param, edge_param, edge_index,
                                     true_V, true_theta, noise_type, rng,
                                     wls_config):
    """Run WLS/Modal-WLS on a corrupted sample, return metrics."""
    z_clean = meas['z'].copy()
    sigma_clean = meas['sigma'].copy()

    z_corr, sigma_corr = inject_noise_to_measurements(
        z_clean, sigma_clean, noise_type, rng
    )

    meas_corr = dict(meas)
    meas_corr['z'] = z_corr
    meas_corr['sigma'] = sigma_corr

    t0 = time.time()
    if wls_config.get('modal', False):
        result = solve_modal_wls(
            meas_corr, node_param, edge_param, edge_index,
            bandwidth=wls_config.get('bandwidth', 2.0),
            lam_phys=wls_config.get('lam_phys', 0.0),
            max_iter=30,
            verbose=0,
        )
    else:
        result = solve_wls(
            meas_corr, node_param, edge_param, edge_index,
            lam_phys=wls_config.get('lam_phys', 0.0),
        )
    elapsed = time.time() - t0

    V_est = result['V_est']
    theta_est = result['theta_est']

    v_rmse = np.sqrt(np.mean((V_est - true_V) ** 2))
    th_rmse = np.sqrt(np.mean((theta_est - true_theta) ** 2))

    x_true = np.concatenate([true_V, true_theta])
    x_est = np.concatenate([V_est, theta_est])
    state_scale = np.concatenate([
        np.full(len(true_V), max(np.std(true_V), 0.005)),
        np.full(len(true_theta), max(np.std(true_theta), 0.005)),
    ])
    r = (x_true - x_est) / (state_scale + 1e-8)
    bw_eval = 2.0
    ms_val = float(np.mean(np.exp(-0.5 * (r / bw_eval) ** 2)))
    hr_val = float(np.mean(np.abs(r) <= 3.0))

    return {
        'v_rmse': v_rmse, 'th_rmse': th_rmse,
        'loading_rmse': np.nan,  # not computed for WLS samples
        'modal_score': ms_val, 'hit_rate': hr_val,
        'runtime': elapsed, 'converged': float(result.get('converged', False)),
    }


# ======================================================================
# Main experiment loop
# ======================================================================


def run_experiments(folder="data/cigre14/", noise_settings=None,
                    n_repeats=3, epochs=200, device=None, smoke=False,
                    experiment_name=None):
    """
    Run the full experiment suite.

    Args:
        folder:          Data folder path.
        noise_settings:  List of noise type strings.
        n_repeats:       Number of noise realizations per setting.
        epochs:          Training epochs for learning models.
        device:          Torch device.
        smoke:           If True, use fewer models/settings/epochs.

    Returns:
        results_df: DataFrame with all metrics.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if noise_settings is None:
        noise_settings = SMOKE_NOISE_SETTINGS if smoke else DEFAULT_NOISE_SETTINGS

    if smoke:
        if epochs is None:
            epochs = 30
        if n_repeats is None:
            n_repeats = 3
    else:
        if epochs is None:
            epochs = 200
        if n_repeats is None:
            n_repeats = 3

    # Output directory
    if experiment_name is None:
        experiment_name = ("smoke_14bus_6models_3noise_3repeats" if smoke
                           else "full_experiment")
    exp_dir = os.path.join(RESULTS_DIR, experiment_name)
    exp_tables = os.path.join(exp_dir, "tables")
    exp_figures = os.path.join(exp_dir, "figures")
    os.makedirs(exp_tables, exist_ok=True)
    os.makedirs(exp_figures, exist_ok=True)

    print(f"Device: {device}")
    print(f"Noise settings: {noise_settings}")
    print(f"Repeats: {n_repeats}, Epochs: {epochs}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    meas_v_buses = [0, 1, 12, 7, 11, 14]
    meas_pf = np.array([0, 10])

    # Test samples for WLS
    wls_test_samples = []
    for idx in range(10):  # 10 test samples
        meas, node_param, edge_param, edge_index, labels = load_sample(
            folder, sample_idx=idx, meas_v_buses=meas_v_buses
        )
        wls_test_samples.append({
            'meas': meas, 'node_param': node_param,
            'edge_param': edge_param, 'edge_index': edge_index,
            'true_V': labels['vm_pu'].values.astype(np.float64),
            'true_theta': labels['va_rad'].values.astype(np.float64),
        })

    # Learning-model test data
    num_nfeat = 8; num_efeat = 6; num_nmeas = 4; num_emeas = 2
    full_dataset, x_mean, x_std, pflow_mean, pflow_std = data_from_pickles(
        folder, num_nfeat, num_efeat, num_nmeas, num_emeas,
        meas_v_buses, meas_pf
    )
    random.shuffle(full_dataset)
    n_train = int(0.9 * len(full_dataset))
    train_dataset = full_dataset[:n_train]
    test_dataset = full_dataset[n_train:]

    print(f"Train graphs: {len(train_dataset)}, Test: {len(test_dataset)}")

    # ------------------------------------------------------------------
    # Train learning-based models
    # ------------------------------------------------------------------
    print("\n--- Training models ---")

    # Full set of models — weakly_supervised_dss2 excluded from comparisons
    all_models = [
        "wls", "modal_wls",
        "supervised_ann", "modal_supervised_ann",
        "supervised_dss2", "modal_supervised_dss2",
    ]

    if smoke:
        models_to_run = list(SMOKE_MODELS)
    else:
        models_to_run = list(all_models)

    trained_models = {}

    if "supervised_ann" in models_to_run:
        print("Training Supervised ANN...")
        ann_model, ann_xm, ann_xs, ann_pm, ann_ps = train_supervised_ann(
            folder, device, epochs=epochs
        )
        trained_models['supervised_ann'] = {
            'model': ann_model, 'x_mean': ann_xm, 'x_std': ann_xs,
            'pflow_mean': ann_pm, 'pflow_std': ann_ps, 'type': 'ann',
        }

    if "modal_supervised_ann" in models_to_run:
        print("Training Modal-Supervised ANN...")
        from modal_supervised_ann import train_modal_supervised_ann
        msann_result = train_modal_supervised_ann(
            folder=folder, epochs=epochs, dim_hid=64, num_layers=3,
            bandwidth_x=2.0, eta=0.0, device=device, verbose=False,
        )
        trained_models['modal_supervised_ann'] = {
            'model': msann_result['model'],
            'x_mean': x_mean, 'x_std': x_std,
            'pflow_mean': pflow_mean, 'pflow_std': pflow_std,
            'type': 'ann',
        }

    if "supervised_dss2" in models_to_run:
        print("Training Supervised DSS2...")
        sd2_model, sd2_xm, sd2_xs, sd2_pm, sd2_ps = train_supervised_dss2(
            folder, device, epochs=epochs
        )
        trained_models['supervised_dss2'] = {
            'model': sd2_model, 'x_mean': sd2_xm, 'x_std': sd2_xs,
            'pflow_mean': sd2_pm, 'pflow_std': sd2_ps, 'type': 'gnn',
        }

    if "modal_supervised_dss2" in models_to_run:
        print("Training Modal-Supervised DSS2...")
        from modal_supervised_dss2 import train_modal_supervised_dss2
        msd2_result = train_modal_supervised_dss2(
            folder=folder, model_name="gat", epochs=epochs,
            dim_hid=32, gnn_layers=3,
            b_state_start=8.0, b_state_end=2.0, b_state_tau=float(epochs) / 3,
            b_meas_start=8.0, b_meas_end=2.0, b_meas_tau=float(epochs) / 3,
            eta=0.5, lambda_phys=1.0, device=device, verbose=False,
        )
        trained_models['modal_supervised_dss2'] = {
            'model': msd2_result['model'],
            'x_mean': x_mean, 'x_std': x_std,
            'pflow_mean': pflow_mean, 'pflow_std': pflow_std,
            'type': 'gnn',
        }

    if "weakly_supervised_dss2" in models_to_run:
        print("Training Weakly-Supervised DSS2...")
        wd2_model, wd2_xm, wd2_xs, wd2_pm, wd2_ps = train_weakly_supervised_dss2(
            folder, device, epochs=epochs
        )
        trained_models['weakly_supervised_dss2'] = {
            'model': wd2_model, 'x_mean': wd2_xm, 'x_std': wd2_xs,
            'pflow_mean': wd2_pm, 'pflow_std': wd2_ps, 'type': 'gnn',
        }

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    print("\n--- Evaluating ---")

    all_rows = []
    rng_base = np.random.RandomState(42)

    for noise_type in noise_settings:
        print(f"\nNoise: {noise_type}")

        for model_name in models_to_run:
            v_list = []; th_list = []; ld_list = []; ms_list = []
            hr_list = []; rt_list = []; cv_list = []

            for rep in range(n_repeats):
                seed = 42 + rep * 100 + hash(noise_type) % 10000
                rng = np.random.RandomState(seed)

                if model_name in ("wls", "modal_wls"):
                    # Per-sample WLS evaluation
                    rep_metrics = {'v_rmse': [], 'th_rmse': [], 'runtime': [],
                                   'converged': [], 'modal_score': [], 'hit_rate': []}
                    for sample in wls_test_samples:
                        wls_cfg = {'modal': model_name == 'modal_wls',
                                   'bandwidth': 2.0, 'lam_phys': 0.0}
                        m = evaluate_wls_on_corrupted_sample(
                            sample['meas'], sample['node_param'],
                            sample['edge_param'], sample['edge_index'],
                            sample['true_V'], sample['true_theta'],
                            noise_type, rng, wls_cfg,
                        )
                        for k in rep_metrics:
                            rep_metrics[k].append(m[k])
                    v_val = np.mean(rep_metrics['v_rmse'])
                    th_val = np.mean(rep_metrics['th_rmse'])
                    ld_val = np.nan
                    ms_val = np.mean(rep_metrics['modal_score'])
                    hr_val = np.mean(rep_metrics['hit_rate'])
                    rt_val = np.mean(rep_metrics['runtime'])
                    cv_val = np.mean(rep_metrics['converged'])

                else:
                    # Learning-based model
                    tm = trained_models.get(model_name)
                    if tm is None:
                        continue

                    model = tm['model'].to(device)
                    m_xm = tm['x_mean']; m_xs = tm['x_std']
                    m_pm = tm['pflow_mean']; m_ps = tm['pflow_std']
                    is_gnn = (tm['type'] == 'gnn')

                    # Corrupt test data
                    corrupted_dataset = [
                        corrupt_graph_data(data, noise_type,
                                          m_xm, m_xs, m_pm, m_ps, rng)
                        for data in test_dataset
                    ]
                    test_loader = DataLoader(corrupted_dataset, batch_size=32,
                                            shuffle=False)

                    # Evaluate
                    model.eval()
                    rep_v = []; rep_th = []; rep_ld = []; rep_ms = []
                    rep_hr = []; rep_rt = []
                    state_scale = torch.tensor([
                        max(float(torch.cat([d.y for d in train_dataset], 0)[:, 0].std()), 0.005),
                        max(float(torch.cat([d.y for d in train_dataset], 0)[:, 1].std()), 0.005),
                    ], device=device)

                    with torch.no_grad():
                        for data in test_loader:
                            data = data.to(device)
                            x_in = data.x[:, :num_nfeat]
                            y_true = data.y
                            node_p = data.x[:, num_nfeat:]
                            edge_p = data.edge_attr[:, num_efeat:]

                            t0 = time.time()
                            if is_gnn:
                                out = model(x_in, data.edge_index,
                                           data.edge_attr[:, :num_efeat])
                            else:
                                out = model(x_in)
                            rep_rt.append(time.time() - t0)

                            out_un = out.clone()
                            out_un[:, 0] = out[:, 0] * m_xs[0].to(device) + m_xm[0].to(device)
                            out_un[:, 1] *= (1.0 - node_p[:, 1])

                            rep_v.append(torch.sqrt(
                                F.mse_loss(out_un[:, :1], y_true[:, :1])).item())
                            rep_th.append(torch.sqrt(
                                F.mse_loss(out_un[:, 1:], y_true[:, 1:])).item())

                            true_ll, true_lt, _, _, _, _, _, _ = get_pflow(
                                y_true, data.edge_index, node_p, edge_p)
                            out_ll, out_lt, _, _, _, _, _, _ = get_pflow(
                                out_un, data.edge_index, node_p, edge_p)
                            mask_ll = true_ll.nonzero(as_tuple=True)[0]
                            if mask_ll.numel() > 0:
                                rep_ld.append(torch.sqrt(
                                    F.mse_loss(out_ll[mask_ll], true_ll[mask_ll])).item())

                            rep_ms.append(
                                modal_score(y_true, out_un, state_scale, 2.0).item())
                            rep_hr.append(
                                hit_rate(y_true, out_un, state_scale, 3.0).item())

                    v_val = np.mean(rep_v)
                    th_val = np.mean(rep_th)
                    ld_val = np.mean(rep_ld) if rep_ld else np.nan
                    ms_val = np.mean(rep_ms)
                    hr_val = np.mean(rep_hr)
                    rt_val = np.mean(rep_rt)
                    cv_val = 1.0

                v_list.append(v_val); th_list.append(th_val)
                ld_list.append(ld_val); ms_list.append(ms_val)
                hr_list.append(hr_val); rt_list.append(rt_val)
                cv_list.append(cv_val)

            # Aggregate across repeats
            row = {
                'noise': noise_type,
                'model': model_name,
                'v_rmse_mean': np.mean(v_list),
                'v_rmse_std': np.std(v_list),
                'th_rmse_mean': np.mean(th_list),
                'th_rmse_std': np.std(th_list),
                'loading_rmse_mean': np.mean(ld_list) if not all(np.isnan(ld_list)) else np.nan,
                'loading_rmse_std': np.std(ld_list) if not all(np.isnan(ld_list)) else np.nan,
                'modal_score_mean': np.mean(ms_list),
                'modal_score_std': np.std(ms_list),
                'hit_rate_mean': np.mean(hr_list),
                'hit_rate_std': np.std(hr_list),
                'runtime_mean': np.mean(rt_list),
                'runtime_std': np.std(rt_list),
                'converged_mean': np.mean(cv_list),
                'converged_std': np.std(cv_list),
                'n_repeats': n_repeats,
            }
            all_rows.append(row)
            print(f"  {model_name:30s}: V_RMSE={v_val:.4f}, "
                  f"θ_RMSE={th_val:.4f}, MS={ms_val:.4f}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(all_rows)

    # Summary mean/std CSV
    summary_csv = os.path.join(exp_tables, "summary_mean_std.csv")
    results_df.to_csv(summary_csv, index=False)
    print(f"\nSummary (mean±std) saved to {summary_csv}")

    # Pivot table
    pivot = results_df.pivot_table(
        index='model', columns='noise',
        values=['v_rmse_mean', 'modal_score_mean', 'hit_rate_mean'],
    )
    pivot_path = os.path.join(exp_tables, "pivot.csv")
    pivot.to_csv(pivot_path)
    print(f"Pivot table saved to {pivot_path}")

    # Key comparison: modal vs original, all noise settings
    _save_key_comparison(results_df, exp_tables)

    # Summary markdown
    summary_name = "summary_14bus_full.md" if not smoke else "summary_smoke.md"
    summary_path = os.path.join(exp_dir, summary_name)
    _write_summary(results_df, summary_path, smoke, models_to_run)
    print(f"Summary saved to {summary_path}")

    # Save metadata
    meta = {
        'experiment_name': experiment_name,
        'models': models_to_run,
        'noise_settings': noise_settings if noise_settings else DEFAULT_NOISE_SETTINGS,
        'n_repeats': n_repeats,
        'epochs': epochs,
        'smoke': smoke,
        'device': str(device),
        'data_folder': folder,
    }
    meta_path = os.path.join(exp_dir, "metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    return results_df


def _save_key_comparison(df, exp_tables):
    """Save a CSV comparing each modal variant to its original baseline."""
    comparisons = [
        ("wls", "modal_wls", "WLS", "Modal-WLS"),
        ("supervised_ann", "modal_supervised_ann",
         "Supervised-ANN", "Modal-Supervised-ANN"),
        ("supervised_dss2", "modal_supervised_dss2",
         "Supervised-DSS2", "Modal-Supervised-DSS2"),
    ]

    rows = []
    for orig, modal, orig_label, modal_label in comparisons:
        for noise in sorted(df['noise'].unique()):
            r_orig = df[(df['model'] == orig) & (df['noise'] == noise)]
            r_modal = df[(df['model'] == modal) & (df['noise'] == noise)]
            if r_orig.empty or r_modal.empty:
                continue

            o = r_orig.iloc[0]
            m = r_modal.iloc[0]

            row = {
                'noise': noise,
                'original_model': orig_label,
                'modal_model': modal_label,
                'orig_v_rmse': o['v_rmse_mean'],
                'modal_v_rmse': m['v_rmse_mean'],
                'v_rmse_change_pct': (
                    (m['v_rmse_mean'] - o['v_rmse_mean']) / max(abs(o['v_rmse_mean']), 1e-10) * 100
                ),
                'orig_modal_score': o['modal_score_mean'],
                'modal_modal_score': m['modal_score_mean'],
                'ms_change': m['modal_score_mean'] - o['modal_score_mean'],
                'orig_converged': o['converged_mean'],
                'modal_converged': m['converged_mean'],
                'orig_runtime': o['runtime_mean'],
                'modal_runtime': m['runtime_mean'],
            }
            rows.append(row)

    if rows:
        comp_df = pd.DataFrame(rows)
        comp_path = os.path.join(exp_tables, "key_comparison_modal_vs_original.csv")
        comp_df.to_csv(comp_path, index=False)
        print(f"Key comparison saved to {comp_path}")


def _write_summary(df, path, smoke, expected_models=None):
    """Write a detailed Markdown summary with failure tracking."""
    noise_types = sorted(df['noise'].unique())
    model_order = sorted(df['model'].unique())

    # Detect issues
    models_found = set(df['model'].unique())
    if expected_models:
        models_missing = set(expected_models) - models_found
    else:
        models_missing = set()

    nan_models = set()
    for _, r in df.iterrows():
        for col in ['v_rmse_mean', 'th_rmse_mean', 'modal_score_mean']:
            val = r[col]
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                nan_models.add(r['model'])
                break

    failed_models = set()
    for _, r in df.iterrows():
        if r.get('converged_mean', 1.0) < 0.5:
            failed_models.add(r['model'])

    # Build report
    lines = [
        "# Modal-Loss DSSE Experiment — Smoke Test Report",
        "",
        "**Status:** SMOKE TEST ONLY — not full paper-level results.",
        "",
        f"**Experiment:** 14-bus (CIGRE 14), {df['n_repeats'].iloc[0]} repeats, "
        f"{len(model_order)} models, {len(noise_types)} noise settings.",
        "",
        "---",
        "",
        "## Completion Status",
        "",
    ]

    if expected_models:
        all_ok = len(models_missing) == 0 and len(nan_models) == 0
        lines.append(f"- **All {len(expected_models)} target models completed:** "
                     f"{'YES' if all_ok else 'NO'}")
        if models_missing:
            lines.append(f"- **Missing models:** {', '.join(sorted(models_missing))}")
        else:
            lines.append("- **Missing models:** none")
    else:
        lines.append("- **All target models completed:** unknown (no expected list)")

    if nan_models:
        lines.append(f"- **Models with NaN/Inf metrics:** {', '.join(sorted(nan_models))}")
    else:
        lines.append("- **NaN/Inf in any metric:** NO")

    if failed_models:
        lines.append(f"- **Models with convergence failure:** {', '.join(sorted(failed_models))}")
    else:
        lines.append("- **Convergence failures:** none detected")

    lines.append(f"- **Original supervised baselines present:** "
                 f"{'YES' if 'supervised_ann' in models_found and 'supervised_dss2' in models_found else 'NO'}")
    lines.append("")

    # Per-noise comparison
    lines.append("---")
    lines.append("")
    lines.append("## Per-Noise Results")
    lines.append("")

    for noise in noise_types:
        lines.append(f"### Noise: `{noise}`")
        lines.append("")
        header = (
            "| Model | V RMSE | θ RMSE | Loading RMSE | "
            "ModalScore | HitRate | Runtime (s) | Converged |"
        )
        sep = (
            "|-------|--------|--------|-------------|"
            "------------|---------|------------|-----------|"
        )
        lines.append(header)
        lines.append(sep)

        subset = df[df['noise'] == noise]
        for model in model_order:
            rows = subset[subset['model'] == model]
            if rows.empty:
                lines.append(f"| {model} | *NOT RUN* |||||")
                continue
            r = rows.iloc[0]

            v_cell = f"{r['v_rmse_mean']:.5f}±{r['v_rmse_std']:.5f}"
            th_cell = f"{r['th_rmse_mean']:.5f}±{r['th_rmse_std']:.5f}"

            ld_val = r['loading_rmse_mean']
            if isinstance(ld_val, float) and np.isnan(ld_val):
                ld_cell = "N/A"
            else:
                ld_cell = f"{ld_val:.5f}±{r['loading_rmse_std']:.5f}"

            ms_cell = f"{r['modal_score_mean']:.4f}±{r['modal_score_std']:.4f}"
            hr_cell = f"{r['hit_rate_mean']:.4f}±{r['hit_rate_std']:.4f}"
            rt_cell = f"{r['runtime_mean']:.3f}±{r['runtime_std']:.3f}"
            cv_cell = f"{r['converged_mean']:.2f}"

            warning = ""
            if isinstance(r['v_rmse_mean'], float) and np.isnan(r['v_rmse_mean']):
                warning = " ⚠ NaN"

            lines.append(
                f"| {model}{warning} | {v_cell} | {th_cell} | {ld_cell} | "
                f"{ms_cell} | {hr_cell} | {rt_cell} | {cv_cell} |"
            )
        lines.append("")

    # Modal-vs-original comparisons (for outlier noise)
    lines.append("---")
    lines.append("")
    lines.append("## Modal vs Original Comparisons (Outlier Noise)")
    lines.append("")

    outlier_noise = "mixture_010"
    if outlier_noise in noise_types:
        subset = df[df['noise'] == outlier_noise]

        def _get(model_name, metric):
            r = subset[subset['model'] == model_name]
            if r.empty:
                return None
            return r.iloc[0][metric]

        comparisons = [
            ("wls", "modal_wls", "Modal-WLS improves over WLS"),
            ("supervised_ann", "modal_supervised_ann",
             "Modal-Supervised ANN improves over Supervised ANN"),
            ("supervised_dss2", "modal_supervised_dss2",
             "Modal-Supervised DSS2 improves over Supervised DSS2"),
        ]

        for orig_name, modal_name, description in comparisons:
            orig_v = _get(orig_name, 'v_rmse_mean')
            modal_v = _get(modal_name, 'v_rmse_mean')
            orig_ms = _get(orig_name, 'modal_score_mean')
            modal_ms = _get(modal_name, 'modal_score_mean')

            if orig_v is None:
                lines.append(f"- **{description}:** Cannot compare — `{orig_name}` not in results")
            elif modal_v is None:
                lines.append(f"- **{description}:** Cannot compare — `{modal_name}` not in results")
            elif np.isnan(orig_v) or np.isnan(modal_v):
                lines.append(f"- **{description}:** Cannot compare — NaN values present")
            else:
                v_better = modal_v < orig_v
                ms_better = modal_ms > orig_ms
                lines.append(
                    f"- **{description}:** "
                    f"V_RMSE: {orig_v:.4f} → {modal_v:.4f} "
                    f"({'BETTER' if v_better else 'WORSE'}), "
                    f"ModalScore: {orig_ms:.4f} → {modal_ms:.4f} "
                    f"({'BETTER' if ms_better else 'WORSE'})"
                )
    else:
        lines.append(f"*(No `{outlier_noise}` noise setting in this experiment)*")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Important Notes")
    lines.append("")
    lines.append("- **These are smoke-test results only.** Not full paper-level experiments.")
    lines.append("- Training: 30 epochs for neural models (insufficient for convergence).")
    lines.append("- Full experiments require: 600 epochs, 10 noise settings, 3+ repeats, "
                   "70-bus and 179-bus networks.")
    lines.append("- Weakly-supervised DSS2 is implemented but excluded from comparisons "
                   "per current experiment plan.")
    lines.append("- ModalScore and HitRate computed at reference bandwidth b=2.0.")
    lines.append("- WLS/Modal-WLS are per-sample SciPy solvers (no training required).")
    lines.append("- Learning-based models trained on clean data, evaluated on noisy data.")

    with open(path, 'w') as f:
        f.write('\n'.join(lines))


# ======================================================================
# Smoke test
# ======================================================================


if __name__ == "__main__":
    import argparse as _argparse
    _parser = _argparse.ArgumentParser(
        description="Experiment runner for DSSE modal loss experiments")
    _parser.add_argument("--case", type=str, default="cigre14",
                        help="Grid case: cigre14, ober_sub, etc. (default: cigre14)")
    _parser.add_argument("--n-repeats", type=int, default=None,
                        help="Number of repeats (default: 3 for full, 3 for smoke)")
    _parser.add_argument("--epochs", type=int, default=None,
                        help="Training epochs (default: 200 for full, 30 for smoke)")
    _parser.add_argument("--experiment-name", type=str, default=None,
                        help="Output subdirectory name")
    _parser.add_argument("--smoke", action="store_true",
                        help="Run in smoke mode (fewer models, fewer noise)")
    _parser.add_argument("--full", action="store_true",
                        help="Run in full mode (default)")
    _parser.add_argument("--skip-wls", action="store_true",
                        help="Skip WLS/Modal-WLS solvers")
    _parser.add_argument("--noise", type=str, nargs="+", default=None,
                        help="Noise settings subset")
    _parser.add_argument("--device", type=str, default=None,
                        help="Torch device (cuda, cpu, etc.)")
    _args = _parser.parse_args()

    smoke_mode = _args.smoke and not _args.full
    if _args.full:
        smoke_mode = False

    # Build folder path from case name
    folder = f"data/{_args.case}/"

    # Determine experiment name
    if _args.experiment_name:
        exp_name = _args.experiment_name
    else:
        exp_name = ("full_14bus_6models_10noise" if not smoke_mode
                     else "smoke_14bus_6models_3noise_3repeats")

    print("=" * 70)
    print("EXPERIMENT RUNNER" + (" (smoke)" if smoke_mode else " (full)"))
    print("=" * 70)

    df = run_experiments(
        folder=folder,
        n_repeats=_args.n_repeats,
        epochs=_args.epochs,
        noise_settings=_args.noise,
        smoke=smoke_mode,
        experiment_name=exp_name,
        device=_args.device,
    )

    print("\n" + "=" * 70)
    print("DONE — all outputs saved to results_modal/")
    print("=" * 70)
