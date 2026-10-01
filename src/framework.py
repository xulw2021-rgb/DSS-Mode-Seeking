"""
Unified experiment framework for the Axioms revision (Stage 2B).

Protocol compliance (EXPERIMENT_PROTOCOL.md):
  - repeat_id -> independent training run (fresh model/optimizer/loader state)
  - training_seed = 100 + repeat_id            (model, optimizer, loader shuffling)
  - data_seed     = 500 + hash(experiment_id)  (dataset build + train/test split)
  - eval_noise_seed = 700 + noise_index        (shared test-noise draws across
                                                 models and repeats -> paired design)
  - every run writes one raw row; summaries are derived from raw CSVs only.

Imports original modules (data.py, networks.py, modal_losses.py,
modal_supervised_dss2.py, modal_wls.py, experiment_runner.py) WITHOUT
modifying them.  New code lives here only.
"""

import os
import sys
import json
import time
import hashlib
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ---- import path: in the public repository all modules live in src/ ----
# (packaging adaptation: the research tree pointed three levels up; the
#  scientific content of this file is otherwise unchanged)
ORIG = os.path.dirname(os.path.abspath(__file__))
if ORIG not in sys.path:
    sys.path.insert(0, ORIG)

# work around the transformers/numpy metadata bug before torch_geometric import
import importlib.metadata as _im
_orig_v = _im.version
_im.version = lambda pkg: np.__version__ if pkg == "numpy" else _orig_v(pkg)

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from networks import ANN_DSSE, GAT_DSSE
from data import data_from_pickles, get_pflow
from modal_losses import (
    state_space_modal_loss, measurement_space_modal_loss,
    modal_score, hit_rate, gaussian_kernel_modal_loss,
)
from modal_supervised_dss2 import _physical_penalty, BandwidthSchedule
from experiment_runner import inject_noise_to_measurements, corrupt_graph_data
from modal_wls import solve_wls, solve_modal_wls, load_sample

# ======================================================================
# 1. Noise registry (index-stable: eval_noise_seed depends on the index)
# ======================================================================

NOISE_SETTINGS = [
    "default", "high_gaussian", "student_t3", "laplace",
    "mixture_005", "mixture_010", "mixture_020",
    "missing", "erroneous", "missing_erroneous",
]
NOISE_INDEX = {k: i for i, k in enumerate(NOISE_SETTINGS)}

# ======================================================================
# 2. Model registry  (MODEL_REGISTRY.md generated from this)
# ======================================================================

MODEL_REGISTRY = {
    "wls": dict(
        model_name="WLS",
        trainable=False,
        architecture="SciPy trust-region WLS (per-sample solver, no training)",
        loss="weighted least squares (measurement-space)",
        params="flat-start initialization",
    ),
    "modal_wls": dict(
        model_name="Modal-WLS",
        trainable=False,
        architecture="SciPy IRLS with Gaussian weights, 30 outer iterations",
        loss="measurement-space modal loss, b=2.0",
        params="flat-start initialization, bandwidth=2.0",
    ),
    "supervised_ann": dict(
        model_name="Supervised ANN (S-ANN)",
        trainable=True,
        architecture="ANN_DSSE(8->64->64->2, LeakyReLU, dropout=0.1)",
        loss="MSE (state-space)",
        params="Adamax lr=3e-3, batch=64",
    ),
    "modal_supervised_ann": dict(
        model_name="Modal-Supervised ANN (M-ANN)",
        trainable=True,
        architecture="ANN_DSSE(8->64->64->2, LeakyReLU, dropout=0.1)",
        loss="state-space modal loss, b=1.0 fixed, lambda_meas=0",
        params="Adamax lr=3e-3, batch=64",
    ),
    "supervised_dss2": dict(
        model_name="Supervised DSS2 (S-DSS2)",
        trainable=True,
        architecture="GAT_DSSE(dim_feat=8, dim_dense=32, dim_out=2, heads=1, num_layers=3, edge_dim=6) = 2 GATv2Conv(width 8) + dense 32->2",
        loss="gpu_wls_loss (squared standardized residuals) + 1.0*MSE",
        params="Adamax lr=3e-3, batch=32 (14-bus) / 16 (70-bus)",
    ),
    "modal_supervised_dss2": dict(
        model_name="Modal-Supervised DSS2 (M-DSS2)",
        trainable=True,
        architecture="GAT_DSSE(dim_feat=8, dim_dense=32, dim_out=2, heads=1, num_layers=3, edge_dim=6) = 2 GATv2Conv(width 8) + dense 32->2",
        loss="state-space modal loss b=1.0 + lambda_meas*meas-modal + lambda_phys*phys-penalty",
        params="Adamax lr=3e-3, batch=32/16; lambda_meas=0, lambda_phys=1.0 (E1/E2)",
    ),
}
LEARNING_MODELS = [k for k, v in MODEL_REGISTRY.items() if v["trainable"]]
SOLVER_MODELS = [k for k, v in MODEL_REGISTRY.items() if not v["trainable"]]

# ======================================================================
# 3. Seed manager
# ======================================================================

def training_seed(repeat_id: int) -> int:
    return 100 + repeat_id

def data_seed(experiment_id: str) -> int:
    h = int(hashlib.md5(experiment_id.encode()).hexdigest(), 16)
    return 500 + h % 1000

def eval_noise_seed(noise: str) -> int:
    return 700 + NOISE_INDEX[noise]

def seal_environment(training_seed_: int):
    """Seed every RNG that the training loop touches."""
    random.seed(training_seed_)
    np.random.seed(training_seed_)
    torch.manual_seed(training_seed_)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed_)

# ======================================================================
# 4. Data loading (seeded, shared across models/repeats)
# ======================================================================

def load_and_split(folder, meas_v, meas_pf, experiment_id, split_coef=0.9,
                   max_train=None, max_test=None):
    """
    Build the dataset ONCE per experiment with data_seed:
      np.random seeded before data_from_pickles (dataset noise draws),
      random.Random(data_seed) for the train/test partition.
    Returns (train_ds, test_ds, x_mean, x_std, pflow_mean, pflow_std,
             state_scale_tensor).
    """
    ds_seed = data_seed(experiment_id)
    np.random.seed(ds_seed)
    dataset, x_mean, x_std, pflow_mean, pflow_std = data_from_pickles(
        folder, 8, 6, 4, 2, meas_v, meas_pf
    )
    rng = random.Random(ds_seed)
    rng.shuffle(dataset)
    n_train = int(split_coef * len(dataset))
    if max_train is not None:
        n_train = min(n_train, max_train)
    n_test = len(dataset) - n_train
    if max_test is not None:
        n_test = min(n_test, max_test)
    train_ds = dataset[:n_train]
    test_ds = dataset[n_train:n_train + n_test]

    all_y = torch.cat([d.y for d in train_ds], dim=0)
    state_scale = torch.tensor([
        max(float(all_y[:, 0].std()), 0.005),
        max(float(all_y[:, 1].std()), 0.005),
    ])
    return train_ds, test_ds, x_mean, x_std, pflow_mean, pflow_std, state_scale

def corrupt_graph_data_lite(data, noise_type, x_mean, x_std, edge_mean, edge_std, rng):
    """
    Memory-safe equivalent of experiment_runner.corrupt_graph_data.

    The original uses deepcopy(data) per graph; because each Data tensor is a
    view of one large shared storage and deepcopy uses a fresh memo per call,
    every graph clone materializes the FULL shared storage (~13.5 MB per graph
    on cigre14) -> 876 graphs x 10 noises ~= 118 GB -> OOM.

    This version clones only the per-graph view tensors (identical corruption
    math, identical rng call order, identical outputs).
    """
    num_nfeat, num_efeat = 8, 6

    # --- node measurements (same math as corrupt_graph_data) ---
    x_in = data.x[:, :num_nfeat].numpy()
    z_nodes = x_in[:, ::2] * x_std[::2].numpy() + x_mean[::2].numpy()
    sigma_nodes = 1.0 / np.sqrt(
        np.abs(x_in[:, 1::2] * x_std[1::2].numpy() + x_mean[1::2].numpy()) + 1e-10)
    z_corr = z_nodes.copy()
    sigma_corr = sigma_nodes.copy()
    for col in range(4):
        mask = z_nodes[:, col] != 0.0
        if mask.sum() == 0:
            continue
        z_corr[mask, col], sigma_corr[mask, col] = inject_noise_to_measurements(
            z_nodes[mask, col], sigma_nodes[mask, col], noise_type, rng)
    x_corr = np.zeros_like(x_in)
    for col in range(4):
        x_corr[:, 2 * col] = (z_corr[:, col] - x_mean[2 * col].item()) / max(
            x_std[2 * col].item(), 1e-10)
        x_corr[:, 2 * col + 1] = x_in[:, 2 * col + 1]

    # --- edge measurements (same math) ---
    e_in = data.edge_attr[:, :num_efeat].numpy()
    z_edges = e_in[:, :4:2] * edge_std[:4:2].numpy() + edge_mean[:4:2].numpy()
    sigma_edges = 1.0 / np.sqrt(
        np.abs(e_in[:, 1:4:2] * edge_std[1:4:2].numpy() + edge_mean[1:4:2].numpy()) + 1e-10)
    z_ec = z_edges.copy()
    sigma_ec = sigma_edges.copy()
    for col in range(2):
        mask = z_edges[:, col] != 0.0
        if mask.sum() == 0:
            continue
        z_ec[mask, col], sigma_ec[mask, col] = inject_noise_to_measurements(
            z_edges[mask, col], sigma_edges[mask, col], noise_type, rng)
    e_corr = np.zeros_like(e_in)
    for col in range(2):
        e_corr[:, 2 * col] = (z_ec[:, col] - edge_mean[2 * col].item()) / max(
            edge_std[2 * col].item(), 1e-10)
        e_corr[:, 2 * col + 1] = e_in[:, 2 * col + 1]

    # --- lite container copy: clone view-sized tensors only ---
    from torch_geometric.data import Data as _Data
    new_x = data.x.clone()
    new_x[:, :num_nfeat] = torch.tensor(x_corr, dtype=torch.float32)
    new_edge = data.edge_attr.clone()
    new_edge[:, :num_efeat] = torch.tensor(e_corr, dtype=torch.float32)
    return _Data(x=new_x, edge_index=data.edge_index, edge_attr=new_edge, y=data.y)


def corrupt_test_sets(test_ds, noise_list, x_mean, x_std, pflow_mean, pflow_std):
    """
    One corrupted copy of the test set per noise condition, seeded by
    eval_noise_seed(noise).  Shared by ALL models and repeats (paired design).
    Uses the memory-safe corrupt_graph_data_lite (identical corruption math).
    """
    out = {}
    for noise in noise_list:
        rng = np.random.RandomState(eval_noise_seed(noise))
        out[noise] = [
            corrupt_graph_data_lite(d, noise, x_mean, x_std, pflow_mean, pflow_std, rng)
            for d in test_ds
        ]
    return out

# ======================================================================
# 5. GPU-compatible WLS-shaped loss (adapter for S-DSS2 on GPU)
#    Same formulation as run_fair_70bus_dss2.gpu_wls_loss.
# ======================================================================

def gpu_wls_loss(x_norm, edge_attr_norm, out_norm, edge_index, node_param,
                 edge_param, x_mean_d, x_std_d, em_d, es_d):
    z_nodes = x_norm[:, ::2] * x_std_d[::2] + x_mean_d[::2]
    r_inv = x_norm[:, 1::2]
    sigma_nodes = 1.0 / torch.sqrt(torch.abs(r_inv * x_std_d[1::2] + x_mean_d[1::2]) + 1e-10)
    z_edges = edge_attr_norm[:, :4:2] * es_d[:4:2] + em_d[:4:2]
    r_edge_inv = edge_attr_norm[:, 1:4:2]
    sigma_edges = 1.0 / torch.sqrt(torch.abs(r_edge_inv * es_d[1:4:2] + em_d[1:4:2]) + 1e-10)

    V_pred = out_norm[:, 0:1] * x_std_d[0:1] + x_mean_d[0:1]
    theta_pred = out_norm[:, 1:2].clone()
    theta_pred *= (1.0 - node_param[:, 1:2])
    y_global = torch.cat([V_pred, theta_pred], dim=1)

    _, _, Pf, Qf, Pt, Qt, _, _ = get_pflow(y_global, edge_index, node_param, edge_param)
    n_nodes = x_norm.shape[0]
    idx_f = edge_index[0]; idx_t = edge_index[1]
    P_inj = -scatter(Pt, idx_t, dim_size=n_nodes) - scatter(Pf, idx_f, dim_size=n_nodes)
    Q_inj = -scatter(Qt, idx_t, dim_size=n_nodes) - scatter(Qf, idx_f, dim_size=n_nodes)
    h_node = torch.cat([V_pred, theta_pred, P_inj.unsqueeze(1), Q_inj.unsqueeze(1)], dim=1)
    h_edge = torch.cat([Pf.unsqueeze(1), Qf.unsqueeze(1)], dim=1)

    mask_n = (z_nodes != 0).float()
    mask_e = (z_edges != 0).float()
    total, n_total = 0.0, 0
    for col in range(4):
        m = mask_n[:, col]
        if m.sum() == 0:
            continue
        r = (z_nodes[:, col] - h_node[:, col]) / (sigma_nodes[:, col] + 1e-10)
        total += (r[m > 0] ** 2).sum()
        n_total += int(m.sum().item())
    for col in range(2):
        m = mask_e[:, col]
        if m.sum() == 0:
            continue
        r = (z_edges[:, col] - h_edge[:, col]) / (sigma_edges[:, col] + 1e-10)
        total += (r[m > 0] ** 2).sum()
        n_total += int(m.sum().item())
    return total / max(n_total, 1)

# ======================================================================
# 6. Training functions  (one independent run each)
# ======================================================================

def make_model(model_id, device):
    if model_id in ("supervised_ann", "modal_supervised_ann"):
        return ANN_DSSE(dim_in=8, dim_hid=64, dim_out=2, num_layers=3,
                        dropout=0.1).to(device)
    if model_id in ("supervised_dss2", "modal_supervised_dss2"):
        return GAT_DSSE(dim_feat=8, dim_dense=32, dim_out=2, heads=1,
                        num_layers=3, edge_dim=6).to(device)
    raise ValueError(model_id)

def train_learning_model(model_id, train_ds, epochs, batch_size, device,
                         x_mean, x_std, pflow_mean, pflow_std, state_scale,
                         repeat_id, bandwidth=1.0, lambda_meas=0.0,
                         lambda_phys=1.0, annealing=False, b_start=8.0,
                         b_end=1.0, verbose=False):
    """
    One complete independent training run (repeat).
    Returns (model, train_loss_last, history_dict).
    """
    seal_environment(training_seed(repeat_id))
    model = make_model(model_id, device)
    opt = torch.optim.Adamax(model.parameters(), lr=3e-3)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    x_mean_d = x_mean.to(device); x_std_d = x_std.to(device)
    em_d = pflow_mean.to(device); es_d = pflow_std.to(device)
    ss = state_scale.to(device)

    sched = BandwidthSchedule(b_start, b_end, float(epochs) / 3.0) if annealing else None

    hist = {"train_loss": [], "b": []}
    is_gnn = model_id.endswith("dss2")
    is_modal = model_id.startswith("modal")

    for epoch in range(epochs):
        b_cur = sched.get(epoch) if sched else bandwidth
        hist["b"].append(b_cur)
        model.train()
        tot = 0.0; nb = 0
        for data in loader:
            data = data.to(device)
            x_in = data.x[:, :8]
            e_in = data.edge_attr[:, :6]
            y_true = data.y
            node_p = data.x[:, 8:]
            edge_p = data.edge_attr[:, 6:]
            opt.zero_grad()
            if is_gnn:
                out = model(x_in, data.edge_index, e_in)
            else:
                out = model(x_in)

            if model_id == "supervised_ann":
                loss = F.mse_loss(out, y_true)
            elif model_id == "modal_supervised_ann":
                loss = state_space_modal_loss(y_true, out, ss, b_cur, reduction="mean")
            elif model_id == "supervised_dss2":
                loss_wls = gpu_wls_loss(x_in, e_in, out, data.edge_index,
                                        node_p, edge_p, x_mean_d, x_std_d, em_d, es_d)
                loss = loss_wls + 1.0 * F.mse_loss(out, y_true)
            elif model_id == "modal_supervised_dss2":
                if lambda_meas > 0:
                    raise ValueError(
                        "lambda_meas>0 must use train_modal_dss2_with_meas_term()")
                loss_state = state_space_modal_loss(y_true, out, ss, b_cur, reduction="mean")
                loss = loss_state + lambda_phys * _physical_penalty(
                    out, data.edge_index, node_p, edge_p, x_std_d, x_mean_d)
            else:
                raise ValueError(model_id)

            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        epoch_loss = tot / max(nb, 1)
        if not np.isfinite(epoch_loss):
            raise TrainingDivergedError(epoch, epoch_loss)
        hist["train_loss"].append(epoch_loss)
        if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1):
            print(f"    [{model_id}] epoch {epoch:4d}: loss={hist['train_loss'][-1]:.5f}")

    return model, hist["train_loss"][-1], hist

def dss2_meas_modal_loss(x_norm, e_in, out, edge_index, node_p, edge_p, b_cur,
                         x_mean_d, x_std_d, em_d, es_d):
    """Module-level measurement-space modal loss for Modal-DSS2 (E4).

    L_meas = mean_k [1 - exp(-0.5 * r_k^2 / b^2)],  r_k = (z_k - h_k(x))/sigma_k
    over node (V, theta, P, Q) and edge (P_flow, Q_flow) measurements,
    with zero-measurement masking. Same formulation as the original
    modal_supervised_dss2._meas_modal_loss.
    """
    z_nodes = x_norm[:, ::2] * x_std_d[::2] + x_mean_d[::2]
    r_inv = x_norm[:, 1::2]
    sigma_nodes = 1.0 / torch.sqrt(torch.abs(r_inv * x_std_d[1::2] + x_mean_d[1::2]) + 1e-10)
    z_edges = e_in[:, :4:2] * es_d[:4:2] + em_d[:4:2]
    r_edge_inv = e_in[:, 1:4:2]
    sigma_edges = 1.0 / torch.sqrt(torch.abs(r_edge_inv * es_d[1:4:2] + em_d[1:4:2]) + 1e-10)
    V_pred = out[:, 0:1] * x_std_d[0:1] + x_mean_d[0:1]
    theta_pred = out[:, 1:2].clone() * (1.0 - node_p[:, 1:2])
    y_global = torch.cat([V_pred, theta_pred], dim=1)
    _, _, Pf, Qf, Pt, Qt, _, _ = get_pflow(y_global, edge_index, node_p, edge_p)
    n_nodes = x_norm.shape[0]
    idx_f = edge_index[0]; idx_t = edge_index[1]
    P_inj = -scatter(Pt, idx_t, dim_size=n_nodes) - scatter(Pf, idx_f, dim_size=n_nodes)
    Q_inj = -scatter(Qt, idx_t, dim_size=n_nodes) - scatter(Qf, idx_f, dim_size=n_nodes)
    h_node = torch.cat([V_pred, theta_pred, P_inj.unsqueeze(1), Q_inj.unsqueeze(1)], dim=1)
    h_edge = torch.cat([Pf.unsqueeze(1), Qf.unsqueeze(1)], dim=1)
    mask_n = (z_nodes != 0).float(); mask_e = (z_edges != 0).float()
    total, n_total = 0.0, 0
    for col in range(4):
        m = mask_n[:, col]
        if m.sum() == 0: continue
        zc = z_nodes[:, col][m > 0]; hc = h_node[:, col][m > 0]
        sc = sigma_nodes[:, col][m > 0]
        total += gaussian_kernel_modal_loss((zc - hc) / sc, b_cur, reduction="sum")
        n_total += int(m.sum().item())
    for col in range(2):
        m = mask_e[:, col]
        if m.sum() == 0: continue
        zc = z_edges[:, col][m > 0]; hc = h_edge[:, col][m > 0]
        sc = sigma_edges[:, col][m > 0]
        total += gaussian_kernel_modal_loss((zc - hc) / sc, b_cur, reduction="sum")
        n_total += int(m.sum().item())
    return total / max(n_total, 1)


def train_modal_dss2_with_meas_term(model_id, train_ds, epochs, batch_size,
                                    device, x_mean, x_std, pflow_mean,
                                    pflow_std, state_scale, repeat_id,
                                    bandwidth=1.0, lambda_meas=0.1,
                                    lambda_phys=1.0, annealing=False):
    """
    E4 path: Modal-Supervised DSS2 with the measurement-space modal term.

    L_total = L_state + lambda_meas * L_meas + lambda_phys * penalty(x)

    Returns (model, train_loss_final, hist) with hist containing
    train_loss, train_state_loss, train_meas_loss, b.
    """
    seal_environment(training_seed(repeat_id))
    model = make_model(model_id, device)
    opt = torch.optim.Adamax(model.parameters(), lr=3e-3)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    x_mean_d = x_mean.to(device); x_std_d = x_std.to(device)
    em_d = pflow_mean.to(device); es_d = pflow_std.to(device)
    ss = state_scale.to(device)
    sched = BandwidthSchedule(8.0, bandwidth, float(epochs) / 3.0) if annealing else None

    hist = {"train_loss": [], "train_state_loss": [], "train_meas_loss": [],
            "b": []}
    for epoch in range(epochs):
        b_cur = sched.get(epoch) if sched else bandwidth
        hist["b"].append(b_cur)
        model.train(); tot = 0.0; tot_s = 0.0; tot_m = 0.0; nb = 0
        for data in loader:
            data = data.to(device)
            x_in = data.x[:, :8]; e_in = data.edge_attr[:, :6]
            y_true = data.y
            node_p = data.x[:, 8:]; edge_p = data.edge_attr[:, 6:]
            opt.zero_grad()
            out = model(x_in, data.edge_index, e_in)
            loss_state = state_space_modal_loss(y_true, out, ss, b_cur, reduction="mean")
            loss_meas = dss2_meas_modal_loss(
                x_in, e_in, out, data.edge_index, node_p, edge_p, b_cur,
                x_mean_d, x_std_d, em_d, es_d)
            loss = loss_state + lambda_meas * loss_meas + lambda_phys * _physical_penalty(
                out, data.edge_index, node_p, edge_p, x_std_d, x_mean_d)
            loss.backward(); opt.step()
            tot += loss.item(); tot_s += loss_state.item()
            tot_m += loss_meas.item(); nb += 1
        epoch_loss = tot / max(nb, 1)
        if not np.isfinite(epoch_loss):
            raise TrainingDivergedError(epoch, epoch_loss)
        hist["train_loss"].append(epoch_loss)
        hist["train_state_loss"].append(tot_s / max(nb, 1))
        hist["train_meas_loss"].append(tot_m / max(nb, 1))
    return model, hist["train_loss"][-1], hist

# ======================================================================
# 7. Evaluation (learning models) — one corrupted test set per noise
# ======================================================================

def evaluate_learning_model(model, model_id, corrupted_by_noise, x_mean, x_std,
                            state_scale, device, batch_size=64):
    """Returns {noise: metrics_dict}; metrics averaged over test batches."""
    num_nfeat, num_efeat = 8, 6
    is_gnn = model_id.endswith("dss2")
    ss = state_scale.to(device)
    out_metrics = {}
    model.eval()
    with torch.no_grad():
        for noise, corr in corrupted_by_noise.items():
            loader = DataLoader(corr, batch_size=batch_size, shuffle=False)
            v_r, th_r, ld_r, ms, hr, v_a, th_a = [], [], [], [], [], [], []
            t0 = time.time()
            for data in loader:
                data = data.to(device)
                x_in = data.x[:, :num_nfeat]
                y_true = data.y
                node_p = data.x[:, num_nfeat:]
                edge_p = data.edge_attr[:, num_efeat:]
                if is_gnn:
                    out = model(x_in, data.edge_index, data.edge_attr[:, :num_efeat])
                else:
                    out = model(x_in)
                out_un = out.clone()
                out_un[:, 0] = out[:, 0] * x_std[0].to(device) + x_mean[0].to(device)
                out_un[:, 1] *= (1.0 - node_p[:, 1])
                v_r.append(torch.sqrt(F.mse_loss(out_un[:, :1], y_true[:, :1])).item())
                th_r.append(torch.sqrt(F.mse_loss(out_un[:, 1:], y_true[:, 1:])).item())
                v_a.append(F.l1_loss(out_un[:, :1], y_true[:, :1]).item())
                th_a.append(F.l1_loss(out_un[:, 1:], y_true[:, 1:]).item())
                true_ll, true_lt, _, _, _, _, _, _ = get_pflow(y_true, data.edge_index, node_p, edge_p)
                out_ll, out_lt, _, _, _, _, _, _ = get_pflow(out_un, data.edge_index, node_p, edge_p)
                m = true_ll.nonzero(as_tuple=True)[0]
                if m.numel() > 0:
                    ld_r.append(torch.sqrt(F.mse_loss(out_ll[m], true_ll[m])).item())
                ms.append(modal_score(y_true, out_un, ss, 2.0).item())
                hr.append(hit_rate(y_true, out_un, ss, 3.0).item())
            runtime = time.time() - t0
            out_metrics[noise] = {
                "v_rmse": float(np.mean(v_r)), "th_rmse": float(np.mean(th_r)),
                "loading_rmse": float(np.mean(ld_r)) if ld_r else np.nan,
                "modal_score": float(np.mean(ms)), "hit_rate": float(np.mean(hr)),
                "runtime_seconds": runtime,
                "v_mae": float(np.mean(v_a)), "th_mae": float(np.mean(th_a)),
            }
    return out_metrics

# ======================================================================
# 8. Evaluation (WLS / Modal-WLS per-sample solvers)
# ======================================================================

def build_wls_samples(folder, meas_v_buses, n_samples=10):
    samples = []
    for idx in range(n_samples):
        meas, node_param, edge_param, edge_index, labels = load_sample(
            folder, sample_idx=idx, meas_v_buses=meas_v_buses)
        samples.append(dict(meas=meas, node_param=node_param, edge_param=edge_param,
                            edge_index=edge_index,
                            true_V=labels["vm_pu"].values.astype(np.float64),
                            true_theta=labels["va_rad"].values.astype(np.float64)))
    return samples

def evaluate_solver_model(model_id, wls_samples, noise_list, n_draws=5):
    """Deterministic per-sample solver; repeats = independent noise draws
    seeded by eval_noise_seed(noise).

    Numerical failures (e.g. scipy LinAlgError: SVD did not converge on
    ill-conditioned corrupted Jacobians) are caught per draw: the draw is
    counted as non-converged (converged=0) and excluded from the error
    metrics. Failure counts are reported in 'n_failed' so nothing is hidden.
    """
    out_metrics = {}
    for noise in noise_list:
        rng = np.random.RandomState(eval_noise_seed(noise))
        v_r, th_r, ms, hr, rt, cv = [], [], [], [], [], []
        n_failed = 0
        for sample in wls_samples:
            z_clean = sample["meas"]["z"].copy()
            sigma_clean = sample["meas"]["sigma"].copy()
            for _ in range(n_draws):
                z_c, s_c = inject_noise_to_measurements(z_clean, sigma_clean, noise, rng)
                meas_c = dict(sample["meas"]); meas_c["z"] = z_c; meas_c["sigma"] = s_c
                t0 = time.time()
                try:
                    if model_id == "modal_wls":
                        res = solve_modal_wls(meas_c, sample["node_param"],
                                              sample["edge_param"], sample["edge_index"],
                                              bandwidth=2.0, lam_phys=0.0, max_iter=30, verbose=0)
                    else:
                        res = solve_wls(meas_c, sample["node_param"],
                                        sample["edge_param"], sample["edge_index"],
                                        lam_phys=0.0)
                    V_est = res["V_est"]; th_est = res["theta_est"]
                    ok = np.all(np.isfinite(V_est)) and np.all(np.isfinite(th_est)) \
                        and bool(res.get("converged", False))
                except Exception as e:
                    n_failed += 1
                    rt.append(time.time() - t0)
                    cv.append(0.0)
                    continue
                rt.append(time.time() - t0)
                cv.append(float(ok))
                if not ok:
                    n_failed += 1
                    continue
                v_r.append(np.sqrt(np.mean((V_est - sample["true_V"]) ** 2)))
                th_r.append(np.sqrt(np.mean((th_est - sample["true_theta"]) ** 2)))
                x_true = np.concatenate([sample["true_V"], sample["true_theta"]])
                x_est = np.concatenate([V_est, th_est])
                sc = np.concatenate([
                    np.full(len(sample["true_V"]), max(np.std(sample["true_V"]), 0.005)),
                    np.full(len(sample["true_theta"]), max(np.std(sample["true_theta"]), 0.005))])
                r = (x_true - x_est) / (sc + 1e-8)
                ms.append(float(np.mean(np.exp(-0.5 * (r / 2.0) ** 2))))
                hr.append(float(np.mean(np.abs(r) <= 3.0)))
        out_metrics[noise] = {
            "v_rmse": float(np.mean(v_r)) if v_r else np.nan,
            "th_rmse": float(np.mean(th_r)) if th_r else np.nan,
            "loading_rmse": np.nan,
            "modal_score": float(np.mean(ms)) if ms else np.nan,
            "hit_rate": float(np.mean(hr)) if hr else np.nan,
            "runtime_seconds": float(np.mean(rt)) if rt else np.nan,
            "converged": float(np.mean(cv)) if cv else np.nan,
            "n_failed": n_failed,
            "n_draws_total": len(wls_samples) * n_draws,
        }
    return out_metrics

# ======================================================================
# 9. Result writer / summary / manifest
# ======================================================================

RAW_COLUMNS = [
    "experiment_id", "dataset_id", "data_source", "external_benchmark",
    "model", "model_name", "architecture", "loss",
    "noise", "bad_data_rate", "bandwidth", "bandwidth_schedule", "lambda_meas",
    "eta", "repeat_id", "training_seed", "data_seed", "eval_noise_seed",
    "epochs", "best_epoch", "train_loss", "val_loss",
    "v_rmse", "th_rmse", "loading_rmse", "modal_score", "hit_rate",
    "runtime_seconds", "converged", "device", "gpu_name",
    "torch_version", "cuda_version", "status", "failure_epoch",
    "failure_reason", "v_mae", "th_mae", "state_loss", "meas_loss",
    "notes",
]

# run status vocabulary (E2 protocol §12-13)
STATUS_SUCCESS = "SUCCESS"
STATUS_TRAINING_FAILURE = "TRAINING_FAILURE"      # non-finite loss / diverged
STATUS_EVALUATION_FAILURE = "EVALUATION_FAILURE"  # trained ok, eval crashed
STATUS_TECHNICAL_ERROR = "TECHNICAL_ERROR"        # unexpected exception

class TrainingDivergedError(RuntimeError):
    """Raised when the training loss becomes non-finite."""
    def __init__(self, epoch: int, loss: float):
        self.epoch = epoch
        self.loss = loss
        super().__init__(f"non-finite training loss at epoch {epoch} (loss={loss})")

def gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

class ResultWriter:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            pd.DataFrame(columns=RAW_COLUMNS).to_csv(path, index=False)

    def append(self, row: dict):
        df = pd.DataFrame([row])
        df = df.reindex(columns=RAW_COLUMNS)
        df.to_csv(self.path, mode="a", header=False, index=False)

def summarize(raw_csv: str, out_csv: str, metrics=("v_rmse", "th_rmse",
              "loading_rmse", "modal_score", "hit_rate")):
    """Summary: n, mean, std, median, min, max per (model, noise)."""
    df = pd.read_csv(raw_csv)
    rows = []
    for (model, noise), g in df.groupby(["model", "noise"]):
        rec = {"model": model, "noise": noise, "n": len(g)}
        for m in metrics:
            v = g[m].dropna()
            if len(v) == 0:
                rec.update({f"{m}_mean": np.nan, f"{m}_std": np.nan,
                            f"{m}_median": np.nan, f"{m}_min": np.nan,
                            f"{m}_max": np.nan})
            else:
                rec.update({f"{m}_mean": v.mean(), f"{m}_std": v.std(),
                            f"{m}_median": v.median(), f"{m}_min": v.min(),
                            f"{m}_max": v.max()})
        rows.append(rec)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv

def paired_bootstrap(raw_csv: str, out_csv: str, metric="v_rmse",
                     pairs=None, n_boot=10000, seed=0):
    """
    Paired bootstrap CI on the per-repeat paired differences.
    pairs: list of (model_a, model_b, label_a, label_b).
    """
    df = pd.read_csv(raw_csv)
    rng = np.random.RandomState(seed)
    rows = []
    for model_a, model_b, la, lb in pairs:
        for noise, g in df.groupby("noise"):
            a = g[g.model == model_a].sort_values("repeat_id")[metric].values
            b = g[g.model == model_b].sort_values("repeat_id")[metric].values
            n = min(len(a), len(b))
            if n < 2:
                continue
            a, b = a[:n], b[:n]
            d = a - b
            boot = np.array([rng.choice(d, n, replace=True).mean() for _ in range(n_boot)])
            rows.append({
                "noise": noise, "model_a": la, "model_b": lb, "metric": metric,
                "n_pairs": n, "mean_diff": d.mean(), "ci_low": np.percentile(boot, 2.5),
                "ci_high": np.percentile(boot, 97.5),
                "p_two_sided": float(min(1.0, 2 * min(
                    (boot <= 0).mean() if d.mean() > 0 else (boot >= 0).mean(), 0.5))),
            })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv

def write_manifest(experiment_id, out_dir, cfg: dict, dataset_hash: str = "",
                   notes: str = ""):
    import torch as _t
    man = {
        "experiment_id": experiment_id,
        "config": cfg,
        "environment": "conda (validated environment documented in docs/REPRODUCIBILITY.md)",
        "python": sys.version.split()[0],
        "torch": _t.__version__,
        "cuda": _t.version.cuda,
        "gpu": gpu_name(),
        "dataset_hash": dataset_hash,
        "config_hash": hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": notes,
    }
    path = os.path.join(out_dir, "RUN_MANIFEST.json")
    with open(path, "w") as f:
        json.dump(man, f, indent=2)
    return path


# ======================================================================
# 10. Corrected (train-only normalization) dataset builder — Stage 2C.
#     Mirrors data.data_from_pickles feature construction EXACTLY
#     (same noise-model math, same np.random seeding => identical raw
#     features), but computes normalization statistics on the TRAIN
#     partition only and applies them to train/val/test.
#     Legacy path (data_from_pickles + load_and_split) is untouched.
# ======================================================================

def _build_raw_features(folder, meas_v, meas_pf):
    """Mirror of data_from_pickles lines 96-178: raw (unnormalized)
    per-sample features, concatenated with row ranges."""
    import pickle as _pickle
    with open(folder + "nodes", "rb") as f:
        df_nodes_list = _pickle.load(f)
    with open(folder + "edges", "rb") as f:
        df_edges_list = _pickle.load(f)
    with open(folder + "labels", "rb") as f:
        df_labels_list = _pickle.load(f)
    with open(folder + "noise_param", "rb") as f:
        df_noise_list = _pickle.load(f)

    nodes_noises = df_noise_list[["v_noise", "v_noise", "PM_NOISE",
                                  "PM_NOISE"]].values
    zero_inj_noises = df_noise_list[["zero_inj_coef", "zero_inj_coef"]].values
    slack_noise = df_noise_list[["v_noise", "zero_inj_coef", "p_noise",
                                 "p_noise"]].values
    pflow_noises = df_noise_list[["p_noise", "p_noise"]].values

    x_rows, edge_rows, y_rows = [], [], []
    node_ranges, edge_ranges = [], []
    edge_index_list = []
    n_nodes_acc, n_edges_acc = 0, 0

    for i in range(len(df_nodes_list)):
        num_nodes = df_nodes_list[i].shape[0]
        meas_bus_mask = np.ones([num_nodes, 4]) * [0, 0, 1, 1]
        for j in meas_v:
            meas_bus_mask[j][0] = 1.0
        x_mean = np.multiply(
            df_nodes_list[i][["vm_pu", "va_rad", "p_mw", "q_mvar"]].values,
            meas_bus_mask)
        slack = df_nodes_list[i]["bool_slack"].values
        x_std = x_mean * (slack_noise * np.expand_dims(slack, axis=1)
                          + nodes_noises * (1 - np.expand_dims(slack, axis=1)))
        x = torch.tensor(x_mean + np.random.normal(
            loc=0.0, scale=np.abs(x_std)), dtype=torch.float32)
        x_std[:, 2:] += zero_inj_noises * np.expand_dims(
            df_nodes_list[i]["bool_zero_inj"].values, axis=1)
        x_std[:, 1:2] += slack_noise[:, 1:2] * np.expand_dims(slack, axis=1)
        x_cov = torch.tensor(1, dtype=torch.float32) / torch.maximum(
            torch.abs(torch.tensor(x_std, dtype=torch.float32)),
            torch.tensor(1e-6, dtype=torch.float32)) ** 2
        x_cov *= (x_cov < 1e12).type(torch.float32)
        x8 = torch.concat([x[:, 0:1], x_cov[:, 0:1], x[:, 1:2], x_cov[:, 1:2],
                           x[:, 2:3], x_cov[:, 2:3], x[:, 3:], x_cov[:, 3:]],
                          axis=1)

        df_closed = df_edges_list[i][df_edges_list[i]["closed line"] == 1.0]
        num_lines = df_closed.shape[0]
        meas_pflow_mask = np.zeros([num_lines, 2])
        for j in meas_pf:
            meas_pflow_mask[j] = np.ones([1, 2])
        edge_index = torch.tensor(
            df_closed[["from_bus", "to_bus"]].values.astype(int),
            dtype=torch.long).t().contiguous()
        edge_attr_mean = np.multiply(
            df_closed[["p_from_mw", "q_from_mvar"]].values, meas_pflow_mask)
        edge_attr_std = edge_attr_mean * pflow_noises
        edge_attr = torch.tensor(edge_attr_mean + np.random.normal(
            loc=0.0, scale=np.abs(edge_attr_std)), dtype=torch.float32)
        edge_attr_cov = torch.tensor(1, dtype=torch.float32) / torch.maximum(
            torch.abs(torch.tensor(edge_attr_std, dtype=torch.float32)),
            torch.tensor(1e-5, dtype=torch.float32)) ** 2
        edge_attr_cov *= (edge_attr_cov < 1e10).type(torch.float32)
        edge_imp = torch.tensor(df_closed[["G", "B"]].values,
                                dtype=torch.float32)
        e6 = torch.concat([edge_attr[:, 0:1], edge_attr_cov[:, 0:1],
                           edge_attr[:, 1:], edge_attr_cov[:, 1:], edge_imp],
                          axis=1)

        y = torch.tensor(df_labels_list[i].values, dtype=torch.float32)
        nodes_param = torch.tensor(
            df_nodes_list[i][["vn_kv", "bool_slack", "bool_zero_inj"]].values,
            dtype=torch.float32)
        edges_param = torch.tensor(
            df_closed[["G", "B", "Gs", "Bs", "closed line", "phase shift",
                       "imax or sn"]].values, dtype=torch.float32)

        x_rows.append(torch.concat([x8, nodes_param], axis=1))
        edge_rows.append(torch.concat([e6, edges_param], axis=1))
        y_rows.append(y)
        node_ranges.append((n_nodes_acc, n_nodes_acc + num_nodes))
        n_nodes_acc += num_nodes
        edge_ranges.append((n_edges_acc, n_edges_acc + num_lines))
        n_edges_acc += num_lines
        edge_index_list.append(edge_index)

    X = torch.cat(x_rows, dim=0)      # (N_total, 11)
    E = torch.cat(edge_rows, dim=0)   # (E_total, 13)
    Y = torch.cat(y_rows, dim=0)      # (N_total, 2)
    return (X, E, Y, node_ranges, edge_ranges, edge_index_list,
            df_noise_list)


def _masked_stats(X, num_nfeat, row_idx):
    """Masked mean/std over the given row indices (train only)."""
    sub = X[row_idx]
    mask = sub != 0.0
    mean = torch.nan_to_num((sub * mask).sum(dim=[0]) / mask.sum(dim=[0]))
    std = torch.nan_to_num(
        torch.sqrt(((sub - mean) ** 2 * mask).sum(dim=[0]) / mask.sum(dim=[0])))
    return mean[:num_nfeat], std[:num_nfeat]


def build_dataset_train_only(folder, meas_v, meas_pf, experiment_id,
                             split_coef=0.9, max_train=None, max_test=None,
                             _test_hook=None):
    """
    Corrected pipeline (preprocessing_version='train_only'):
    raw features (identical construction/noise draws as legacy) ->
    split by data_seed -> normalization stats fitted on TRAIN partition only
    -> applied to train and test.  Returns the same tuple as load_and_split.
    _test_hook (unit tests only): called as
        hook(X, E, Y, train_idx, test_idx) and may modify test rows in place
        BEFORE stats computation.
    """
    ds_seed = data_seed(experiment_id)
    np.random.seed(ds_seed)                      # same noise draws as legacy
    X, E, Y, node_ranges, edge_ranges, edge_index_list, _ = \
        _build_raw_features(folder, meas_v, meas_pf)

    n_samples = len(node_ranges)
    order = list(range(n_samples))
    random.Random(ds_seed).shuffle(order)        # same split as legacy
    n_train = int(split_coef * n_samples)
    if max_train is not None:
        n_train = min(n_train, max_train)
    n_test = n_samples - n_train
    if max_test is not None:
        n_test = min(n_test, max_test)
    train_order = order[:n_train]
    test_order = order[n_train:n_train + n_test]

    train_row_idx = np.concatenate(
        [np.arange(a, b) for (a, b) in
         [node_ranges[k] for k in train_order]]).astype(np.int64)
    train_edge_idx = np.concatenate(
        [np.arange(a, b) for (a, b) in
         [edge_ranges[k] for k in train_order]]).astype(np.int64)
    test_row_idx = np.concatenate(
        [np.arange(a, b) for (a, b) in
         [node_ranges[k] for k in test_order]]).astype(np.int64)

    if _test_hook is not None:
        _test_hook(X, E, Y, train_row_idx, test_row_idx)

    x_mean, x_std = _masked_stats(X, 8, train_row_idx)
    pflow_mean, pflow_std = _masked_stats(E, 6, train_edge_idx)

    # normalize the 8/6 feature columns with the train-only stats;
    # node/edge param columns stay raw (equivalent to the legacy pipeline,
    # which normalized all columns then overwrote params with raw values)
    x_mask = X[:, :8] != 0.0
    X_n = X.clone()
    X_n[:, :8] = torch.nan_to_num((X[:, :8] - x_mean) * x_mask / x_std)
    e_mask = E[:, :6] != 0.0
    E_n = E.clone()
    E_n[:, :6] = torch.nan_to_num((E[:, :6] - pflow_mean) * e_mask / pflow_std)

    def make_ds(sample_ids):
        out = []
        for k in sample_ids:
            n0, n1 = node_ranges[k]
            e0, e1 = edge_ranges[k]
            out.append(Data(x=X_n[n0:n1], edge_index=edge_index_list[k],
                            edge_attr=E_n[e0:e1], y=Y[n0:n1]))
        return out

    train_ds = make_ds(train_order)
    test_ds = make_ds(test_order)
    all_y = torch.cat([d.y for d in train_ds], dim=0)
    state_scale = torch.tensor([
        max(float(all_y[:, 0].std()), 0.005),
        max(float(all_y[:, 1].std()), 0.005),
    ])
    return train_ds, test_ds, x_mean, x_std, pflow_mean, pflow_std, state_scale
