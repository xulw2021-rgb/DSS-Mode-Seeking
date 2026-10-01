"""
Modal-Supervised ANN training for distribution system state estimation.

Replaces the standard MSE loss with the state-space modal loss:

    L_state = mean_j[1 - exp(-0.5 * ((x_true_j - x_pred_j) / s_j / b_x)^2)]

and optionally adds a hybrid measurement consistency term:

    L_meas = mean_k[1 - exp(-0.5 * ((z_k - h_k(x_pred)) / sigma_k / b_z)^2)]

    L_total = L_state + eta * L_meas

The ANN treats each bus independently (MLP), using no graph structure.
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
import time

# Work around a transformers/numpy metadata bug affecting torch_geometric import
import importlib.metadata
_orig_metadata_version = importlib.metadata.version
importlib.metadata.version = lambda pkg: (
    np.__version__ if pkg == 'numpy' else _orig_metadata_version(pkg)
)

from torch_geometric.loader import DataLoader

from networks import ANN_DSSE
from data import data_from_pickles, get_pflow
from modal_losses import (
    state_space_modal_loss,
    measurement_space_modal_loss,
    modal_score,
    hit_rate,
    gaussian_kernel_modal_loss,
)

# ---------------------------------------------------------------------------
# Data preparation: flatten graph data to node-level
# ---------------------------------------------------------------------------


def _flatten_graphs_to_nodes(dataset, num_nfeat):
    """Convert list of PyG Data objects to (X_nodes, y_nodes) tensors."""
    xs, ys = [], []
    for data in dataset:
        xs.append(data.x[:, :num_nfeat])
        ys.append(data.y)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


# ---------------------------------------------------------------------------
# Measurement consistency term
# ---------------------------------------------------------------------------


def _measurement_modal_loss_on_batch(
    x_norm, edge_attr_norm, out_norm, edge_index, batch_idx,
    x_mean, x_std, edge_mean, edge_std,
    node_param, edge_param, bandwidth_z
):
    """
    Compute L_meas = mean_k[1 - exp(-0.5 * r_k^2 / b_z^2)]
    where r_k = (z_k - h_k(x_pred)) / sigma_k.

    Handles batched PyG Data (multiple graphs in one batch).
    Works graph-by-graph within the batch.
    """
    total_nodes = x_norm.shape[0]
    num_nfeat = 8  # V, Cov_V, θ, Cov_θ, P, Cov_P, Q, Cov_Q
    num_efeat = 6  # P_flow, Cov_Pf, Q_flow, Cov_Qf, G, B

    # Un-normalize predictions
    V_pred = out_norm[:, 0:1] * x_std[0:1] + x_mean[0:1]
    theta_pred = out_norm[:, 1:2].clone()
    theta_pred *= (1.0 - node_param[:, 1:2])  # enforce slack θ = 0

    # Un-normalize node measurements: z (V, θ, P, Q)
    z_nodes = x_norm[:, ::2] * x_std[::2] + x_mean[::2]  # (N, 4)
    r_inv = x_norm[:, 1::2]  # inverse covariances (normalized)
    sigma_nodes = 1.0 / torch.sqrt(
        torch.abs(r_inv * x_std[1::2] + x_mean[1::2]) + 1e-10
    )

    # Un-normalize edge measurements: z_edge (P_flow, Q_flow)
    edge_z = edge_attr_norm[:, :4:2] * edge_std[:4:2] + edge_mean[:4:2]  # (E, 2)
    r_edge_inv = edge_attr_norm[:, 1:4:2]
    sigma_edges = 1.0 / torch.sqrt(
        torch.abs(r_edge_inv * edge_std[1:4:2] + edge_mean[1:4:2]) + 1e-10
    )

    # Compute h(x_pred) for each graph in batch independently
    n_graphs = int(batch_idx.max().item()) + 1
    idx_from = edge_index[0]
    idx_to = edge_index[1]

    # Global predictions (already computed)
    y_global = torch.cat([V_pred, theta_pred], dim=1)

    total_loss = 0.0
    total_meas = 0

    edge_offset = 0
    node_offset = 0

    for g in range(n_graphs):
        node_mask = (batch_idx == g)
        n_nodes_g = int(node_mask.sum().item())

        # Find edges belonging to this graph: both endpoints in graph
        local_node_ids = torch.where(node_mask)[0]
        min_node = local_node_ids.min().item()
        max_node = local_node_ids.max().item()

        edge_mask = (idx_from >= min_node) & (idx_from <= max_node)
        n_edges_g = int(edge_mask.sum().item())

        # Extract per-graph data
        V_g = V_pred[node_mask, 0]
        theta_g = theta_pred[node_mask, 0]
        ei_g = edge_index[:, edge_mask]
        np_g = node_param[node_mask]
        ep_g = edge_param[edge_mask]

        # Remap edge indices to local [0, n_nodes_g)
        ei_local = ei_g - min_node
        ei_local = ei_local.long()

        # Compute power flows
        _, _, Pf, Qf, _, _, _, _ = get_pflow(
            torch.cat([V_g.unsqueeze(1), theta_g.unsqueeze(1)], dim=1),
            ei_local,
            np_g,
            ep_g,
        )

        # Build h_node: [V, P_inj, Q_inj] — use the prediction directly for V
        # For P, Q injections, scatter flows
        from torch_geometric.utils import scatter
        P_inj = -scatter(torch.zeros_like(Pf), ei_local[0], dim_size=n_nodes_g)  # placeholder
        # Actually compute properly
        P_f = Pf
        Q_f = Qf
        # We also need P_to, Q_to. Let's re-get from get_pflow
        _, _, Pf2, Qf2, Pt, Qt, _, _ = get_pflow(
            torch.cat([V_g.unsqueeze(1), theta_g.unsqueeze(1)], dim=1),
            ei_local,
            np_g,
            ep_g,
        )

        # Bus injections: use same convention as gsp_wls_edge
        P_inj = -scatter(Pt, ei_local[1], dim_size=n_nodes_g) - scatter(Pf2, ei_local[0], dim_size=n_nodes_g)
        Q_inj = -scatter(Qt, ei_local[1], dim_size=n_nodes_g) - scatter(Qf2, ei_local[0], dim_size=n_nodes_g)

        h_node = torch.cat([
            V_g.unsqueeze(1),
            theta_g.unsqueeze(1),
            P_inj.unsqueeze(1),
            Q_inj.unsqueeze(1),
        ], dim=1)  # (n_nodes_g, 4)

        h_edge = torch.cat([Pf2.unsqueeze(1), Qf2.unsqueeze(1)], dim=1)

        # Get corresponding z and sigma
        z_node_g = z_nodes[node_mask]
        sigma_node_g = sigma_nodes[node_mask]
        z_edge_g = edge_z[edge_mask]
        sigma_edge_g = sigma_edges[edge_mask]

        # Mask out zero measurements
        mask_n = (z_node_g != 0).float()
        mask_e = (z_edge_g != 0).float()

        # Compute modal losses
        n_meas_g = int(mask_n.sum().item() + mask_e.sum().item())
        if n_meas_g == 0:
            edge_offset += n_edges_g
            node_offset += n_nodes_g
            continue

        loss_n = 0.0
        if mask_n.sum() > 0:
            for col in range(4):
                if mask_n[:, col].sum() > 0:
                    z_c = z_node_g[:, col][mask_n[:, col] > 0]
                    h_c = h_node[:, col][mask_n[:, col] > 0]
                    s_c = sigma_node_g[:, col][mask_n[:, col] > 0]
                    loss_n += gaussian_kernel_modal_loss(
                        (z_c - h_c) / s_c, bandwidth_z, reduction='sum'
                    )

        loss_e = 0.0
        if mask_e.sum() > 0:
            for col in range(2):
                if mask_e[:, col].sum() > 0:
                    z_c = z_edge_g[:, col][mask_e[:, col] > 0]
                    h_c = h_edge[:, col][mask_e[:, col] > 0]
                    s_c = sigma_edge_g[:, col][mask_e[:, col] > 0]
                    loss_e += gaussian_kernel_modal_loss(
                        (z_c - h_c) / s_c, bandwidth_z, reduction='sum'
                    )

        total_loss += loss_n + loss_e
        total_meas += n_meas_g

    if total_meas == 0:
        return torch.tensor(0.0, device=x_norm.device)

    return total_loss / total_meas


# ---------------------------------------------------------------------------
# Training run
# ---------------------------------------------------------------------------


def train_modal_supervised_ann(
    folder="data/cigre14/",
    meas_v_buses=None,
    meas_pflow_edges=None,
    batch_size=64,
    epochs=600,
    lr=3e-3,
    dim_hid=64,
    num_layers=3,
    dropout=0.1,
    bandwidth_x=2.0,
    bandwidth_z=2.0,
    eta=0.0,                # measurement consistency weight (0 = pure supervised)
    lam_phys=0.0,
    split_coef=0.9,
    device=None,
    verbose=True,
):
    """
    Train a Modal-Supervised ANN for DSSE.

    Args:
        folder:           Path to pickled data.
        meas_v_buses:     Bus indices with V measurements.
        meas_pflow_edges: Edge indices with P_flow measurements.
        batch_size:       Training batch size.
        epochs:           Number of training epochs.
        lr:               Learning rate.
        dim_hid:          Hidden layer size.
        num_layers:       Number of hidden layers.
        dropout:          Dropout rate.
        bandwidth_x:      Kernel bandwidth for state-space modal loss.
        bandwidth_z:      Kernel bandwidth for measurement-space modal loss.
        eta:              Weight of measurement consistency term (0 = off).
        lam_phys:         Physical penalty weight.
        split_coef:       Train/test split ratio.
        device:           Torch device.
        verbose:          Print per-epoch metrics.

    Returns:
        Dict with training history and final metrics.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if meas_v_buses is None:
        meas_v_buses = np.array([0, 1, 12, 7, 11, 14])
    if meas_pflow_edges is None:
        meas_pflow_edges = np.array([0, 10])

    num_nfeat = 8
    num_efeat = 6
    num_nmeas = 4
    num_emeas = 2

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    dataset, x_mean, x_std, pflow_mean, pflow_std = data_from_pickles(
        folder, num_nfeat, num_efeat, num_nmeas, num_emeas,
        meas_v_buses, meas_pflow_edges
    )

    random.shuffle(dataset)
    split_idx = int(split_coef * len(dataset))
    train_dataset = dataset[:split_idx]
    test_dataset = dataset[split_idx:]

    if verbose:
        print(f"Train graphs: {len(train_dataset)}, "
              f"Test graphs: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # Compute state scales from training data
    # ------------------------------------------------------------------
    all_y = []
    for data in train_dataset:
        all_y.append(data.y)
    all_y = torch.cat(all_y, dim=0)
    state_scale = torch.tensor([
        max(all_y[:, 0].std(), 0.005),   # V scale
        max(all_y[:, 1].std(), 0.005),   # θ scale
    ], device=device)

    if verbose:
        print(f"State scales: V={state_scale[0].item():.4f}, "
              f"θ={state_scale[1].item():.4f}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = ANN_DSSE(
        dim_in=num_nfeat,
        dim_hid=dim_hid,
        dim_out=2,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    x_mean_dev = x_mean.to(device)
    x_std_dev = x_std.to(device)
    pflow_mean_dev = pflow_mean.to(device)
    pflow_std_dev = pflow_std.to(device)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history = {
        'train_loss': [], 'train_mse': [], 'train_modal_loss': [],
        'train_meas_loss': [],
        'val_mse': [], 'val_modal_loss': [], 'val_modal_score': [],
        'val_v_rmse': [], 'val_th_rmse': [],
        'val_loading_rmse': [],
    }

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_modal = 0.0
        total_meas = 0.0
        n_batches = 0

        for data in train_loader:
            data = data.to(device)
            x_in = data.x[:, :num_nfeat]
            y_true = data.y

            optimizer.zero_grad()
            out = model(x_in)

            # State-space modal loss
            loss_state = state_space_modal_loss(
                y_true, out, state_scale, bandwidth_x, reduction='mean'
            )

            loss = loss_state

            # Optional measurement consistency term
            loss_meas = torch.tensor(0.0, device=device)
            if eta > 0:
                loss_meas = _measurement_modal_loss_on_batch(
                    x_in, data.edge_attr[:, :num_efeat], out,
                    data.edge_index, data.batch,
                    x_mean_dev, x_std_dev,
                    pflow_mean_dev, pflow_std_dev,
                    data.x[:, num_nfeat:],
                    data.edge_attr[:, num_efeat:],
                    bandwidth_z,
                )
                loss = loss_state + eta * loss_meas

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_modal += loss_state.item()
            total_meas += loss_meas.item() if eta > 0 else 0.0
            total_mse += F.mse_loss(out, y_true).item()
            n_batches += 1

        history['train_loss'].append(total_loss / n_batches)
        history['train_mse'].append(total_mse / n_batches)
        history['train_modal_loss'].append(total_modal / n_batches)
        history['train_meas_loss'].append(total_meas / n_batches if eta > 0 else 0.0)

        # ----------------------------------------------------------------
        # Validation
        # ----------------------------------------------------------------
        model.eval()
        val_mse = 0.0
        val_modal = 0.0
        val_ms = 0.0
        val_v_rmse = 0.0
        val_th_rmse = 0.0
        val_load_rmse = 0.0
        n_val = 0

        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                x_in = data.x[:, :num_nfeat]
                y_true = data.y

                out = model(x_in)

                # Un-normalize for RMSE computation
                out_un = out.clone()
                out_un[:, 0] = out[:, 0] * x_std_dev[0] + x_mean_dev[0]
                out_un[:, 1] *= (1.0 - data.x[:, 9])

                val_mse += F.mse_loss(out_un, y_true).item()
                val_modal += state_space_modal_loss(
                    y_true, out, state_scale, bandwidth_x, reduction='mean'
                ).item()
                val_ms += modal_score(
                    y_true, out, state_scale, bandwidth_x
                ).item()
                val_v_rmse += torch.sqrt(
                    F.mse_loss(out_un[:, :1], y_true[:, :1])
                ).item()
                val_th_rmse += torch.sqrt(
                    F.mse_loss(out_un[:, 1:], y_true[:, 1:])
                ).item()

                # Line loading RMSE (per-graph within batch)
                n_graphs = int(data.batch.max().item()) + 1
                for g in range(n_graphs):
                    node_mask = (data.batch == g)
                    ng_nodes = node_mask.sum().item()
                    min_n = torch.where(node_mask)[0].min().item()
                    max_n = torch.where(node_mask)[0].max().item()
                    edge_mask = (
                        (data.edge_index[0] >= min_n) &
                        (data.edge_index[0] <= max_n)
                    )

                    if edge_mask.sum() == 0:
                        continue

                    ei_g = data.edge_index[:, edge_mask] - min_n
                    np_g = data.x[node_mask, num_nfeat:]
                    ep_g = data.edge_attr[edge_mask, num_efeat:]

                    y_g = y_true[node_mask]
                    out_g = out_un[node_mask]

                    _, _, _, _, _, _, _, _ = get_pflow(
                        y_g, ei_g, np_g, ep_g
                    )
                    true_ll, true_lt, _, _, _, _, _, _ = get_pflow(
                        y_g, ei_g, np_g, ep_g
                    )
                    out_ll, out_lt, _, _, _, _, _, _ = get_pflow(
                        out_g, ei_g, np_g, ep_g
                    )

                    true_load = true_ll + true_lt
                    out_load = out_ll + out_lt
                    if true_load.numel() > 0:
                        val_load_rmse += torch.sqrt(
                            F.mse_loss(out_load, true_load)
                        ).item()

                n_val += 1

        n_val = max(n_val, 1)
        history['val_mse'].append(val_mse / n_val)
        history['val_modal_loss'].append(val_modal / n_val)
        history['val_modal_score'].append(val_ms / n_val)
        history['val_v_rmse'].append(val_v_rmse / n_val)
        history['val_th_rmse'].append(val_th_rmse / n_val)
        history['val_loading_rmse'].append(val_load_rmse / n_val)

        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(
                f"Epoch {epoch:4d}: "
                f"loss={history['train_loss'][-1]:.4e}, "
                f"MSE={history['train_mse'][-1]:.4e}, "
                f"V_RMSE={history['val_v_rmse'][-1]:.6f}, "
                f"θ_RMSE={history['val_th_rmse'][-1]:.6f}, "
                f"MS={history['val_modal_score'][-1]:.4f}"
            )

    # Final metrics
    model.eval()
    final_mse = history['val_mse'][-1]
    final_v_rmse = history['val_v_rmse'][-1]
    final_th_rmse = history['val_th_rmse'][-1]
    final_ms = history['val_modal_score'][-1]
    final_load_rmse = history['val_loading_rmse'][-1]

    return {
        'model': model,
        'history': history,
        'final_mse': final_mse,
        'final_v_rmse': final_v_rmse,
        'final_th_rmse': final_th_rmse,
        'final_modal_score': final_ms,
        'final_loading_rmse': final_load_rmse,
        'state_scale': state_scale,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=" * 70)
    print("SMOKE TEST — Modal-Supervised ANN")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Quick training run on CIGRE 14
    result = train_modal_supervised_ann(
        folder="data/cigre14/",
        epochs=20,           # very short for smoke test
        batch_size=64,
        dim_hid=32,
        num_layers=2,
        bandwidth_x=2.0,
        eta=0.0,             # pure supervised modal ANN
        device=device,
        verbose=True,
    )

    h = result['history']

    print("\n--- Smoke test checks ---")

    # Check 1: Loss decreases
    loss_first = h['train_loss'][0]
    loss_last = h['train_loss'][-1]
    print(f"Train loss: {loss_first:.6f} -> {loss_last:.6f}")
    assert loss_last < loss_first * 0.95, (
        f"Loss should decrease: {loss_first:.4f} -> {loss_last:.4f}"
    )
    print("  [PASS] Loss decreased during training")

    # Check 2: Modal score increases
    ms_first = h['val_modal_score'][0]
    ms_last = h['val_modal_score'][-1]
    print(f"ModalScore: {ms_first:.4f} -> {ms_last:.4f}")
    assert ms_last > ms_first * 0.95, (
        f"ModalScore should not degrade: {ms_first:.4f} -> {ms_last:.4f}"
    )
    print("  [PASS] ModalScore stable/improving")

    # Check 3: V RMSE is reasonable
    v_rmse = h['val_v_rmse'][-1]
    print(f"Final V RMSE: {v_rmse:.6f}")
    assert v_rmse < 0.5, f"V RMSE too high: {v_rmse:.4f}"
    print("  [PASS] V RMSE is reasonable")

    # Check 4: Gradients are valid (no NaN)
    for name, val in [
        ('loss', loss_last), ('V_RMSE', v_rmse),
        ('ModalScore', ms_last)
    ]:
        assert not np.isnan(val) and not np.isinf(val), (
            f"{name} is NaN or Inf: {val}"
        )
    print("  [PASS] No NaN/Inf in metrics")

    # Check 5: All expected metrics present
    expected = ['train_loss', 'train_mse', 'train_modal_loss',
                'val_mse', 'val_modal_loss', 'val_modal_score',
                'val_v_rmse', 'val_th_rmse', 'val_loading_rmse']
    for key in expected:
        assert key in h and len(h[key]) > 0, f"Missing metric: {key}"
    print(f"  [PASS] All {len(expected)} metrics logged")

    print(f"\nFinal metrics:")
    print(f"  V RMSE:       {result['final_v_rmse']:.6f}")
    print(f"  θ RMSE:       {result['final_th_rmse']:.6f}")
    print(f"  Loading RMSE: {result['final_loading_rmse']:.6f}")
    print(f"  ModalScore:   {result['final_modal_score']:.6f}")
    print(f"  MSE:          {result['final_mse']:.6f}")

    print()
    print("=" * 70)
    print("SMOKE TEST PASSED")
    print("=" * 70)
