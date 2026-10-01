"""
Modal-Supervised DSS² training.

Uses the same GNN architecture (GAT / SkipPFN) as the original DSS², but
replaces the squared-error loss with modal kernel losses:

    L = L_state_modal + eta * L_meas_modal + lambda_phys * penalty(x)

where:
    L_state_modal = mean_j [1 - exp(-0.5 * r_j^2 / b_state^2)]
        with  r_j = (x_true_j - x_pred_j) / state_scale_j

    L_meas_modal  = mean_k [1 - exp(-0.5 * r_k^2 / b_meas^2)]
        with  r_k = (z_k - h_k(x_pred)) / sigma_k

Bandwidth annealing: start with a large bandwidth (MSE-like behaviour) and
gradually reduce it during training to increase robustness.
"""

import torch
import torch.nn.functional as F
import numpy as np
import random

# Work around transformers/numpy metadata bug in this environment
import importlib.metadata as _im
_orig_v = _im.version
_im.version = lambda pkg: np.__version__ if pkg == 'numpy' else _orig_v(pkg)

from torch_geometric.loader import DataLoader
from torchmetrics.regression import MeanAbsoluteError

from networks import GAT_DSSE, SkipPFN
from data import data_from_pickles, get_pflow
from loadsampling import progressBar

from modal_losses import (
    gaussian_kernel_modal_loss,
    kernel_value,
    state_space_modal_loss,
    modal_score,
    hit_rate,
)


# ---------------------------------------------------------------------------
# Bandwidth annealing
# ---------------------------------------------------------------------------


class BandwidthSchedule:
    """Exponential decay bandwidth schedule.

        b(epoch) = b_end + (b_start - b_end) * exp(-epoch / tau)
    """

    def __init__(self, b_start=10.0, b_end=2.0, tau=100.0):
        self.b_start = b_start
        self.b_end = b_end
        self.tau = tau

    def get(self, epoch):
        return self.b_end + (self.b_start - self.b_end) * np.exp(
            -epoch / self.tau
        )


# ---------------------------------------------------------------------------
# Measurement modal loss — batched GNN version
# ---------------------------------------------------------------------------


def _meas_modal_loss(
    x_norm, edge_attr_norm, out_norm, edge_index,
    x_mean, x_std, edge_mean, edge_std,
    node_param, edge_param, bandwidth,
):
    """
    Compute measurement-space modal loss on a (possibly batched) graph.

    Standardized residual:  r_k = (z_k - h_k(x_pred)) / sigma_k

    Args:
        x_norm:           Normalized node features  (N, num_nfeat).
        edge_attr_norm:   Normalized edge features  (E, num_efeat).
        out_norm:         Normalized predictions     (N, 2).
        edge_index:       (2, E).
        x_mean, x_std:    Node measurement stats     (num_nfeat,).
        edge_mean, edge_std: Edge measurement stats  (num_efeat,).
        node_param:       Raw node params            (N, 3).
        edge_param:       Raw edge params            (E, 7).
        bandwidth:        Kernel bandwidth (scalar).

    Returns:
        Scalar measurement modal loss (mean over all measurements).
    """
    num_nfeat = 8
    num_efeat = 6

    # --- Un-normalise node measurements ---
    z_nodes = x_norm[:, ::2] * x_std[::2] + x_mean[::2]          # (N, 4)
    r_inv = x_norm[:, 1::2]                                       # (N, 4)
    sigma_nodes = 1.0 / torch.sqrt(
        torch.abs(r_inv * x_std[1::2] + x_mean[1::2]) + 1e-10
    )

    # --- Un-normalise edge measurements ---
    z_edges = edge_attr_norm[:, :4:2] * edge_std[:4:2] + edge_mean[:4:2]  # (E, 2)
    r_edge_inv = edge_attr_norm[:, 1:4:2]
    sigma_edges = 1.0 / torch.sqrt(
        torch.abs(r_edge_inv * edge_std[1:4:2] + edge_mean[1:4:2]) + 1e-10
    )

    # --- Un-normalise predictions ---
    V_pred = out_norm[:, 0:1] * x_std[0:1] + x_mean[0:1]
    theta_pred = out_norm[:, 1:2].clone()
    theta_pred *= (1.0 - node_param[:, 1:2])  # enforce slack theta = 0

    y_global = torch.cat([V_pred, theta_pred], dim=1)  # (N, 2)

    # --- Compute h(x) = power flows ---
    _, _, Pf, Qf, Pt, Qt, _, _ = get_pflow(
        y_global, edge_index, node_param, edge_param
    )

    # --- Node injection measurements: scatter flows ---
    from torch_geometric.utils import scatter
    idx_f = edge_index[0]
    idx_t = edge_index[1]
    n_nodes = x_norm.shape[0]

    P_inj = -scatter(Pt, idx_t, dim_size=n_nodes) - scatter(
        Pf, idx_f, dim_size=n_nodes
    )
    Q_inj = -scatter(Qt, idx_t, dim_size=n_nodes) - scatter(
        Qf, idx_f, dim_size=n_nodes
    )

    h_node = torch.cat([
        V_pred,
        theta_pred,
        P_inj.unsqueeze(1),
        Q_inj.unsqueeze(1),
    ], dim=1)  # (N, 4)

    h_edge = torch.cat([Pf.unsqueeze(1), Qf.unsqueeze(1)], dim=1)  # (E, 2)

    # --- Mask for present measurements ---
    mask_n = (z_nodes != 0).float()
    mask_e = (z_edges != 0).float()

    # --- Per-measurement modal loss ---
    total_loss = 0.0
    n_total = 0

    for col in range(4):
        m = mask_n[:, col]
        if m.sum() == 0:
            continue
        z_c = z_nodes[:, col][m > 0]
        h_c = h_node[:, col][m > 0]
        s_c = sigma_nodes[:, col][m > 0]
        r = (z_c - h_c) / (s_c + 1e-10)
        total_loss += gaussian_kernel_modal_loss(
            r, bandwidth, reduction='sum'
        )
        n_total += int(m.sum().item())

    for col in range(2):
        m = mask_e[:, col]
        if m.sum() == 0:
            continue
        z_c = z_edges[:, col][m > 0]
        h_c = h_edge[:, col][m > 0]
        s_c = sigma_edges[:, col][m > 0]
        r = (z_c - h_c) / (s_c + 1e-10)
        total_loss += gaussian_kernel_modal_loss(
            r, bandwidth, reduction='sum'
        )
        n_total += int(m.sum().item())

    if n_total == 0:
        return torch.tensor(0.0, device=x_norm.device)

    return total_loss / n_total


# ---------------------------------------------------------------------------
# Physical penalty
# ---------------------------------------------------------------------------


def _physical_penalty(out_norm, edge_index, node_param, edge_param,
                      x_std, x_mean):
    """
    Voltage bounds, angle-difference, and line-loading penalty.

    Same formulation as data.gsp_wls_edge.
    """
    V_pred = out_norm[:, 0:1] * x_std[0:1] + x_mean[0:1]
    theta_pred = out_norm[:, 1:2].clone()
    theta_pred *= (1.0 - node_param[:, 1:2])

    y_global = torch.cat([V_pred, theta_pred], dim=1)
    loading_lines, loading_trafos, _, _, _, _, _, _ = get_pflow(
        y_global, edge_index, node_param, edge_param
    )
    loading = loading_lines + loading_trafos

    idx_f = edge_index[0]
    idx_t = edge_index[1]
    theta_ij = torch.abs(
        torch.gather(theta_pred[:, 0], 0, idx_f)
        - torch.gather(theta_pred[:, 0], 0, idx_t)
    )

    J_v = torch.mean(
        torch.relu(V_pred - 1.1) + torch.relu(0.9 - V_pred)
    ) ** 2
    J_theta = torch.mean(torch.relu(theta_ij - 0.5)) ** 2
    J_loading = torch.mean(torch.relu(loading - 1.5)) ** 2

    return J_v + J_theta + J_loading


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------


def train_modal_supervised_dss2(
    folder="data/cigre14/",
    meas_v_buses=None,
    meas_pflow_edges=None,
    model_name="gat",
    batch_size=64,
    epochs=600,
    lr=3e-3,
    dim_hid=32,
    gnn_layers=8,
    heads=1,
    K=2,
    dropout_rate=0.3,
    L=5,
    # --- Modal loss config ---
    b_state_start=10.0,
    b_state_end=2.0,
    b_state_tau=100.0,
    b_meas_start=10.0,
    b_meas_end=2.0,
    b_meas_tau=100.0,
    use_annealing=True,
    eta=1.0,               # measurement consistency weight
    lambda_phys=1e2,       # physical penalty weight
    # --- Misc ---
    split_coef=0.9,
    device=None,
    verbose=True,
):
    """
    Train a Modal-Supervised DSS² GNN.

    Loss = L_state_modal + eta * L_meas_modal + lambda_phys * penalty(x)

    Args:
        folder:          Path to pickled data.
        meas_v_buses:    Bus indices with V measurements.
        meas_pflow_edges: Edge indices with P_flow measurements.
        model_name:      'gat' or 'pfn'.
        batch_size, epochs, lr:  Standard training hyperparameters.
        dim_hid, gnn_layers, heads, K, dropout_rate, L:  GNN architecture.
        b_state_*:       Bandwidth schedule for state-space modal loss.
        b_meas_*:        Bandwidth schedule for measurement-space modal loss.
        use_annealing:   If False, use b_state_end / b_meas_end directly.
        eta:             Weight of measurement modal loss.
        lambda_phys:     Weight of physical penalty.
        split_coef:      Train/test split.
        device:          Torch device.
        verbose:         Print per-epoch metrics.

    Returns:
        Dict with model, history, and final metrics.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_nfeat = 8
    num_efeat = 6
    num_nmeas = 4
    num_emeas = 2

    if meas_v_buses is None:
        meas_v_buses = np.array([0, 1, 12, 7, 11, 14])
    if meas_pflow_edges is None:
        meas_pflow_edges = np.array([0, 10])

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dataset, x_mean, x_std, pflow_mean, pflow_std = data_from_pickles(
        folder, num_nfeat, num_efeat, num_nmeas, num_emeas,
        meas_v_buses, meas_pflow_edges
    )

    random.shuffle(dataset)
    n_train = int(split_coef * len(dataset))
    train_dataset = dataset[:n_train]
    test_dataset = dataset[n_train:]

    if verbose:
        print(f"Train graphs: {len(train_dataset)}, "
              f"Test graphs: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # State scales from training data
    # ------------------------------------------------------------------
    all_y = torch.cat([d.y for d in train_dataset], dim=0)
    state_scale = torch.tensor([
        max(float(all_y[:, 0].std()), 0.005),
        max(float(all_y[:, 1].std()), 0.005),
    ], device=device)

    if verbose:
        print(f"State scales: V={state_scale[0].item():.4f}, "
              f"θ={state_scale[1].item():.4f}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if model_name == 'gat':
        model = GAT_DSSE(
            dim_feat=num_nfeat, dim_dense=dim_hid, dim_out=2,
            heads=heads, num_layers=gnn_layers, edge_dim=num_efeat,
        ).to(device)
    elif model_name == 'pfn':
        model = SkipPFN(
            dim_featn=num_nfeat, dim_feate=num_efeat, dim_out=2,
            dim_hid=dim_hid, n_gnn_layers=gnn_layers, K=K,
            dropout_rate=dropout_rate, L=L,
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)

    x_mean_d = x_mean.to(device)
    x_std_d = x_std.to(device)
    em_d = pflow_mean.to(device)
    es_d = pflow_std.to(device)

    # Bandwidth schedules
    b_state_sched = BandwidthSchedule(b_state_start, b_state_end, b_state_tau)
    b_meas_sched = BandwidthSchedule(b_meas_start, b_meas_end, b_meas_tau)

    # ------------------------------------------------------------------
    # History containers
    # ------------------------------------------------------------------
    history = {
        'train_loss': [], 'train_state_loss': [], 'train_meas_loss': [],
        'train_phys_pen': [],
        'val_v_rmse': [], 'val_th_rmse': [],
        'val_loading_rmse': [], 'val_loading_trafos_rmse': [],
        'val_modal_score': [], 'val_hit_rate': [],
        'b_state': [], 'b_meas': [],
    }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(epochs):
        # Bandwidths for this epoch
        if use_annealing:
            b_s = b_state_sched.get(epoch)
            b_m = b_meas_sched.get(epoch)
        else:
            b_s = b_state_end
            b_m = b_meas_end

        history['b_state'].append(b_s)
        history['b_meas'].append(b_m)

        # ---- Train step ----
        model.train()
        epoch_loss = 0.0
        epoch_state_l = 0.0
        epoch_meas_l = 0.0
        epoch_phys = 0.0
        n_batches = 0

        for data in train_loader:
            data = data.to(device)
            x_in = data.x[:, :num_nfeat]
            e_in = data.edge_attr[:, :num_efeat]
            y_true = data.y
            node_p = data.x[:, num_nfeat:]
            edge_p = data.edge_attr[:, num_efeat:]

            optimizer.zero_grad()
            out = model(x_in, data.edge_index, e_in)

            # State-space modal loss
            loss_state = state_space_modal_loss(
                y_true, out, state_scale, b_s, reduction='mean'
            )

            # Measurement-space modal loss
            loss_meas = _meas_modal_loss(
                x_in, e_in, out, data.edge_index,
                x_mean_d, x_std_d, em_d, es_d,
                node_p, edge_p, b_m,
            )

            # Physical penalty
            phys_pen = _physical_penalty(
                out, data.edge_index, node_p, edge_p, x_std_d, x_mean_d
            )

            loss = loss_state + eta * loss_meas + lambda_phys * phys_pen

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_state_l += loss_state.item()
            epoch_meas_l += loss_meas.item()
            epoch_phys += phys_pen.item()
            n_batches += 1

        history['train_loss'].append(epoch_loss / n_batches)
        history['train_state_loss'].append(epoch_state_l / n_batches)
        history['train_meas_loss'].append(epoch_meas_l / n_batches)
        history['train_phys_pen'].append(epoch_phys / n_batches)

        # ---- Validation ----
        model.eval()
        val_v = 0.0; val_th = 0.0; val_load = 0.0
        val_load_trafos = 0.0; val_ms = 0.0; val_hr = 0.0
        n_val = 0

        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                x_in = data.x[:, :num_nfeat]
                e_in = data.edge_attr[:, :num_efeat]
                y_true = data.y
                node_p = data.x[:, num_nfeat:]
                edge_p = data.edge_attr[:, num_efeat:]

                out = model(x_in, data.edge_index, e_in)

                # Un-normalise predictions
                out_un = out.clone()
                out_un[:, 0] = out[:, 0] * x_std_d[0] + x_mean_d[0]
                out_un[:, 1] *= (1.0 - node_p[:, 1])

                # RMSE
                val_v += torch.sqrt(
                    F.mse_loss(out_un[:, :1], y_true[:, :1])
                ).item()
                val_th += torch.sqrt(
                    F.mse_loss(out_un[:, 1:], y_true[:, 1:])
                ).item()

                # ModalScore & HitRate
                val_ms += modal_score(
                    y_true, out_un, state_scale, b_s
                ).item()
                val_hr += hit_rate(
                    y_true, out_un, state_scale, delta=3.0
                ).item()

                # Line loading RMSE
                true_ll, true_lt, _, _, _, _, _, _ = get_pflow(
                    y_true, data.edge_index, node_p, edge_p
                )
                out_ll, out_lt, _, _, _, _, _, _ = get_pflow(
                    out_un, data.edge_index, node_p, edge_p
                )

                mask_ll = true_ll.nonzero(as_tuple=True)[0]
                mask_lt = true_lt.nonzero(as_tuple=True)[0]

                if mask_ll.numel() > 0:
                    val_load += torch.sqrt(
                        F.mse_loss(out_ll[mask_ll], true_ll[mask_ll])
                    ).item()
                if mask_lt.numel() > 0:
                    val_load_trafos += torch.sqrt(
                        F.mse_loss(out_lt[mask_lt], true_lt[mask_lt])
                    ).item()

                n_val += 1

        n_val = max(n_val, 1)
        history['val_v_rmse'].append(val_v / n_val)
        history['val_th_rmse'].append(val_th / n_val)
        history['val_loading_rmse'].append(val_load / max(n_val, 1))
        history['val_loading_trafos_rmse'].append(
            val_load_trafos / max(n_val, 1)
        )
        history['val_modal_score'].append(val_ms / n_val)
        history['val_hit_rate'].append(val_hr / n_val)

        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(
                f"Epoch {epoch:4d}: "
                f"loss={history['train_loss'][-1]:.4e} "
                f"st={history['train_state_loss'][-1]:.4e} "
                f"ms={history['train_meas_loss'][-1]:.4e} "
                f"ph={history['train_phys_pen'][-1]:.4e} | "
                f"b_s={b_s:.2f} b_m={b_m:.2f} | "
                f"V={history['val_v_rmse'][-1]:.5f} "
                f"th={history['val_th_rmse'][-1]:.5f} "
                f"MS={history['val_modal_score'][-1]:.4f}"
            )

    return {
        'model': model,
        'history': history,
        'final_v_rmse': history['val_v_rmse'][-1],
        'final_th_rmse': history['val_th_rmse'][-1],
        'final_loading_rmse': history['val_loading_rmse'][-1],
        'final_loading_trafos_rmse': history['val_loading_trafos_rmse'][-1],
        'final_modal_score': history['val_modal_score'][-1],
        'final_hit_rate': history['val_hit_rate'][-1],
        'state_scale': state_scale,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=" * 70)
    print("SMOKE TEST — Modal-Supervised DSS²")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    result = train_modal_supervised_dss2(
        folder="data/cigre14/",
        model_name="gat",
        epochs=30,
        batch_size=32,
        dim_hid=32,
        gnn_layers=3,
        heads=1,
        # Quick bandwidth annealing
        b_state_start=8.0,
        b_state_end=2.0,
        b_state_tau=10.0,
        b_meas_start=8.0,
        b_meas_end=2.0,
        b_meas_tau=10.0,
        use_annealing=True,
        eta=0.5,
        lambda_phys=1.0,
        device=device,
        verbose=True,
    )

    h = result['history']

    print("\n--- Smoke test checks ---")

    # 1. Loss decreases
    l0 = h['train_loss'][0]
    lf = h['train_loss'][-1]
    print(f"Train loss: {l0:.6f} -> {lf:.6f}")
    assert lf < l0 * 0.95, f"Loss should decrease: {l0:.4f} -> {lf:.4f}"
    print("  [PASS] Loss decreased")

    # 2. All loss components logged
    for key in ['train_state_loss', 'train_meas_loss', 'train_phys_pen']:
        assert key in h and len(h[key]) > 0, f"Missing: {key}"
    print("  [PASS] All loss components logged separately")

    # 3. Bandwidth annealing recorded
    assert len(h['b_state']) == 30 and len(h['b_meas']) == 30
    assert h['b_state'][0] > h['b_state'][-1], "State bandwidth should decrease"
    assert h['b_meas'][0] > h['b_meas'][-1], "Meas bandwidth should decrease"
    print(f"  [PASS] Bandwidth annealed: "
          f"b_s {h['b_state'][0]:.2f}->{h['b_state'][-1]:.2f}, "
          f"b_m {h['b_meas'][0]:.2f}->{h['b_meas'][-1]:.2f}")

    # 4. No NaN/Inf
    for k in ['train_loss', 'val_v_rmse', 'val_modal_score']:
        assert not np.isnan(h[k][-1]) and not np.isinf(h[k][-1]), \
            f"{k} is NaN/Inf"
    print("  [PASS] No NaN/Inf in metrics")

    # 5. Metrics at fixed reference bandwidth (avoid annealing confound)
    ms0_fixed = modal_score(
        torch.zeros(1, 2), torch.zeros(1, 2),
        result['state_scale'], 2.0
    ).item()  # should be 1.0
    print(f"  Fixed-bw ModalScore sanity: {ms0_fixed:.4f} (expected 1.0)")
    assert ms0_fixed > 0.999, "ModalScore at zero residual should be ~1"
    print("  [PASS] ModalScore at zero residual is 1.0")

    # 6. V and θ RMSE are physically reasonable
    assert result['final_v_rmse'] < 0.2, f"V RMSE too high"
    assert result['final_th_rmse'] < 0.1, f"θ RMSE too high"
    print(f"  [PASS] Physical metrics reasonable "
          f"(V={result['final_v_rmse']:.4f}, θ={result['final_th_rmse']:.4f})")

    # 7. All expected metrics
    expected = [
        'train_loss', 'train_state_loss', 'train_meas_loss', 'train_phys_pen',
        'val_v_rmse', 'val_th_rmse', 'val_loading_rmse',
        'val_loading_trafos_rmse', 'val_modal_score', 'val_hit_rate',
        'b_state', 'b_meas',
    ]
    for k in expected:
        assert k in h and len(h[k]) > 0, f"Missing metric: {k}"
    print(f"  [PASS] All {len(expected)} metrics logged")

    print(f"\nFinal metrics:")
    print(f"  V RMSE:          {result['final_v_rmse']:.6f}")
    print(f"  θ RMSE:          {result['final_th_rmse']:.6f}")
    print(f"  Loading RMSE:    {result['final_loading_rmse']:.6f}")
    print(f"  Trafos RMSE:     {result['final_loading_trafos_rmse']:.6f}")
    print(f"  ModalScore:      {result['final_modal_score']:.6f}")
    print(f"  HitRate:         {result['final_hit_rate']:.6f}")

    print()
    print("=" * 70)
    print("SMOKE TEST PASSED")
    print("=" * 70)
