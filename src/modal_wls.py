"""
Modal-WLS solver for distribution system state estimation.

Implements both standard WLS and Modal-WLS (iteratively reweighted) using
SciPy's least_squares optimizer.  The Modal-WLS replaces the squared residual
with the Gaussian kernel modal loss, downweighting outlier measurements.

    L_MWLS(x) = sum_k [1 - exp(-r_k(x)^2 / (2 b^2))] + lambda * penalty(x)

where r_k = (z_k - h_k(x)) / sigma_k.

The IRLS scheme at iteration t uses effective weights:

    w_k^(t) = exp(-r_k^(t)^2 / (2 b^2)) / sigma_k^2

and solves a reweighted least-squares subproblem.
"""

import numpy as np
from scipy.optimize import least_squares
import time
import pickle
import sys

from modal_losses import (
    gaussian_kernel_modal_loss_np,
    kernel_value_np,
    measurement_space_modal_loss_np,
    modal_score_np,
    hit_rate_np,
)


# ---------------------------------------------------------------------------
# NumPy port of the power-flow measurement function h(x)
# ---------------------------------------------------------------------------


def get_pflow_np(V, theta, edge_index, node_param, edge_param):
    """
    NumPy port of data.get_pflow.

    Computes power flows and line loadings given state (V, theta).

    Args:
        V:           Voltage magnitudes (n_nodes,), per-unit.
        theta:       Voltage angles (n_nodes,), radians.
        edge_index:  (2, n_edges) — from/to bus indices.
        node_param:  (n_nodes, 3) — [vn_kv, bool_slack, bool_zero_inj].
        edge_param:  (n_edges, 7) — [G, B, Gs, Bs, closed, phase_shift,
                     imax_or_sn].

    Returns:
        loading_lines, loading_trafos,
        P_ij_from, Q_ij_from, P_ij_to, Q_ij_to,
        I_ij_from, I_ij_to
    """
    V_n = node_param[:, 0]
    V_hv = V_n.max()
    V_lv = V_n.min()
    ratio = V_hv / V_lv

    idx_from = edge_index[0].astype(int)
    idx_to = edge_index[1].astype(int)

    Y1 = edge_param[:, 0]  # G
    Y2 = edge_param[:, 1]  # B
    Ys1 = edge_param[:, 2]  # Gs
    Ys2 = edge_param[:, 3]  # Bs
    shift = edge_param[:, 5]  # phase_shift
    trafo_pos = np.ceil(shift).copy()
    imax_or_sn = edge_param[:, 6]

    V_i = V[idx_from]
    Th_i = theta[idx_from]
    V_j = V[idx_to]
    Th_j = theta[idx_to]

    cos_term = np.cos(Th_i - Th_j - shift)
    sin_term = np.sin(Th_i - Th_j - shift)

    Vlv2 = V_lv ** 2

    P_ij_from = (
        -V_i * V_j * (Y1 * cos_term + Y2 * sin_term)
        + (Y1 + Ys1 / 2.0) * V_i ** 2
    ) * Vlv2

    Q_ij_from = (
        V_i * V_j * (-Y1 * sin_term + Y2 * cos_term)
        - (Y2 + Ys2 / 2.0) * V_i ** 2
    ) * Vlv2

    P_ij_to = (
        -V_i * V_j * (Y1 * cos_term - Y2 * sin_term)
        + (Y1 + Ys1 / 2.0) * V_j ** 2
    ) * Vlv2

    Q_ij_to = (
        V_i * V_j * (Y1 * sin_term + Y2 * cos_term)
        - (Y2 + Ys2 / 2.0) * V_j ** 2
    ) * Vlv2

    I_ij_from = np.abs(P_ij_from - 1j * Q_ij_from) / (
        V_i * V_lv * np.sqrt(3.0) + 1e-10
    )
    I_ij_from = I_ij_from / (1.0 - trafo_pos * (1.0 - ratio) + 1e-10)

    I_ij_to = np.abs(P_ij_to - 1j * Q_ij_to) / (
        V_j * V_lv * np.sqrt(3.0) + 1e-10
    )

    loading_lines = (
        (1.0 - trafo_pos) * np.maximum(I_ij_from, I_ij_to)
    ) / (imax_or_sn + 1e-10)
    loading_trafo = (
        trafo_pos * np.maximum(I_ij_from * V_hv, I_ij_to * V_lv)
    ) / (imax_or_sn + 1e-10)

    return (
        loading_lines,
        loading_trafo,
        P_ij_from,
        Q_ij_from,
        P_ij_to,
        Q_ij_to,
        I_ij_from,
        I_ij_to,
    )


def compute_injections(P_from, Q_from, P_to, Q_to, edge_index, n_nodes):
    """
    Compute bus power injections from edge flows.

    Uses the same convention as data.gsp_wls_edge:
        p_i = -scatter(p_to, idx_to) - scatter(p_from, idx_from)

    Returns (P_inj, Q_inj) each of shape (n_nodes,).
    """
    idx_from = edge_index[0].astype(int)
    idx_to = edge_index[1].astype(int)

    P_inj = np.zeros(n_nodes)
    Q_inj = np.zeros(n_nodes)

    for k in range(len(idx_from)):
        f = idx_from[k]
        t = idx_to[k]
        P_inj[f] -= P_from[k]
        P_inj[t] -= P_to[k]
        Q_inj[f] -= Q_from[k]
        Q_inj[t] -= Q_to[k]

    return P_inj, Q_inj


def measurement_function(x, edge_index, node_param, edge_param, meas_structure):
    """
    Compute the full measurement vector h(x) from state x.

    Args:
        x:               State vector [V_0,...,V_{n-1}, θ_0,...,θ_{n-1}].
        edge_index:      (2, n_edges).
        node_param:      (n_nodes, 3).
        edge_param:      (n_edges, 7).
        meas_structure:  Dict describing which measurements to include.
                         See build_measurement_structure().

    Returns:
        h: (n_meas,) measurement vector.
    """
    n_nodes = node_param.shape[0]
    V = x[:n_nodes]
    theta = x[n_nodes:]

    # Power flows on all edges
    _, _, P_from, Q_from, P_to, Q_to, _, _ = get_pflow_np(
        V, theta, edge_index, node_param, edge_param
    )

    # Bus injections
    P_inj, Q_inj = compute_injections(
        P_from, Q_from, P_to, Q_to, edge_index, n_nodes
    )

    h_parts = []

    if meas_structure["use_v"]:
        h_parts.append(V[meas_structure["v_buses"]])
    if meas_structure["use_p_inj"]:
        h_parts.append(P_inj[meas_structure["p_buses"]])
    if meas_structure["use_q_inj"]:
        h_parts.append(Q_inj[meas_structure["q_buses"]])
    if meas_structure["use_p_flow"]:
        h_parts.append(P_from[meas_structure["pf_edges"]])
    if meas_structure["use_q_flow"]:
        h_parts.append(Q_from[meas_structure["qf_edges"]])

    return np.concatenate(h_parts)


def physical_penalty_np(x, edge_index, node_param, edge_param):
    """
    Physical penalty terms (voltage bounds, angle difference, line loading).

    Returns scalar penalty value.
    """
    n_nodes = node_param.shape[0]
    V = x[:n_nodes]
    theta = x[n_nodes:]

    idx_from = edge_index[0].astype(int)
    idx_to = edge_index[1].astype(int)

    loading_lines, loading_trafos, _, _, _, _, _, _ = get_pflow_np(
        V, theta, edge_index, node_param, edge_param
    )

    # Voltage bounds: penalize V outside [0.9, 1.1]
    J_v = np.mean(np.maximum(V - 1.1, 0) + np.maximum(0.9 - V, 0)) ** 2

    # Angle difference: penalize |θ_ij| > 0.5 rad
    theta_ij = np.abs(theta[idx_from] - theta[idx_to])
    J_theta = np.mean(np.maximum(theta_ij - 0.5, 0)) ** 2

    # Loading: penalize loading > 150%
    loading = loading_lines + loading_trafos
    J_loading = np.mean(np.maximum(loading - 1.5, 0)) ** 2

    return J_v + J_theta + J_loading


# ---------------------------------------------------------------------------
# Measurement structure builder
# ---------------------------------------------------------------------------


def build_measurement_structure(node_data, edge_data, noise_params, meas_v_buses):
    """
    Build a measurement structure dict from a single sample.

    Args:
        node_data:  DataFrame with columns [vn_kv, bool_slack, bool_zero_inj,
                    vm_pu, va_rad, p_mw, q_mvar].
        edge_data:  DataFrame with columns [from_bus, to_bus, G, B, Gs, Bs,
                    closed line, phase shift, imax or sn, p_from_mw,
                    q_from_mvar, ...].  Should be pre-filtered to closed edges.
        noise_params: DataFrame with [v_noise, PM_NOISE, ...].
        meas_v_buses: List of bus indices with V measurements.

    Returns:
        meas:       Dict with measurement vector z, sigma, and structure info.
        node_param: (n_nodes, 3) array.
        edge_param: (n_edges, 7) array.
        edge_index: (2, n_edges) integer array.
    """
    n_nodes = node_data.shape[0]
    n_edges = edge_data.shape[0]

    v_noise = float(noise_params["v_noise"].values[0])
    PM_NOISE = float(noise_params["PM_NOISE"].values[0])
    p_noise = float(noise_params["p_noise"].values[0])

    # Node / edge parameters
    node_param = node_data[["vn_kv", "bool_slack", "bool_zero_inj"]].values.astype(
        np.float64
    )
    edge_param_cols = ["G", "B", "Gs", "Bs", "closed line", "phase shift", "imax or sn"]
    edge_param = edge_data[edge_param_cols].values.astype(np.float64)
    edge_index = edge_data[["from_bus", "to_bus"]].values.astype(int).T  # (2, n_edges)

    # Build measurement mask (matching data.py convention)
    meas_mask = np.ones((n_nodes, 4)) * [0.0, 0.0, 1.0, 1.0]
    for j in meas_v_buses:
        meas_mask[j, 0] = 1.0

    z_parts = []
    sigma_parts = []

    # V measurements
    v_buses = np.where(meas_mask[:, 0] == 1.0)[0]
    z_parts.append(node_data["vm_pu"].values[v_buses])
    sigma_parts.append(np.abs(node_data["vm_pu"].values[v_buses]) * v_noise + 1e-6)

    # P injection pseudomeasurements
    p_buses = np.where(meas_mask[:, 2] == 1.0)[0]
    z_parts.append(node_data["p_mw"].values[p_buses])
    sigma_parts.append(np.abs(node_data["p_mw"].values[p_buses]) * PM_NOISE + 1e-6)

    # Q injection pseudomeasurements
    q_buses = np.where(meas_mask[:, 3] == 1.0)[0]
    z_parts.append(node_data["q_mvar"].values[q_buses])
    sigma_parts.append(np.abs(node_data["q_mvar"].values[q_buses]) * PM_NOISE + 1e-6)

    # P flow measurements (all closed edges)
    pf_edges = np.arange(n_edges)
    z_parts.append(edge_data["p_from_mw"].values[pf_edges])
    sigma_parts.append(
        np.abs(edge_data["p_from_mw"].values[pf_edges]) * p_noise + 1e-5
    )

    # Q flow measurements (all closed edges)
    qf_edges = np.arange(n_edges)
    z_parts.append(edge_data["q_from_mvar"].values[qf_edges])
    sigma_parts.append(
        np.abs(edge_data["q_from_mvar"].values[qf_edges]) * p_noise + 1e-5
    )

    meas = {
        "z": np.concatenate(z_parts).astype(np.float64),
        "sigma": np.concatenate(sigma_parts).astype(np.float64),
        "v_buses": v_buses.astype(int),
        "p_buses": p_buses.astype(int),
        "q_buses": q_buses.astype(int),
        "pf_edges": pf_edges.astype(int),
        "qf_edges": qf_edges.astype(int),
        "use_v": True,
        "use_p_inj": True,
        "use_q_inj": True,
        "use_p_flow": True,
        "use_q_flow": True,
    }

    return meas, node_param, edge_param, edge_index


# ---------------------------------------------------------------------------
# Standard WLS solver
# ---------------------------------------------------------------------------

def _wls_residual(x, z, sigma, edge_index, node_param, edge_param, meas_structure):
    """Residual function for scipy.optimize.least_squares (standard WLS)."""
    h_x = measurement_function(x, edge_index, node_param, edge_param, meas_structure)
    return (z - h_x) / sigma


def solve_wls(
    meas,
    node_param,
    edge_param,
    edge_index,
    x0=None,
    lam_phys=0.0,
    method="trf",
    max_nfev=500,
    xtol=1e-8,
    gtol=1e-8,
    verbose=0,
):
    """
    Standard WLS state estimation via SciPy least_squares.

    Args:
        meas:        Measurement dict from build_measurement_structure().
        node_param:  (n_nodes, 3) array.
        edge_param:  (n_edges, 7) array.
        edge_index:  (2, n_edges) array.
        x0:          Initial state (flat start if None).
        lam_phys:    Physical penalty weight (default 0 = pure WLS).
        method:      'trf', 'dogbox', or 'lm'.
        max_nfev:    Max function evaluations.
        xtol, gtol:  Convergence tolerances.
        verbose:     least_squares verbosity.

    Returns:
        Dict with keys: x_est, V_est, theta_est, n_fev, n_iter, converged,
        runtime_s, objective, h_final, residuals_final.
    """
    n_nodes = node_param.shape[0]
    z = meas["z"]
    sigma = meas["sigma"]

    if x0 is None:
        x0 = np.concatenate([np.ones(n_nodes), np.zeros(n_nodes)])

    def residual(x):
        r = _wls_residual(x, z, sigma, edge_index, node_param, edge_param, meas)
        if lam_phys > 0:
            # Add physical penalty as extra residual terms (sqrt-weighted)
            pen = np.sqrt(lam_phys) * np.sqrt(physical_penalty_np(x, edge_index, node_param, edge_param))
            r = np.concatenate([r, [pen]])
        return r

    t0 = time.time()
    result = least_squares(
        residual,
        x0,
        method=method,
        max_nfev=max_nfev,
        xtol=xtol,
        gtol=gtol,
        verbose=verbose,
    )
    elapsed = time.time() - t0

    x_est = result.x
    h_final = measurement_function(
        x_est, edge_index, node_param, edge_param, meas
    )
    r_final = (z - h_final) / sigma
    obj = 0.5 * np.sum(r_final ** 2)

    return {
        "x_est": x_est,
        "V_est": x_est[:n_nodes],
        "theta_est": x_est[n_nodes:],
        "n_fev": result.nfev,
        "n_iter": result.nit if hasattr(result, "nit") else -1,
        "converged": result.success,
        "runtime_s": elapsed,
        "objective": obj,
        "h_final": h_final,
        "residuals_final": r_final,
        "status": result.status,
        "message": result.message,
    }


# ---------------------------------------------------------------------------
# Modal-WLS solver (iteratively reweighted least squares)
# ---------------------------------------------------------------------------


def solve_modal_wls(
    meas,
    node_param,
    edge_param,
    edge_index,
    x0=None,
    bandwidth=2.0,
    lam_phys=1.0,
    max_iter=30,
    tol=1e-6,
    method="trf",
    max_nfev=500,
    verbose=0,
):
    """
    Modal-WLS state estimation via IRLS.

    At each outer iteration t:
        w_k^(t) = exp(-r_k^(t)^2 / (2 b^2))
        x^(t+1) = argmin Σ w_k * ((z_k - h_k(x)) / σ_k)^2 + λ·penalty(x)

    Args:
        meas:        Measurement dict.
        node_param:  (n_nodes, 3) array.
        edge_param:  (n_edges, 7) array.
        edge_index:  (2, n_edges) array.
        x0:          Initial state (flat start if None).
        bandwidth:   Kernel bandwidth b.
        lam_phys:    Physical penalty weight.
        max_iter:    Maximum IRLS outer iterations.
        tol:         Convergence tolerance on ||Δx||.
        method:      least_squares method per inner iteration.
        max_nfev:    Max function evaluations per inner iteration.
        verbose:     Print iteration info if > 0.

    Returns:
        Dict with keys: x_est, V_est, theta_est, n_iter, converged,
        runtime_s, objective, objective_wls, weights_final, history.
    """
    n_nodes = node_param.shape[0]
    z = meas["z"]
    sigma = meas["sigma"]
    n_meas = len(z)

    if x0 is None:
        x0 = np.concatenate([np.ones(n_nodes), np.zeros(n_nodes)])

    x_curr = x0.copy()
    weights = np.ones(n_meas)  # uniform weights at start
    t0_total = time.time()

    history = {
        "objective": [],
        "objective_wls": [],
        "delta_x": [],
        "n_fev": [],
    }

    converged = False
    n_total_fev = 0

    for iteration in range(max_iter):
        # Define weighted residual for this iteration
        # sqrt(w) * (z - h(x)) / sigma
        w_sqrt = np.sqrt(weights)

        def weighted_residual(x):
            r = _wls_residual(
                x, z, sigma, edge_index, node_param, edge_param, meas
            )
            r_weighted = w_sqrt * r
            if lam_phys > 0:
                pen = np.sqrt(lam_phys) * np.sqrt(
                    physical_penalty_np(x, edge_index, node_param, edge_param)
                )
                r_weighted = np.concatenate([r_weighted, [pen]])
            return r_weighted

        t0 = time.time()
        result = least_squares(
            weighted_residual,
            x_curr,
            method=method,
            max_nfev=max_nfev,
            xtol=1e-10,
            gtol=1e-10,
            verbose=0,
        )
        n_total_fev += result.nfev

        x_new = result.x
        delta = np.linalg.norm(x_new - x_curr)

        # Compute residuals with new state
        h_new = measurement_function(
            x_new, edge_index, node_param, edge_param, meas
        )
        r_new = (z - h_new) / sigma

        # Update weights using kernel
        weights = kernel_value_np(r_new, bandwidth)

        # Compute objectives
        obj_modal = np.sum(
            gaussian_kernel_modal_loss_np(r_new, bandwidth, reduction="none")
        )
        obj_wls = 0.5 * np.sum(r_new ** 2)
        if lam_phys > 0:
            phys = physical_penalty_np(x_new, edge_index, node_param, edge_param)
            obj_modal += lam_phys * phys
            obj_wls += lam_phys * phys

        history["objective"].append(float(obj_modal))
        history["objective_wls"].append(float(obj_wls))
        history["delta_x"].append(float(delta))
        history["n_fev"].append(int(result.nfev))

        if verbose:
            eff_w = weights.mean()
            print(
                f"  IRLS iter {iteration+1:3d}: "
                f"obj_modal={obj_modal:.6e}, "
                f"delta={delta:.2e}, "
                f"mean_w={eff_w:.4f}, "
                f"nfev={result.nfev}"
            )

        x_curr = x_new

        if delta < tol:
            converged = True
            if verbose:
                print(f"  Converged at iteration {iteration+1}")
            break

    elapsed = time.time() - t0_total

    # Final evaluation
    h_final = measurement_function(
        x_curr, edge_index, node_param, edge_param, meas
    )
    r_final = (z - h_final) / sigma

    return {
        "x_est": x_curr,
        "V_est": x_curr[:n_nodes],
        "theta_est": x_curr[n_nodes:],
        "n_iter": len(history["objective"]),
        "n_fev_total": n_total_fev,
        "converged": converged,
        "runtime_s": elapsed,
        "objective": history["objective"][-1] if history["objective"] else np.nan,
        "objective_wls": history["objective_wls"][-1] if history["objective_wls"] else np.nan,
        "weights_final": weights,
        "h_final": h_final,
        "residuals_final": r_final,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Utility: load a single sample
# ---------------------------------------------------------------------------


def load_sample(folder, sample_idx=0, meas_v_buses=None):
    """
    Load a single sample from pickle files.

    Args:
        folder:        e.g. 'data/cigre14/'.
        sample_idx:    Which sample to load.
        meas_v_buses:  List of bus indices with V measurements.

    Returns:
        meas, node_param, edge_param, edge_index, labels
    """
    with open(folder + "nodes", "rb") as f:
        nodes_list = pickle.load(f)
    with open(folder + "edges", "rb") as f:
        edges_list = pickle.load(f)
    with open(folder + "labels", "rb") as f:
        labels_list = pickle.load(f)
    with open(folder + "noise_param", "rb") as f:
        noise = pickle.load(f)

    nodes = nodes_list[sample_idx]
    edges_all = edges_list[sample_idx]
    labels = labels_list[sample_idx]

    # Filter to closed edges only
    edges = edges_all[edges_all["closed line"] == 1.0].copy()
    edges.index = np.arange(edges.shape[0])

    if meas_v_buses is None:
        meas_v_buses = [0, 1, 12, 7, 11, 14]  # CIGRE 14 default

    meas, node_param, edge_param, edge_index = build_measurement_structure(
        nodes, edges, noise, meas_v_buses
    )

    return meas, node_param, edge_param, edge_index, labels


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _run_smoke_test():
    """Run WLS and Modal-WLS on one CIGRE 14 sample, with and without outliers."""
    print("=" * 70)
    print("SMOKE TEST — Modal-WLS solver")
    print("=" * 70)

    # -------------------------------------------------------------------
    # Load sample
    # -------------------------------------------------------------------
    folder = "data/cigre14/"
    meas_v_buses = [0, 1, 12, 7, 11, 14]

    try:
        meas, node_param, edge_param, edge_index, labels = load_sample(
            folder, sample_idx=0, meas_v_buses=meas_v_buses
        )
    except FileNotFoundError:
        print("SKIP: CIGRE 14 data not found. Run toy_network.py first.")
        return

    n_nodes = node_param.shape[0]
    true_V = labels["vm_pu"].values.astype(np.float64)
    true_theta = labels["va_rad"].values.astype(np.float64)
    x_true = np.concatenate([true_V, true_theta])

    print(f"\nLoaded sample 0: {n_nodes} buses, {edge_index.shape[1]} edges, "
          f"{len(meas['z'])} measurements")

    # -------------------------------------------------------------------
    # Test 1: Standard WLS on clean data
    # -------------------------------------------------------------------
    print("\n[Test 1] Standard WLS (clean data)")
    result_wls = solve_wls(meas, node_param, edge_param, edge_index, lam_phys=0.0)

    err_V = np.abs(result_wls["V_est"] - true_V).max()
    err_theta = np.abs(result_wls["theta_est"] - true_theta).max()
    print(f"  Converged:   {result_wls['converged']}")
    print(f"  Runtime:     {result_wls['runtime_s']:.4f} s")
    print(f"  Objective:   {result_wls['objective']:.6e}")
    print(f"  Max V error:   {err_V:.6f} pu")
    print(f"  Max θ error:   {err_theta:.6f} rad")

    assert result_wls["converged"], "WLS should converge on clean data"
    assert err_V < 0.1, f"WLS V error too large: {err_V}"
    print("  PASS")

    # -------------------------------------------------------------------
    # Test 2: Modal-WLS on clean data
    # -------------------------------------------------------------------
    print("\n[Test 2] Modal-WLS (clean data)")
    result_mwls = solve_modal_wls(
        meas,
        node_param,
        edge_param,
        edge_index,
        bandwidth=2.0,
        lam_phys=0.0,
        max_iter=20,
        verbose=1,
    )

    err_V = np.abs(result_mwls["V_est"] - true_V).max()
    print(f"  Converged:   {result_mwls['converged']}")
    print(f"  IRLS iters:  {result_mwls['n_iter']}")
    print(f"  Runtime:     {result_mwls['runtime_s']:.4f} s")
    print(f"  Objective:   {result_mwls['objective']:.6e}")
    print(f"  Max V error:   {err_V:.6f} pu")

    assert result_mwls["converged"], "Modal-WLS should converge on clean data"
    assert err_V < 0.1, f"Modal-WLS V error too large: {err_V}"
    print("  PASS")

    # -------------------------------------------------------------------
    # Test 3: Compare under synthetic outlier noise
    # -------------------------------------------------------------------
    print("\n[Test 3] Outlier robustness comparison")

    # Create a corrupted measurement vector: add large errors to 10% of meas
    np.random.seed(42)
    n_meas = len(meas["z"])
    n_outliers = max(1, n_meas // 10)
    outlier_idx = np.random.choice(n_meas, size=n_outliers, replace=False)

    meas_outlier = dict(meas)
    z_corrupted = meas["z"].copy()
    outlier_scale = 20.0 * np.abs(meas["z"][outlier_idx]) + 1.0
    z_corrupted[outlier_idx] += (
        np.sign(np.random.randn(n_outliers)) * outlier_scale
    )
    meas_outlier["z"] = z_corrupted

    print(f"  Added {n_outliers} outliers to {n_meas} measurements "
          f"({100*n_outliers/n_meas:.0f}%)")

    # Run both on corrupted data
    result_wls_out = solve_wls(
        meas_outlier, node_param, edge_param, edge_index, lam_phys=0.0
    )
    result_mwls_out = solve_modal_wls(
        meas_outlier,
        node_param,
        edge_param,
        edge_index,
        bandwidth=2.0,
        lam_phys=0.0,
        max_iter=30,
    )

    # Compute estimation errors vs true state
    wls_V_rmse = np.sqrt(np.mean((result_wls_out["V_est"] - true_V) ** 2))
    mwls_V_rmse = np.sqrt(np.mean((result_mwls_out["V_est"] - true_V) ** 2))
    wls_theta_rmse = np.sqrt(
        np.mean((result_wls_out["theta_est"] - true_theta) ** 2)
    )
    mwls_theta_rmse = np.sqrt(
        np.mean((result_mwls_out["theta_est"] - true_theta) ** 2)
    )

    # Modal score on true state
    state_scale = np.concatenate(
        [np.full(n_nodes, 0.02), np.full(n_nodes, 0.01)]
    )
    ms_wls = modal_score_np(
        x_true, result_wls_out["x_est"], state_scale, bandwidth=2.0
    )
    ms_mwls = modal_score_np(
        x_true, result_mwls_out["x_est"], state_scale, bandwidth=2.0
    )
    hr_wls = hit_rate_np(
        x_true, result_wls_out["x_est"], state_scale, delta=3.0
    )
    hr_mwls = hit_rate_np(
        x_true, result_mwls_out["x_est"], state_scale, delta=3.0
    )

    print(f"\n  {'':>20}  {'WLS':>12}  {'Modal-WLS':>12}")
    print(f"  {'V RMSE':>20}  {wls_V_rmse:12.6f}  {mwls_V_rmse:12.6f}")
    print(f"  {'θ RMSE':>20}  {wls_theta_rmse:12.6f}  {mwls_theta_rmse:12.6f}")
    print(f"  {'ModalScore':>20}  {ms_wls:12.6f}  {ms_mwls:12.6f}")
    print(f"  {'HitRate':>20}  {hr_wls:12.6f}  {hr_mwls:12.6f}")
    print(f"  {'Converged':>20}  {str(result_wls_out['converged']):>12}  "
          f"{str(result_mwls_out['converged']):>12}")
    print(f"  {'Runtime (s)':>20}  {result_wls_out['runtime_s']:12.4f}  "
          f"{result_mwls_out['runtime_s']:12.4f}")

    # Modal-WLS should be more robust
    assert mwls_V_rmse < wls_V_rmse * 1.1, (
        f"Modal-WLS V RMSE ({mwls_V_rmse:.4f}) should not be much worse "
        f"than WLS ({wls_V_rmse:.4f})"
    )
    assert ms_mwls >= ms_wls * 0.95, (
        f"Modal-WLS ModalScore ({ms_mwls:.4f}) should not be much worse "
        f"than WLS ({ms_wls:.4f})"
    )
    print("  PASS")

    # -------------------------------------------------------------------
    # Test 4: Modal-WLS downweights outliers
    # -------------------------------------------------------------------
    print("\n[Test 4] Verify outlier downweighting")
    weights = result_mwls_out["weights_final"]
    w_outlier = weights[outlier_idx]
    w_inlier = np.delete(weights, outlier_idx)
    mean_w_out = w_outlier.mean()
    mean_w_in = w_inlier.mean()

    print(f"  Mean weight on outliers:  {mean_w_out:.4f}")
    print(f"  Mean weight on inliers:   {mean_w_in:.4f}")
    assert mean_w_out < mean_w_in, (
        f"Outlier weights ({mean_w_out:.4f}) should be lower than "
        f"inlier weights ({mean_w_in:.4f})"
    )
    print("  PASS")

    # -------------------------------------------------------------------
    # Test 5: Bandwidth sensitivity
    # -------------------------------------------------------------------
    print("\n[Test 5] Bandwidth sensitivity")
    for bw in [0.5, 1.0, 3.0, 10.0]:
        r_mwls = solve_modal_wls(
            meas_outlier,
            node_param,
            edge_param,
            edge_index,
            bandwidth=bw,
            lam_phys=0.0,
            max_iter=30,
            verbose=0,
        )
        v_rmse = np.sqrt(np.mean((r_mwls["V_est"] - true_V) ** 2))
        mean_w = r_mwls["weights_final"].mean()
        print(f"  b={bw:5.1f}: V_RMSE={v_rmse:.6f}, mean_w={mean_w:.4f}, "
              f"iters={r_mwls['n_iter']}")

    print()
    print("=" * 70)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    _run_smoke_test()
