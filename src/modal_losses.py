"""
Modal regression loss module for robust distribution system state estimation.

Provides both PyTorch (for GNN/ANN training) and NumPy (for SciPy-based WLS
optimization) implementations of:

  - Gaussian kernel modal loss
  - Measurement-space modal loss
  - State-space modal loss
  - ModalScore metric
  - HitRate metric

The modal loss replaces the squared-error loss with a kernel-based objective
that is robust to outliers and heavy-tailed noise:

    L_modal(r; b) = 1 - exp(-0.5 * (r / b)^2)

where r is the standardized residual and b is the bandwidth parameter.
"""

import torch
import numpy as np

# ---------------------------------------------------------------------------
# PyTorch implementations — for GNN / ANN training
# ---------------------------------------------------------------------------


def gaussian_kernel_modal_loss(r, bandwidth, reduction="mean"):
    """
    Gaussian kernel modal loss.

        L(r; b) = 1 - exp(-0.5 * (r / b)^2)

    Args:
        r:          Standardized residuals, any shape.
        bandwidth:  Kernel bandwidth (scalar or broadcastable tensor). Clipped
                    to min 1e-8 for numerical stability.
        reduction:  'mean', 'sum', or 'none'.

    Returns:
        Modal loss tensor.
    """
    b = torch.as_tensor(bandwidth, dtype=r.dtype, device=r.device)
    b = torch.clamp(b, min=1e-8)
    loss_per_element = 1.0 - torch.exp(-0.5 * (r / b).pow(2))
    if reduction == "mean":
        return loss_per_element.mean()
    elif reduction == "sum":
        return loss_per_element.sum()
    elif reduction == "none":
        return loss_per_element
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def kernel_value(r, bandwidth):
    """
    The kernel value (modal score per element).

        k(r; b) = exp(-0.5 * (r / b)^2)

    This equals 1 - modal_loss(r, bandwidth, reduction='none').
    """
    b = torch.as_tensor(bandwidth, dtype=r.dtype, device=r.device)
    b = torch.clamp(b, min=1e-8)
    return torch.exp(-0.5 * (r / b).pow(2))


def measurement_space_modal_loss(z, h_x, sigma, bandwidth, reduction="mean"):
    """
    Measurement-space modal loss.

        r_k = (z_k - h_k(x)) / sigma_k
        L   = mean_or_sum(1 - exp(-0.5 * r_k^2 / bandwidth^2))

    Args:
        z:          Observed measurements, shape (N, D).
        h_x:        Reconstructed measurements from estimated state, shape
                    (N, D).
        sigma:      Measurement standard deviations, shape (N, D) or
                    broadcastable. Clipped to min 1e-8.
        bandwidth:  Kernel bandwidth (scalar or broadcastable).
        reduction:  'mean', 'sum', or 'none'.

    Returns:
        Modal loss value.
    """
    s = torch.as_tensor(sigma, dtype=z.dtype, device=z.device)
    s = torch.clamp(s, min=1e-8)
    r = (z - h_x) / s
    return gaussian_kernel_modal_loss(r, bandwidth, reduction=reduction)


def state_space_modal_loss(x_true, x_pred, state_scale, bandwidth, reduction="mean"):
    """
    State-space modal loss for supervised learning.

        r_j = (x_true_j - x_pred_j) / state_scale_j
        L   = mean_or_sum(1 - exp(-0.5 * r_j^2 / bandwidth^2))

    Args:
        x_true:       Ground-truth state values, shape (N, D).
        x_pred:       Predicted state values, shape (N, D).
        state_scale:  Scale per state variable, shape (D,) or broadcastable.
                      Clipped to min 1e-8.
        bandwidth:    Kernel bandwidth (scalar or broadcastable).
        reduction:    'mean', 'sum', or 'none'.

    Returns:
        Modal loss value.
    """
    sc = torch.as_tensor(state_scale, dtype=x_true.dtype, device=x_true.device)
    sc = torch.clamp(sc, min=1e-8)
    r = (x_true - x_pred) / sc
    return gaussian_kernel_modal_loss(r, bandwidth, reduction=reduction)


def modal_score(x_true, x_pred, state_scale, bandwidth):
    """
    ModalScore — higher is better.

        score = mean(exp(-0.5 * ((x_true - x_pred) / state_scale / bandwidth)^2))

    This is the average kernel value.  Range: [0, 1].
    A perfect prediction gives 1.0; a very poor prediction gives ~0.
    """
    sc = torch.as_tensor(state_scale, dtype=x_true.dtype, device=x_true.device)
    sc = torch.clamp(sc, min=1e-8)
    r = (x_true - x_pred) / sc
    return kernel_value(r, bandwidth).mean()


def hit_rate(x_true, x_pred, state_scale, delta):
    """
    HitRate — fraction of predictions within ±delta of the true value.

        hit_rate = mean(|(x_true - x_pred) / state_scale| <= delta)

    Args:
        x_true:       Ground-truth state values, shape (N, D).
        x_pred:       Predicted state values, shape (N, D).
        state_scale:  Scale per state variable, shape (D,) or broadcastable.
        delta:        Threshold (scalar).

    Returns:
        Hit rate in [0, 1].
    """
    sc = torch.as_tensor(state_scale, dtype=x_true.dtype, device=x_true.device)
    sc = torch.clamp(sc, min=1e-8)
    r = (x_true - x_pred) / sc
    return (torch.abs(r) <= delta).float().mean()


# ---------------------------------------------------------------------------
# Additional robust loss functions — for comparison experiments
# ---------------------------------------------------------------------------


def huber_loss(r, delta=1.34, reduction="mean"):
    """
    Huber loss — quadratic for |r|≤δ, linear for |r|>δ.

        L(r; δ) = 0.5·r²          if |r| ≤ δ
                   δ·(|r| − 0.5·δ)  otherwise

    Args:
        r:          Residuals, any shape.
        delta:      Threshold parameter (default 1.34 for 95% efficiency).
        reduction:  'mean', 'sum', or 'none'.

    Returns:
        Huber loss tensor.
    """
    abs_r = torch.abs(r)
    quadratic = 0.5 * r.pow(2)
    linear = delta * (abs_r - 0.5 * delta)
    loss = torch.where(abs_r <= delta, quadratic, linear)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def tukey_biweight_loss(r, c=4.68, reduction="mean"):
    """
    Tukey's biweight (bisquare) loss — redescending to zero beyond |r|>c.

        L(r; c) = (c²/6)·[1 − (1−(r/c)²)³]  if |r| ≤ c
                  c²/6                         otherwise

    Args:
        r:          Residuals, any shape.
        c:          Tuning constant (default 4.68 for 95% efficiency).
        reduction:  'mean', 'sum', or 'none'.

    Returns:
        Tukey biweight loss tensor.
    """
    abs_r = torch.abs(r)
    mask = abs_r <= c
    r_scaled = r / c
    loss = torch.where(
        mask,
        (c ** 2 / 6.0) * (1.0 - (1.0 - r_scaled.pow(2)).pow(3)),
        torch.full_like(r, c ** 2 / 6.0),
    )
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def quantile_loss(r, tau=0.5, reduction="mean"):
    """
    Quantile (pinball) loss.

        L(r; τ) = r·(τ − 1[r<0])

    τ=0.5 gives median regression (absolute loss scaled by ½).

    Args:
        r:          Residuals, any shape.
        tau:        Quantile level in (0, 1).
        reduction:  'mean', 'sum', or 'none'.

    Returns:
        Quantile loss tensor.
    """
    loss = r * (tau - (r < 0).float())
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


# ---------------------------------------------------------------------------
# NumPy implementations — for SciPy-based WLS optimization
# ---------------------------------------------------------------------------


def gaussian_kernel_modal_loss_np(r, bandwidth, reduction="mean"):
    """NumPy version of gaussian_kernel_modal_loss."""
    b = np.maximum(np.asarray(bandwidth), 1e-8)
    loss = 1.0 - np.exp(-0.5 * (r / b) ** 2)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def kernel_value_np(r, bandwidth):
    """NumPy version of kernel_value."""
    b = np.maximum(np.asarray(bandwidth), 1e-8)
    return np.exp(-0.5 * (r / b) ** 2)


def measurement_space_modal_loss_np(z, h_x, sigma, bandwidth, reduction="mean"):
    """NumPy version of measurement_space_modal_loss."""
    s = np.maximum(np.asarray(sigma), 1e-8)
    r = (z - h_x) / s
    return gaussian_kernel_modal_loss_np(r, bandwidth, reduction=reduction)


def state_space_modal_loss_np(x_true, x_pred, state_scale, bandwidth, reduction="mean"):
    """NumPy version of state_space_modal_loss."""
    sc = np.maximum(np.asarray(state_scale), 1e-8)
    r = (x_true - x_pred) / sc
    return gaussian_kernel_modal_loss_np(r, bandwidth, reduction=reduction)


def modal_score_np(x_true, x_pred, state_scale, bandwidth):
    """NumPy version of modal_score."""
    sc = np.maximum(np.asarray(state_scale), 1e-8)
    r = (x_true - x_pred) / sc
    return kernel_value_np(r, bandwidth).mean()


def hit_rate_np(x_true, x_pred, state_scale, delta):
    """NumPy version of hit_rate."""
    sc = np.maximum(np.asarray(state_scale), 1e-8)
    r = (x_true - x_pred) / sc
    return np.mean(np.abs(r) <= delta)
