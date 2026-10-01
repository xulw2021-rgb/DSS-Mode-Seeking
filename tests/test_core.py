"""Lightweight correctness tests (no model training required)."""

import os
import sys
import numpy as np
import pandas as pd
import pytest

EXP = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(EXP, "src"))
sys.path.insert(0, os.path.join(EXP, "scripts"))

import torch  # noqa: E402
from modal_losses import (  # noqa: E402
    gaussian_kernel_modal_loss, huber_loss, tukey_biweight_loss,
    quantile_loss, state_space_modal_loss,
)


def test_modal_loss_bounds():
    r = torch.linspace(-20, 20, 4001)
    L = gaussian_kernel_modal_loss(r, 1.0, reduction="none")
    assert torch.all(L >= 0) and torch.all(L <= 1)
    assert abs(float(gaussian_kernel_modal_loss(torch.zeros(1), 1.0))) < 1e-6
    assert abs(float(gaussian_kernel_modal_loss(torch.tensor([50.0]), 1.0)) - 1.0) < 1e-6


def test_modal_score_relation():
    r = torch.linspace(-5, 5, 101)
    L = gaussian_kernel_modal_loss(r, 2.0, reduction="none")
    k = torch.exp(-0.5 * (r / 2.0) ** 2)
    assert torch.allclose(L, 1 - k, atol=1e-6)


def test_huber_form():
    r = torch.tensor([0.5, 2.0, -2.0])
    L = huber_loss(r, delta=1.0, reduction="none")
    expected = torch.tensor([0.5 * 0.25, 1.0 * (2.0 - 0.5), 1.0 * (2.0 - 0.5)])
    assert torch.allclose(L, expected, atol=1e-6)


def test_tukey_redescending():
    r = torch.tensor([0.0, 1.0, 2.0, 3.0])
    L = tukey_biweight_loss(r, c=2.0, reduction="none")
    # flat beyond |r| >= c: all of the last two equal c^2/6
    assert abs(float(L[2] - L[3])) < 1e-6
    assert abs(float(L[2]) - 4.0 / 6.0) < 1e-6


def test_quantile_median():
    r = torch.tensor([-1.0, 1.0, -2.0, 2.0])
    L = quantile_loss(r, tau=0.5, reduction="none")
    expected = 0.5 * torch.abs(r)
    assert torch.allclose(L, expected, atol=1e-6)


def test_state_space_loss_scaling():
    x_true = torch.tensor([[1.0, 0.0]])
    x_pred = torch.tensor([[1.1, 0.05]])
    scale = torch.tensor([0.05, 0.02])
    L = state_space_modal_loss(x_true, x_pred, scale, 1.0, reduction="mean")
    r = torch.tensor([(1.0 - 1.1) / 0.05, (0.0 - 0.05) / 0.02])
    expected = (1 - torch.exp(-0.5 * r ** 2)).mean()
    assert torch.allclose(L, expected, atol=1e-6)


def test_frozen_result_files_present():
    res = os.path.join(EXP, "results", "final")
    for f in ["E1_learning_corrected.csv", "E2_corrected.csv", "E3_corrected.csv",
              "E4_corrected.csv", "E5_corrected.csv", "E6A_ieee30_corrected.csv",
              "E6B_pantagruel_corrected.csv", "E1_solvers.csv",
              "e7_start_level.csv", "e7_selected_solution.csv"]:
        assert os.path.exists(os.path.join(res, f)), f
    # frozen result sanity: E5 grid complete
    e5 = pd.read_csv(os.path.join(res, "E5_corrected.csv"))
    assert len(e5) == 120 and e5.status.eq("SUCCESS").all()
    # corrected results carry the train-only marker
    e1 = pd.read_csv(os.path.join(res, "E1_learning_corrected.csv"))
    assert e1.notes.str.contains("train_only").all()


def test_publication_generator_runs():
    # table/figure generation from frozen results only (no training)
    import subprocess
    out = subprocess.run(
        [sys.executable, os.path.join(EXP, "scripts", "reproduce_publication_artifacts.py")],
        capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    for t in ["Table_2_main_14bus", "Table_3_70bus", "Table_4_cross_system",
              "Table_5_robust_loss", "Table_6_sensitivity", "Table_7_multistart"]:
        assert os.path.exists(os.path.join(EXP, "reproduced", "tables", t + ".tex")), t
    for f in ["fig_1_influence.pdf", "fig_2_main_14bus.pdf",
              "fig_3_cross_system_effect.pdf", "fig_4_bandwidth.pdf",
              "fig_4_lambda.pdf", "fig_5_multistart.pdf"]:
        assert os.path.exists(os.path.join(EXP, "reproduced", "figures", f)), f


def test_verification_script_passes():
    import subprocess
    out = subprocess.run(
        [sys.executable, os.path.join(EXP, "scripts", "verify_publication_artifacts.py")],
        capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr
