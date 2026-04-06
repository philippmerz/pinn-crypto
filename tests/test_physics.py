"""Tests for physics module — analytical solutions, residual computation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import pytest
from pinn.physics import AlmgrenChrissParams, almgren_chriss_analytical, almgren_chriss_residual


@pytest.fixture
def params():
    return AlmgrenChrissParams(
        initial_inventory=1000.0,
        horizon=1.0,
        sigma=0.02,
        eta=0.01,
        theta=0.005,
        lambda_risk=1e-2,
        s0=100.0,
    )


def test_analytical_boundary_conditions(params):
    """X(0) = Q, X(T) = 0."""
    t0 = torch.tensor([0.0])
    tT = torch.tensor([params.horizon])
    assert torch.allclose(almgren_chriss_analytical(t0, params), torch.tensor([params.initial_inventory]), atol=1e-6)
    assert torch.allclose(almgren_chriss_analytical(tT, params), torch.tensor([0.0]), atol=1e-6)


def test_analytical_monotonically_decreasing(params):
    """Inventory should decrease over time."""
    t = torch.linspace(0, params.horizon, 100)
    X = almgren_chriss_analytical(t, params)
    diffs = X[1:] - X[:-1]
    assert (diffs <= 0).all(), "Trajectory should be monotonically decreasing"


def test_residual_of_analytical_is_zero(params):
    """The analytical solution should have zero ODE residual."""
    tau = torch.linspace(0.01, 0.99, 100, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    t = tau * params.horizon
    X = almgren_chriss_analytical(t, params)

    residual = almgren_chriss_residual(tau, X, params.kappa, params.horizon)
    assert residual.abs().max().item() < 1e-8, f"Residual too large: {residual.abs().max().item()}"


def test_kappa_scaling(params):
    """Higher lambda_risk → higher kappa → more front-loaded."""
    from dataclasses import replace

    params_low = replace(params, lambda_risk=1e-4)
    params_high = replace(params, lambda_risk=1.0)

    assert params_high.kappa > params_low.kappa

    # At t = T/4, high-urgency should have liquidated more
    t_quarter = torch.tensor([params.horizon * 0.25])
    X_low = almgren_chriss_analytical(t_quarter, params_low)
    X_high = almgren_chriss_analytical(t_quarter, params_high)
    assert X_high < X_low, "Higher kappa should liquidate faster"
