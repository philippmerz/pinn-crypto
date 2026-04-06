"""Tests for network module — architecture, hard constraints."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import pytest
from pinn.network import MLP, HardConstrainedExecution


def test_mlp_output_shape():
    net = MLP(input_dim=1, output_dim=1, hidden_dim=32, num_layers=3)
    x = torch.randn(50, 1)
    y = net(x)
    assert y.shape == (50, 1)


def test_mlp_multidim_input():
    net = MLP(input_dim=3, output_dim=1, hidden_dim=32, num_layers=3)
    x = torch.randn(50, 3)
    y = net(x)
    assert y.shape == (50, 1)


def test_hard_constraint_boundaries():
    """Hard constraint must enforce X(0) = Q, X(T) = 0 exactly."""
    Q = 1000.0
    base = MLP(input_dim=1, output_dim=1, hidden_dim=32, num_layers=3)
    model = HardConstrainedExecution(base, initial_inventory=Q)

    # Test at τ = 0
    tau_0 = torch.zeros(1, 1)
    X_0 = model(tau_0)
    assert torch.allclose(X_0, torch.tensor([[Q]])), f"X(0) should be {Q}, got {X_0.item()}"

    # Test at τ = 1
    tau_1 = torch.ones(1, 1)
    X_1 = model(tau_1)
    assert torch.allclose(X_1, torch.tensor([[0.0]])), f"X(T) should be 0, got {X_1.item()}"


def test_hard_constraint_differentiable():
    """Must be able to compute second derivatives through the hard constraint."""
    Q = 1000.0
    base = MLP(input_dim=1, output_dim=1, hidden_dim=32, num_layers=3)
    model = HardConstrainedExecution(base, initial_inventory=Q)

    tau = torch.linspace(0.01, 0.99, 50).reshape(-1, 1).requires_grad_(True)
    X = model(tau)

    dX = torch.autograd.grad(X, tau, grad_outputs=torch.ones_like(X), create_graph=True)[0]
    d2X = torch.autograd.grad(dX, tau, grad_outputs=torch.ones_like(dX), create_graph=True)[0]

    assert d2X.shape == (50, 1)
    assert torch.isfinite(d2X).all(), "Second derivatives must be finite"
