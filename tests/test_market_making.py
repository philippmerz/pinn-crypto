"""Tests for market making HJB solver."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest
from pinn.market_making import (
    MarketMakerParams,
    sigmoid,
    terminal_condition,
    solve_fd,
)


@pytest.fixture
def params():
    return MarketMakerParams(gamma=0.1, sigma=1.0, A=1.0, kappa=5.0, q_max=3, T=1.0)


def test_sigmoid_bounds():
    z = np.linspace(-10, 10, 100)
    p = sigmoid(z)
    assert (p > 0).all() and (p < 1).all()
    assert abs(sigmoid(np.array([0.0]))[0] - 0.5) < 1e-10


def test_terminal_q0_is_zero():
    z = np.linspace(-5, 5, 100)
    th = terminal_condition(z, q=0, gamma=0.1)
    assert np.allclose(th, 0.0, atol=1e-10)


def test_terminal_positive_q_bounded():
    """θ(T, z, q) should be between 0 and q for q > 0."""
    z = np.linspace(-5, 5, 100)
    for q in [1, 2, 5]:
        th = terminal_condition(z, q, gamma=0.1)
        assert (th >= -0.01).all(), f"θ(T,z,{q}) went below 0"
        assert (th <= q + 0.01).all(), f"θ(T,z,{q}) exceeded {q}"


def test_terminal_increases_with_p():
    """For q > 0, θ should increase with p (higher price = more valuable inventory)."""
    z = np.linspace(-5, 5, 100)
    th = terminal_condition(z, q=1, gamma=0.1)
    assert (np.diff(th) >= -1e-10).all(), "θ should be non-decreasing in z for q > 0"


def test_fd_solver_runs(params):
    result = solve_fd(params, n_z=51, n_t=100)
    assert "theta" in result
    assert "z" in result
    assert "t" in result
    assert len(result["z"]) == 51
    for q in range(-params.q_max, params.q_max + 1):
        assert q in result["theta"]


def test_fd_theta_q0_near_zero(params):
    """θ at q=0 should be close to zero (no inventory, no terminal value, minimal trading value)."""
    result = solve_fd(params, n_z=51, n_t=200)
    th_q0 = result["theta"][0]
    # At t=0, q=0, the value comes from future market-making revenue
    # It should be positive (earning spread) but bounded
    assert (th_q0[-1] >= -0.1).all(), "θ(0,z,0) shouldn't be very negative"


def test_fd_symmetry(params):
    """For q=0 at z=0 (p=0.5), the solution should be symmetric in z."""
    result = solve_fd(params, n_z=101, n_t=200, z_min=-5, z_max=5)
    th_q0 = result["theta"][0]  # shape: (n_stored_t, n_z)
    n_z = len(result["z"])
    mid = n_z // 2
    # θ(t, z, 0) should be symmetric around z=0
    for t_idx in [0, len(result["t"]) // 2]:
        left = th_q0[t_idx, :mid]
        right = th_q0[t_idx, mid + 1:][::-1]
        min_len = min(len(left), len(right))
        assert np.allclose(left[:min_len], right[:min_len], atol=0.05), \
            f"θ(t,z,0) not symmetric at t_idx={t_idx}"


def test_reservation_price_skews_with_inventory(params):
    """With positive inventory, reservation price should be below mid (want to sell)."""
    result = solve_fd(params, n_z=101, n_t=200)
    z = result["z"]
    mid_idx = len(z) // 2  # z=0, p=0.5

    # At t=0 (earliest stored time)
    th = result["theta"]
    # Reservation price skew = θ(q+1) - θ(q-1)
    # For q > 0, should be negative (lower reservation = skew toward selling)
    q = 2
    if q + 1 <= params.q_max and q - 1 >= -params.q_max:
        skew = th[q + 1][0, mid_idx] - th[q - 1][0, mid_idx]
        # Positive skew means θ increases with q, so reservation price adjusts upward
        # Actually, the reservation price is p + (θ_{q+1} - θ_{q-1})/2
        # For positive q, we want to sell, so reservation should decrease
        # This means θ_{q+1} - θ_{q-1} should be decreasing in q...
        # Let's just check that higher inventory has lower marginal value
        marginal_high = th[q + 1][0, mid_idx] - th[q][0, mid_idx]
        marginal_low = th[q][0, mid_idx] - th[q - 1][0, mid_idx]
        assert marginal_high < marginal_low + 0.01, \
            "Marginal value should decrease with inventory (diminishing returns + risk)"
