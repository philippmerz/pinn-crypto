"""Tests for execution strategies — trade sizing, total execution, boundary behavior."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import numpy as np
import pytest
from functools import partial

from pinn.network import MLP, HardConstrainedExecution
from pinn.physics import almgren_chriss_residual
from pinn.training import TrainConfig, train_pinn
from pinn.strategies import PINNStrategy
from pinn.backtest import TWAPStrategy, AnalyticalACStrategy


@pytest.fixture
def trained_pinn():
    """Train a small PINN at κ=3 for testing."""
    Q = 1.0
    kappa = 3.0
    base = MLP(input_dim=1, output_dim=1, hidden_dim=64, num_layers=4)
    model = HardConstrainedExecution(base, initial_inventory=Q)
    residual_fn = partial(almgren_chriss_residual, kappa=kappa, horizon=1.0)
    config = TrainConfig(n_collocation=200, adam_epochs=5000, lbfgs_epochs=200)
    train_pinn(model, residual_fn, config)
    return model


def test_pinn_total_execution_matches_inventory(trained_pinn):
    """PINN strategy must sell approximately the full inventory."""
    actual_btc = 0.5
    n_intervals = 30
    strategy = PINNStrategy(trained_pinn, actual_btc, n_intervals)

    total_sold = 0.0
    remaining = actual_btc
    for i in range(n_intervals):
        t = i / n_intervals
        trade = strategy.get_trade_rate(t, remaining, {})
        assert trade >= 0, f"Trade should be non-negative, got {trade}"
        assert trade <= remaining + 1e-8, f"Trade {trade} exceeds remaining {remaining}"
        total_sold += trade
        remaining -= trade

    rel_error = abs(total_sold - actual_btc) / actual_btc
    assert rel_error < 0.05, (
        f"Total sold {total_sold:.4f} differs from target {actual_btc} by {rel_error:.1%}"
    )


def test_pinn_no_single_interval_dumps_everything(trained_pinn):
    """No single interval should consume more than 50% of total inventory."""
    actual_btc = 1.0
    n_intervals = 30
    strategy = PINNStrategy(trained_pinn, actual_btc, n_intervals)

    remaining = actual_btc
    for i in range(n_intervals):
        t = i / n_intervals
        trade = strategy.get_trade_rate(t, remaining, {})
        assert trade < actual_btc * 0.5, (
            f"Interval {i} trades {trade:.4f}, which is >{50}% of inventory — likely a scaling bug"
        )
        remaining -= trade


def test_twap_total_execution():
    """TWAP should sell exactly the full inventory."""
    strategy = TWAPStrategy(1.0, 30)
    total = sum(strategy.get_trade_rate(i / 30, 1.0 - i / 30, {}) for i in range(30))
    assert abs(total - 1.0) < 1e-6


def test_analytical_ac_total_execution():
    """Analytical AC should sell approximately the full inventory."""
    strategy = AnalyticalACStrategy(1.0, 5.0, 30)
    total = 0.0
    remaining = 1.0
    for i in range(30):
        trade = strategy.get_trade_rate(i / 30, remaining, {})
        total += trade
        remaining -= trade
    assert abs(total - 1.0) < 0.01, f"AC sold {total:.4f}, expected ~1.0"
