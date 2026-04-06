"""PINN-based execution strategies."""

import torch
import numpy as np


class PINNStrategy:
    """Execution strategy using a trained Phase 1 or Phase 2 PINN.

    The PINN predicts the optimal normalized inventory trajectory X(τ) ∈ [0, 1].
    The trade at interval i is:
        trade_btc = (X(τ_i) - X(τ_{i+1})) * actual_inventory
    """

    def __init__(
        self,
        model,
        actual_inventory: float,
        n_intervals: int,
        kappa: float | None = None,
    ):
        self.model = model
        self.model.eval()
        self.actual_inventory = actual_inventory
        self.n_intervals = n_intervals
        self.dt = 1.0 / n_intervals
        self.kappa = kappa

    def _predict_normalized_inventory(self, tau: float) -> float:
        """Predict normalized remaining inventory X(τ) ∈ [0, 1]."""
        t_tensor = torch.tensor([[tau]], dtype=torch.float32)
        if self.kappa is not None:
            k_tensor = torch.tensor([[self.kappa]], dtype=torch.float32)
            t_tensor = torch.cat([t_tensor, k_tensor], dim=1)
        with torch.no_grad():
            return self.model(t_tensor).item()

    def get_trade_rate(self, t: float, remaining: float, market_state: dict) -> float:
        """Compute trade size for this interval."""
        tau_next = min(t + self.dt, 1.0)
        X_now = self._predict_normalized_inventory(t)
        X_next = self._predict_normalized_inventory(tau_next)

        # Denormalize: model was trained with Q=1, scale to actual inventory
        trade = (X_now - X_next) * self.actual_inventory
        trade = max(trade, 0.0)
        return min(trade, remaining)


class AdaptivePINNStrategy:
    """PINN strategy that recalibrates κ each interval based on market state.

    Uses a parametric PINN (Phase 2) and adapts the urgency parameter
    based on current realized volatility and remaining time.

    κ = sqrt(λ * σ² * S₀² / η)

    When volatility increases, κ increases (more urgent execution).
    When volume decreases, η effectively increases, decreasing κ.
    """

    def __init__(
        self,
        model,
        actual_inventory: float,
        n_intervals: int,
        lambda_risk: float,
        base_sigma: float,
        base_eta: float,
        base_price: float,
    ):
        self.model = model
        self.model.eval()
        self.actual_inventory = actual_inventory
        self.n_intervals = n_intervals
        self.dt = 1.0 / n_intervals
        self.lambda_risk = lambda_risk
        self.base_sigma = base_sigma
        self.base_eta = base_eta
        self.base_price = base_price

    def _compute_kappa(self, market_state: dict) -> float:
        """Compute adaptive κ from current market conditions."""
        sigma = market_state.get("volatility", self.base_sigma)
        if sigma == 0 or np.isnan(sigma):
            sigma = self.base_sigma
        price = market_state.get("price", self.base_price)
        return np.sqrt(self.lambda_risk * sigma**2 * price**2 / self.base_eta)

    def get_trade_rate(self, t: float, remaining: float, market_state: dict) -> float:
        """Compute trade size with adaptive κ."""
        kappa = self._compute_kappa(market_state)
        kappa = np.clip(kappa, 0.1, 25.0)

        tau_next = min(t + self.dt, 1.0)
        tau_tensor = torch.tensor([[t, kappa]], dtype=torch.float32)
        tau_next_tensor = torch.tensor([[tau_next, kappa]], dtype=torch.float32)

        with torch.no_grad():
            X_now = self.model(tau_tensor).item()
            X_next = self.model(tau_next_tensor).item()

        trade = (X_now - X_next) * self.actual_inventory
        trade = max(trade, 0.0)
        return min(trade, remaining)
