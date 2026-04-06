"""PINN-based execution strategies."""

import torch
import numpy as np


class PINNStrategy:
    """Execution strategy using a trained Phase 1 or Phase 2 PINN.

    The PINN predicts the optimal inventory trajectory X(τ).
    The trade rate at each interval is X(τ) - X(τ + dτ).
    """

    def __init__(self, model, total_inventory: float, kappa: float | None = None):
        """
        Args:
            model: trained HardConstrainedExecution model
            total_inventory: Q (used for denormalization)
            kappa: for parametric PINN (Phase 2), the κ value to use.
                   If None, assumes a single-κ model (Phase 1).
        """
        self.model = model
        self.model.eval()
        self.Q = total_inventory
        self.kappa = kappa

    def _predict_inventory(self, tau: float) -> float:
        """Predict remaining inventory at normalized time τ."""
        t_tensor = torch.tensor([[tau]], dtype=torch.float32)
        if self.kappa is not None:
            k_tensor = torch.tensor([[self.kappa]], dtype=torch.float32)
            t_tensor = torch.cat([t_tensor, k_tensor], dim=1)

        with torch.no_grad():
            return self.model(t_tensor).item()

    def get_trade_rate(self, t: float, remaining: float, market_state: dict) -> float:
        """Compute trade size for this interval."""
        dt = 1e-3  # small step for derivative approximation
        X_now = self._predict_inventory(t)
        X_next = self._predict_inventory(min(t + dt, 1.0))

        # Trade rate = negative inventory change, scaled by interval size
        # The PINN works in normalized time; we need to convert to interval size
        rate_per_unit_time = -(X_next - X_now) / dt

        # Scale by actual interval duration (1/n_intervals in normalized time)
        # This is handled by the caller — we just return the per-interval trade
        trade = max(rate_per_unit_time * dt * 1000, 0)  # rough scaling
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
        total_inventory: float,
        lambda_risk: float,
        base_sigma: float,
        base_eta: float,
        base_price: float,
    ):
        self.model = model
        self.model.eval()
        self.Q = total_inventory
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
        kappa = np.clip(kappa, 0.1, 25.0)  # stay within trained range

        tau = torch.tensor([[t, kappa]], dtype=torch.float32)
        dt = 1e-3
        tau_next = torch.tensor([[min(t + dt, 1.0), kappa]], dtype=torch.float32)

        with torch.no_grad():
            X_now = self.model(tau).item()
            X_next = self.model(tau_next).item()

        rate = -(X_next - X_now) / dt
        trade = max(rate * dt * 1000, 0)
        return min(trade, remaining)
