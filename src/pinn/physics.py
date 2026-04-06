"""Financial differential equations as physics-informed constraints."""

import torch
from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class AlmgrenChrissParams:
    """Parameters for the Almgren-Chriss optimal execution model.

    The model minimizes E[cost] + lambda_risk * Var[cost] for liquidating
    `initial_inventory` shares over horizon `T`, subject to linear
    temporary impact (eta) and permanent impact (theta).

    The Euler-Lagrange ODE is: d²X/dt² - kappa² * X(t) = 0
    where kappa = sqrt(lambda_risk * sigma² * S0² / eta).
    """

    initial_inventory: float  # Q: shares to liquidate
    horizon: float  # T: time horizon (seconds, minutes, or normalized)
    sigma: float  # daily (or per-period) volatility of the asset
    eta: float  # temporary impact coefficient
    theta: float  # permanent impact coefficient
    lambda_risk: float  # risk aversion parameter
    s0: float = 1.0  # reference price (for normalization)

    @property
    def kappa(self) -> float:
        """Urgency parameter: higher kappa → more front-loaded execution."""
        return sqrt(self.lambda_risk * self.sigma**2 * self.s0**2 / self.eta)


def almgren_chriss_analytical(t: torch.Tensor, params: AlmgrenChrissParams) -> torch.Tensor:
    """Closed-form optimal trajectory: X(t) = Q * sinh(κ(T-t)) / sinh(κT)."""
    k = params.kappa
    T = params.horizon
    Q = params.initial_inventory
    return Q * torch.sinh(k * (T - t)) / torch.sinh(torch.tensor(k * T))


def almgren_chriss_residual(
    tau: torch.Tensor,
    X: torch.Tensor,
    kappa: float | torch.Tensor,
    horizon: float,
) -> torch.Tensor:
    """Compute the ODE residual d²X/dt² - κ²X via automatic differentiation.

    Args:
        tau: normalized time points in [0,1], shape (N, 1), requires_grad=True
        X: network output (inventory), shape (N, 1)
        kappa: urgency parameter (scalar or tensor for parametric PINN)
        horizon: time horizon T (used to scale derivatives back to original time)

    Returns:
        Residual tensor of shape (N, 1). Should be zero for exact solutions.
    """
    # dX/dτ via autograd
    dX_dtau = torch.autograd.grad(
        X,
        tau,
        grad_outputs=torch.ones_like(X),
        create_graph=True,
        retain_graph=True,
    )[0]

    # d²X/dτ² via autograd
    d2X_dtau2 = torch.autograd.grad(
        dX_dtau,
        tau,
        grad_outputs=torch.ones_like(dX_dtau),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Chain rule: d²X/dt² = (1/T²) * d²X/dτ²
    # ODE: d²X/dt² - κ²X = 0
    # Normalized form: (1/(κT)²) d²X/dτ² - X = 0
    # This keeps the residual O(1) regardless of κ, preventing gradient explosion.
    T = horizon
    kappa_T_sq = (kappa * T) ** 2
    return (1.0 / kappa_T_sq) * d2X_dtau2 - X
