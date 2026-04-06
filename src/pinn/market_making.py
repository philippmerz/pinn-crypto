"""Modified Avellaneda-Stoikov HJB for prediction market making.

Solves for the certainty-equivalent value function θ(t, z, q) where:
- t: time (0 = now, T = expiry)
- z: logit(price), z = log(p/(1-p))
- q: inventory (integer shares held)

The market maker quotes bid/ask on a binary outcome token p ∈ (0, 1)
that settles at $0 or $1 at expiry.

Reference: Guéant, Lehalle, Fernandez-Tapia (2013),
"Dealing with the inventory risk: a solution to the market making problem."
"""

from dataclasses import dataclass
from math import log

import numpy as np


@dataclass(frozen=True)
class MarketMakerParams:
    """Parameters for the prediction market making HJB."""

    gamma: float  # risk aversion (higher = more conservative)
    sigma: float  # volatility in logit space
    A: float  # baseline order arrival rate (per unit time)
    kappa: float  # order arrival decay rate (higher = less sensitive to spread)
    q_max: int  # maximum absolute inventory
    T: float  # time horizon

    @property
    def base_spread(self) -> float:
        """Minimum half-spread from the A-S formula: (1/γ)·log(1 + γ/κ)."""
        return (1.0 / self.gamma) * log(1.0 + self.gamma / self.kappa)

    @property
    def hamiltonian_coeff(self) -> float:
        """Coefficient in front of the Hamiltonian terms after substituting optimal δ."""
        gk = self.gamma + self.kappa
        return self.A * (self.kappa / gk) ** (self.kappa / self.gamma) / gk


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def terminal_condition(z: np.ndarray, q: int, gamma: float) -> np.ndarray:
    """Terminal value θ(T, z, q) for binary settlement.

    θ(T, z, q) = -(1/γ) · log(p·exp(-γq) + (1-p))
    where p = sigmoid(z).

    At expiry, each share is worth $1 (if YES) or $0 (if NO).
    The certainty equivalent accounts for the risk of binary settlement.
    """
    p = sigmoid(z)
    if q == 0:
        return np.zeros_like(z)
    # Numerically stable version
    inner = p * np.exp(-gamma * q) + (1.0 - p)
    inner = np.maximum(inner, 1e-300)  # avoid log(0)
    return -(1.0 / gamma) * np.log(inner)


def optimal_quotes(
    theta: dict,
    z: np.ndarray,
    q: int,
    q_max: int,
    params: MarketMakerParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute optimal bid and ask prices from the value function.

    Since θ includes the full certainty equivalent (with mark-to-market),
    the optimal quotes are derived from indifference pricing:

    bid  = θ(q+1) - θ(q) - base_spread   (marginal value of buying minus spread)
    ask  = θ(q) - θ(q-1) + base_spread   (marginal cost of selling plus spread)

    At q=0, p=0.5 with small γ: bid ≈ 0.5 - base, ask ≈ 0.5 + base (symmetric).
    With inventory: quotes skew toward reducing position.
    """
    base = params.base_spread

    q_plus = min(q + 1, q_max)
    q_minus = max(q - 1, -q_max)

    # Marginal value of inventory (indifference prices)
    bid_price = theta[q_plus] - theta[q] - base
    ask_price = theta[q] - theta[q_minus] + base

    bid_price = np.clip(bid_price, 0.01, 0.99)
    ask_price = np.clip(ask_price, 0.01, 0.99)

    return bid_price, ask_price


def solve_fd(
    params: MarketMakerParams,
    n_z: int = 201,
    n_t: int = 1000,
    z_min: float = -6.0,
    z_max: float = 6.0,
) -> dict:
    """Solve the modified A-S HJB via finite differences.

    Uses explicit time-stepping (backward from T) on a grid (t, z, q).

    The PDE for each inventory level q:

    ∂θ/∂t + (σ²/2)·∂²θ/∂z² - (γσ²/2)·(∂θ/∂z)²
        + C·[exp(-γ(θ_q - θ_{q+1}))·𝟙{q < Q} + exp(-γ(θ_q - θ_{q-1}))·𝟙{q > -Q}] = 0

    where C = A·(κ/(κ+γ))^(κ/γ) / (κ+γ).

    Returns dict with keys: 'theta', 'z', 't', 'params'.
    theta[q] is an (n_t, n_z) array.
    """
    gamma = params.gamma
    sigma = params.sigma
    Q = params.q_max
    C = params.hamiltonian_coeff

    dz = (z_max - z_min) / (n_z - 1)
    dt = params.T / (n_t - 1)
    z = np.linspace(z_min, z_max, n_z)
    t = np.linspace(0, params.T, n_t)

    # CFL condition check for explicit scheme
    cfl = sigma**2 * dt / dz**2
    if cfl > 0.5:
        # Increase time resolution to satisfy CFL
        n_t = int(2.0 * sigma**2 * params.T / dz**2) + 10
        dt = params.T / (n_t - 1)
        t = np.linspace(0, params.T, n_t)
        cfl = sigma**2 * dt / dz**2

    # Initialize: theta[q] at time T (terminal condition)
    theta_now = {}
    for q in range(-Q, Q + 1):
        theta_now[q] = terminal_condition(z, q, gamma)

    # Storage for full solution (only store at coarser time intervals)
    store_every = max(1, n_t // 200)
    stored_t = []
    stored_theta = {q: [] for q in range(-Q, Q + 1)}

    # Store terminal condition
    stored_t.append(params.T)
    for q in range(-Q, Q + 1):
        stored_theta[q].append(theta_now[q].copy())

    # Backward time-stepping (from T to 0)
    for step in range(n_t - 2, -1, -1):
        theta_next = {}

        for q in range(-Q, Q + 1):
            th = theta_now[q]

            # Spatial derivatives (central differences, interior only)
            d2th_dz2 = np.zeros_like(th)
            dth_dz = np.zeros_like(th)
            d2th_dz2[1:-1] = (th[2:] - 2 * th[1:-1] + th[:-2]) / dz**2
            dth_dz[1:-1] = (th[2:] - th[:-2]) / (2 * dz)

            # Diffusion term
            diffusion = (sigma**2 / 2) * d2th_dz2

            # Nonlinear risk term
            risk = -(gamma * sigma**2 / 2) * dth_dz**2

            # Hamiltonian terms (market making revenue)
            H = np.zeros_like(th)
            if q < Q:  # can buy (bid)
                H += C * np.exp(-gamma * (th - theta_now[q + 1]))
            if q > -Q:  # can sell (ask)
                H += C * np.exp(-gamma * (th - theta_now[q - 1]))

            # Explicit Euler step (backward in time: θ at t = θ at t+dt + dt·RHS)
            theta_next[q] = th + dt * (diffusion + risk + H)

            # Boundary conditions in z: extrapolate linearly
            theta_next[q][0] = 2 * theta_next[q][1] - theta_next[q][2]
            theta_next[q][-1] = 2 * theta_next[q][-2] - theta_next[q][-3]

        theta_now = theta_next

        if step % store_every == 0:
            stored_t.append(t[step])
            for q in range(-Q, Q + 1):
                stored_theta[q].append(theta_now[q].copy())

    # Reverse time ordering (so index 0 = t=0)
    stored_t.reverse()
    result_theta = {}
    for q in range(-Q, Q + 1):
        stored_theta[q].reverse()
        result_theta[q] = np.array(stored_theta[q])

    return {
        "theta": result_theta,
        "z": z,
        "t": np.array(stored_t),
        "params": params,
    }
