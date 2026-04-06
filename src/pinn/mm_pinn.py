"""PINN for the modified Avellaneda-Stoikov HJB on prediction markets.

Learns the certainty-equivalent value function θ(t, z, q) by minimizing
the HJB residual. Validates against the FD solver, then extends to a
parametric version over (γ, σ, κ) for instant inference across markets.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass

from pinn.market_making import MarketMakerParams, sigmoid, terminal_condition, solve_fd


class MMValueNetwork(nn.Module):
    """Network that learns θ(t, z, q) for the market-making HJB.

    Inputs: (τ, z, q_normalized) where τ = t/T ∈ [0,1], z ∈ ℝ, q/Q ∈ [-1,1]
    Output: θ (scalar)

    The network is shared across all inventory levels — q enters as a
    continuous input (normalized to [-1, 1]), so the network learns the
    full θ surface rather than separate functions per q.
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 5):
        super().__init__()
        # Input: (τ, z, q_norm) = 3 dimensions
        # Modified MLP with input encoding (same architecture as execution PINN)
        self.encoder = nn.Linear(3, hidden_dim)
        self.U = nn.Linear(3, hidden_dim)
        self.V = nn.Linear(3, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])
        self.output = nn.Linear(hidden_dim, 1)
        self.act = nn.Tanh()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.act(self.U(x))
        v = self.act(self.V(x))
        h = self.act(self.encoder(x))
        for layer in self.layers:
            h = self.act(layer(h))
            h = h * u + (1 - h) * v
        return self.output(h)


def _sigmoid_torch(z: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(z)


def compute_hjb_residual(
    model: nn.Module,
    tau: torch.Tensor,
    z: torch.Tensor,
    q_norm: torch.Tensor,
    params: MarketMakerParams,
) -> torch.Tensor:
    """Compute the HJB PDE residual at collocation points.

    The PDE (for each q):
    ∂θ/∂t + (σ²/2)·∂²θ/∂z² - (γσ²/2)·(∂θ/∂z)²
        + C·[exp(-γ(θ_q - θ_{q+1}))·𝟙{q<Q} + exp(-γ(θ_q - θ_{q-1}))·𝟙{q>-Q}] = 0

    Since t goes from 0 to T but we normalize τ = t/T:
    (1/T)·∂θ/∂τ + (σ²/2)·∂²θ/∂z² - (γσ²/2)·(∂θ/∂z)² + H = 0
    """
    gamma = params.gamma
    sigma = params.sigma
    T = params.T
    Q = params.q_max
    C = params.hamiltonian_coeff
    dq = 1.0 / Q  # q_norm step for one share

    # Build input and enable gradients
    inp = torch.cat([tau, z, q_norm], dim=1)
    inp.requires_grad_(True)

    theta = model(inp)

    # First derivatives via autograd
    grads = torch.autograd.grad(
        theta, inp, grad_outputs=torch.ones_like(theta),
        create_graph=True, retain_graph=True,
    )[0]
    dtheta_dtau = grads[:, 0:1]
    dtheta_dz = grads[:, 1:2]

    # Second derivative w.r.t. z
    d2theta_dz2 = torch.autograd.grad(
        dtheta_dz, inp, grad_outputs=torch.ones_like(dtheta_dz),
        create_graph=True, retain_graph=True,
    )[0][:, 1:2]

    # PDE terms
    time_term = (1.0 / T) * dtheta_dtau
    diffusion = (sigma**2 / 2) * d2theta_dz2
    risk = -(gamma * sigma**2 / 2) * dtheta_dz**2

    # Hamiltonian: evaluate θ at neighboring q values
    q_actual = q_norm * Q  # denormalize to actual inventory
    q_plus_norm = torch.clamp(q_norm + dq, -1.0, 1.0)
    q_minus_norm = torch.clamp(q_norm - dq, -1.0, 1.0)

    inp_plus = torch.cat([tau, z, q_plus_norm], dim=1)
    inp_minus = torch.cat([tau, z, q_minus_norm], dim=1)

    with torch.no_grad():
        theta_plus = model(inp_plus)
        theta_minus = model(inp_minus)

    H = torch.zeros_like(theta)
    # Can buy (q < Q): bid Hamiltonian
    can_buy = (q_actual < Q - 0.5).float()
    H += can_buy * C * torch.exp(-gamma * (theta - theta_plus))
    # Can sell (q > -Q): ask Hamiltonian
    can_sell = (q_actual > -Q + 0.5).float()
    H += can_sell * C * torch.exp(-gamma * (theta - theta_minus))

    residual = time_term + diffusion + risk + H
    return residual


def compute_terminal_loss(
    model: nn.Module,
    z_terminal: torch.Tensor,
    params: MarketMakerParams,
    n_q_samples: int = 21,
) -> torch.Tensor:
    """Loss from terminal condition θ(T, z, q) = -(1/γ)·log(p·exp(-γq) + (1-p))."""
    Q = params.q_max
    gamma = params.gamma
    q_values = torch.linspace(-Q, Q, n_q_samples)

    total_loss = torch.tensor(0.0)
    n_z = z_terminal.shape[0]

    for q_val in q_values:
        q_norm = (q_val / Q) * torch.ones(n_z, 1)
        tau_T = torch.ones(n_z, 1)
        inp = torch.cat([tau_T, z_terminal, q_norm], dim=1)

        theta_pred = model(inp)

        # Analytical terminal condition
        p = torch.sigmoid(z_terminal)
        q_int = int(q_val.item())
        if q_int == 0:
            theta_exact = torch.zeros_like(p)
        else:
            inner = p * torch.exp(torch.tensor(-gamma * q_int)) + (1 - p)
            inner = torch.clamp(inner, min=1e-30)
            theta_exact = -(1.0 / gamma) * torch.log(inner)

        total_loss += torch.mean((theta_pred - theta_exact) ** 2)

    return total_loss / len(q_values)


@dataclass
class MMPINNConfig:
    """Training configuration for market-making PINN."""

    n_colloc_tau: int = 30
    n_colloc_z: int = 60
    n_colloc_q: int = 15
    n_terminal_z: int = 100
    adam_epochs: int = 15_000
    adam_lr: float = 1e-3
    lbfgs_epochs: int = 300
    w_pde: float = 1.0
    w_terminal: float = 10.0
    log_every: int = 1000


def train_mm_pinn(
    params: MarketMakerParams,
    config: MMPINNConfig = MMPINNConfig(),
    callback=None,
) -> tuple[nn.Module, list[float]]:
    """Train the market-making PINN."""
    model = MMValueNetwork(hidden_dim=128, num_layers=5)
    Q = params.q_max

    # Collocation points
    tau_pts = torch.linspace(0.01, 0.99, config.n_colloc_tau)
    z_pts = torch.linspace(-4.0, 4.0, config.n_colloc_z)
    q_norm_pts = torch.linspace(-1.0, 1.0, config.n_colloc_q)

    # Create grid (subsample for memory)
    tau_grid, z_grid, q_grid = torch.meshgrid(tau_pts, z_pts, q_norm_pts, indexing="ij")
    tau_flat = tau_grid.reshape(-1, 1)
    z_flat = z_grid.reshape(-1, 1)
    q_flat = q_grid.reshape(-1, 1)

    # Terminal points
    z_terminal = torch.linspace(-4.0, 4.0, config.n_terminal_z).reshape(-1, 1)

    losses = []

    # Phase 1: Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=config.adam_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.adam_epochs)

    for epoch in range(config.adam_epochs):
        optimizer.zero_grad()

        # PDE residual (on a random subset for efficiency)
        n_total = tau_flat.shape[0]
        idx = torch.randperm(n_total)[:min(5000, n_total)]
        residual = compute_hjb_residual(
            model, tau_flat[idx], z_flat[idx], q_flat[idx], params,
        )
        loss_pde = torch.mean(residual**2)

        # Terminal condition
        loss_terminal = compute_terminal_loss(model, z_terminal, params)

        loss = config.w_pde * loss_pde + config.w_terminal * loss_terminal
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if callback and (epoch % config.log_every == 0 or epoch == config.adam_epochs - 1):
            callback(epoch, loss.item(), loss_pde.item(), loss_terminal.item())

    # Phase 2: L-BFGS
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=20,
        history_size=50, line_search_fn="strong_wolfe",
    )

    for epoch in range(config.lbfgs_epochs):
        def closure():
            optimizer.zero_grad()
            idx = torch.randperm(n_total)[:min(5000, n_total)]
            residual = compute_hjb_residual(
                model, tau_flat[idx], z_flat[idx], q_flat[idx], params,
            )
            loss_pde = torch.mean(residual**2)
            loss_terminal = compute_terminal_loss(model, z_terminal, params)
            loss = config.w_pde * loss_pde + config.w_terminal * loss_terminal
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        losses.append(loss.item())

    return model, losses


def compare_with_fd(
    model: nn.Module,
    fd_result: dict,
    params: MarketMakerParams,
) -> dict[int, float]:
    """Compare PINN solution against FD ground truth. Returns max relative error per q."""
    z_fd = fd_result["z"]
    theta_fd = fd_result["theta"]
    Q = params.q_max
    model.eval()

    errors = {}
    # Compare at t=0 (first stored time)
    z_torch = torch.tensor(z_fd, dtype=torch.float32).reshape(-1, 1)
    tau_0 = torch.zeros_like(z_torch)

    for q in range(-Q, Q + 1):
        q_norm = torch.full_like(z_torch, q / Q)
        inp = torch.cat([tau_0, z_torch, q_norm], dim=1)

        with torch.no_grad():
            theta_pinn = model(inp).numpy().flatten()

        theta_exact = theta_fd[q][0]  # t=0 slice
        max_err = np.max(np.abs(theta_pinn - theta_exact))
        scale = np.max(np.abs(theta_exact)) + 1e-10
        errors[q] = max_err / scale

    return errors
