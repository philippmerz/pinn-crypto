"""Parametric market-making PINN: one model for any market parameters.

Takes 6D input (τ, z, q_norm, σ, κ, γ) and outputs θ, the certainty-
equivalent value function. Trained with a hybrid loss:

L = w_data * L_supervised + w_pde * L_HJB_residual + w_terminal * L_terminal

The supervised loss anchors absolute θ values using FD reference solutions
(solving the level-shift problem from the single-market PINN). The PDE
loss ensures physical consistency at unseen parameter combinations.
"""

import torch
import torch.nn as nn
import numpy as np
from math import log

from pinn.fd_dataset import FDDataset


class ParametricMMNetwork(nn.Module):
    """Network for θ(τ, z, q_norm, σ, κ, γ).

    6D input, 1D output. Modified MLP with multiplicative gating.
    """

    def __init__(self, hidden_dim: int = 192, num_layers: int = 6):
        super().__init__()
        input_dim = 6
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.U = nn.Linear(input_dim, hidden_dim)
        self.V = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        )
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


def compute_parametric_hjb_residual(
    model: nn.Module,
    tau: torch.Tensor,
    z: torch.Tensor,
    q_norm: torch.Tensor,
    sigma: torch.Tensor,
    kappa: torch.Tensor,
    gamma: torch.Tensor,
    q_max: int,
    T: float,
    A: float,
) -> torch.Tensor:
    """Compute HJB residual for the parametric PINN.

    All inputs are (N, 1) tensors. The PDE:
    (1/T)∂θ/∂τ + (σ²/2)∂²θ/∂z² - (γσ²/2)(∂θ/∂z)² + H = 0
    """
    dq = 1.0 / q_max

    inp = torch.cat([tau, z, q_norm, sigma, kappa, gamma], dim=1)
    inp.requires_grad_(True)

    theta = model(inp)

    # First derivatives
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
    sigma_sq = sigma ** 2
    time_term = (1.0 / T) * dtheta_dtau
    diffusion = (sigma_sq / 2) * d2theta_dz2
    risk = -(gamma * sigma_sq / 2) * dtheta_dz ** 2

    # Hamiltonian coefficient: A * (κ/(κ+γ))^(κ/γ) / (κ+γ)
    gk = gamma + kappa
    C = A * torch.exp((kappa / gamma) * torch.log(kappa / gk)) / gk

    # Evaluate θ at neighboring q values (detached to avoid coupling through autograd)
    q_plus = torch.clamp(q_norm + dq, -1.0, 1.0)
    q_minus = torch.clamp(q_norm - dq, -1.0, 1.0)
    inp_plus = torch.cat([tau, z, q_plus, sigma, kappa, gamma], dim=1)
    inp_minus = torch.cat([tau, z, q_minus, sigma, kappa, gamma], dim=1)

    with torch.no_grad():
        theta_plus = model(inp_plus)
        theta_minus = model(inp_minus)

    q_actual = q_norm * q_max
    can_buy = (q_actual < q_max - 0.5).float()
    can_sell = (q_actual > -q_max + 0.5).float()

    H = can_buy * C * torch.exp(-gamma * (theta - theta_plus))
    H += can_sell * C * torch.exp(-gamma * (theta - theta_minus))

    return time_term + diffusion + risk + H


def compute_parametric_terminal_loss(
    model: nn.Module,
    z_points: torch.Tensor,
    sigma: torch.Tensor,
    kappa: torch.Tensor,
    gamma: torch.Tensor,
    q_max: int,
) -> torch.Tensor:
    """Terminal condition loss at τ=1 for given parameters."""
    n_z = z_points.shape[0]
    total_loss = torch.tensor(0.0, device=z_points.device)

    for q in range(-q_max, q_max + 1):
        q_norm = torch.full((n_z, 1), q / q_max, device=z_points.device)
        tau_T = torch.ones(n_z, 1, device=z_points.device)
        inp = torch.cat([tau_T, z_points, q_norm, sigma.expand(n_z, 1),
                         kappa.expand(n_z, 1), gamma.expand(n_z, 1)], dim=1)

        theta_pred = model(inp)

        p = torch.sigmoid(z_points)
        if q == 0:
            theta_exact = torch.zeros_like(p)
        else:
            gamma_val = gamma[0].item() if gamma.dim() > 0 else gamma.item()
            inner = p * torch.exp(torch.tensor(-gamma_val * q)) + (1 - p)
            inner = torch.clamp(inner, min=1e-30)
            theta_exact = -(1.0 / gamma_val) * torch.log(inner)

        total_loss = total_loss + torch.mean((theta_pred - theta_exact) ** 2)

    return total_loss / (2 * q_max + 1)


def train_parametric_pinn(
    train_ds: FDDataset,
    holdout_ds: FDDataset | None = None,
    hidden_dim: int = 192,
    num_layers: int = 6,
    adam_epochs: int = 20_000,
    adam_lr: float = 1e-3,
    lbfgs_epochs: int = 200,
    supervised_batch: int = 4096,
    pde_batch: int = 2048,
    w_data: float = 1.0,
    w_pde: float = 0.1,
    w_terminal: float = 1.0,
    log_every: int = 1000,
    device: str = "cpu",
) -> tuple[nn.Module, list[dict]]:
    """Train the parametric PINN with hybrid data + PDE loss."""

    model = ParametricMMNetwork(hidden_dim, num_layers).to(device)
    q_max = train_ds.q_max
    T = train_ds.T
    A = train_ds.A

    # Parameter ranges for PDE collocation
    sigma_range = (train_ds.sigmas.min(), train_ds.sigmas.max())
    kappa_range = (train_ds.kappas.min(), train_ds.kappas.max())
    gamma_range = (train_ds.gammas.min(), train_ds.gammas.max())

    rng = np.random.default_rng(42)
    z_terminal = torch.linspace(-4.0, 4.0, 80, device=device).reshape(-1, 1)

    history = []

    # Phase 1: Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, adam_epochs)

    for epoch in range(adam_epochs):
        optimizer.zero_grad()

        # --- Supervised loss from FD data ---
        batch = train_ds.sample_supervised_batch(supervised_batch, rng)
        inp_data = torch.tensor(
            np.column_stack([batch["tau"], batch["z"], batch["q_norm"],
                             batch["sigma"], batch["kappa"], batch["gamma"]]),
            dtype=torch.float32, device=device,
        )
        theta_target = torch.tensor(batch["theta"], dtype=torch.float32, device=device).reshape(-1, 1)
        theta_pred = model(inp_data)
        loss_data = torch.mean((theta_pred - theta_target) ** 2)

        # --- PDE residual loss at random collocation points ---
        tau_pde = torch.rand(pde_batch, 1, device=device) * 0.98 + 0.01
        z_pde = torch.randn(pde_batch, 1, device=device) * 2.0
        q_pde = torch.rand(pde_batch, 1, device=device) * 2.0 - 1.0  # [-1, 1]
        sigma_pde = torch.rand(pde_batch, 1, device=device) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
        kappa_pde = torch.rand(pde_batch, 1, device=device) * (kappa_range[1] - kappa_range[0]) + kappa_range[0]
        gamma_pde = torch.exp(
            torch.rand(pde_batch, 1, device=device)
            * (np.log(gamma_range[1]) - np.log(gamma_range[0]))
            + np.log(gamma_range[0])
        )

        residual = compute_parametric_hjb_residual(
            model, tau_pde, z_pde, q_pde, sigma_pde, kappa_pde, gamma_pde,
            q_max, T, A,
        )
        loss_pde = torch.mean(residual ** 2)

        # --- Terminal loss at random parameters ---
        idx = rng.integers(train_ds.n_solutions)
        sigma_t = torch.tensor([[train_ds.sigmas[idx]]], dtype=torch.float32, device=device)
        kappa_t = torch.tensor([[train_ds.kappas[idx]]], dtype=torch.float32, device=device)
        gamma_t = torch.tensor([[train_ds.gammas[idx]]], dtype=torch.float32, device=device)
        loss_terminal = compute_parametric_terminal_loss(
            model, z_terminal, sigma_t, kappa_t, gamma_t, q_max,
        )

        loss = w_data * loss_data + w_pde * loss_pde + w_terminal * loss_terminal
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % log_every == 0 or epoch == adam_epochs - 1:
            entry = {
                "epoch": epoch,
                "loss": loss.item(),
                "data": loss_data.item(),
                "pde": loss_pde.item(),
                "terminal": loss_terminal.item(),
            }
            history.append(entry)
            print(f"  epoch {epoch:>6d}  loss={loss.item():.2e}  "
                  f"data={loss_data.item():.2e}  pde={loss_pde.item():.2e}  "
                  f"term={loss_terminal.item():.2e}", flush=True)

    # Phase 2: L-BFGS (supervised only — PDE is too expensive per closure)
    print("  Switching to L-BFGS (supervised refinement)...", flush=True)
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=20,
        history_size=50, line_search_fn="strong_wolfe",
    )

    for epoch in range(lbfgs_epochs):
        def closure():
            optimizer.zero_grad()
            batch = train_ds.sample_supervised_batch(supervised_batch, rng)
            inp = torch.tensor(
                np.column_stack([batch["tau"], batch["z"], batch["q_norm"],
                                 batch["sigma"], batch["kappa"], batch["gamma"]]),
                dtype=torch.float32, device=device,
            )
            target = torch.tensor(batch["theta"], dtype=torch.float32, device=device).reshape(-1, 1)
            pred = model(inp)
            loss = torch.mean((pred - target) ** 2)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        if epoch % 50 == 0:
            print(f"  L-BFGS {epoch}: loss={loss.item():.2e}", flush=True)

    return model, history


def validate_on_holdout(
    model: nn.Module,
    holdout_ds: FDDataset,
    device: str = "cpu",
) -> dict[str, float]:
    """Validate parametric PINN on held-out FD solutions."""
    model = model.to(device)
    model.eval()
    q_max = holdout_ds.q_max
    z = holdout_ds.z

    errors_per_solution = []

    for i in range(holdout_ds.n_solutions):
        sigma = holdout_ds.sigmas[i]
        kappa = holdout_ds.kappas[i]
        gamma = holdout_ds.gammas[i]
        sol = holdout_ds.solutions[i]

        max_rel_error = 0.0
        for q in range(-q_max, q_max + 1):
            theta_fd = sol[q][0]  # t=0 slice, shape (n_z,)
            n_z = len(z)

            q_norm = q / q_max
            inp = torch.tensor(
                np.column_stack([
                    np.zeros(n_z),  # τ=0
                    z,
                    np.full(n_z, q_norm),
                    np.full(n_z, sigma),
                    np.full(n_z, kappa),
                    np.full(n_z, gamma),
                ]),
                dtype=torch.float32, device=device,
            )

            with torch.no_grad():
                theta_pinn = model(inp).cpu().numpy().flatten()

            scale = np.max(np.abs(theta_fd)) + 1e-10
            rel_err = np.max(np.abs(theta_pinn - theta_fd)) / scale
            max_rel_error = max(max_rel_error, rel_err)

        errors_per_solution.append(max_rel_error)

    errors = np.array(errors_per_solution)
    return {
        "mean_error": float(errors.mean()),
        "max_error": float(errors.max()),
        "median_error": float(np.median(errors)),
        "pct_under_10pct": float((errors < 0.1).mean()),
        "pct_under_5pct": float((errors < 0.05).mean()),
        "n_holdout": len(errors),
    }
