"""Phase 2: Parametric PINN — learn the solution family over κ.

A single network takes (τ, κ) as input and outputs X(τ; κ), the optimal
execution trajectory for any urgency parameter. This eliminates the need
to retrain when market conditions change — just feed in the new κ.

Training uses κ-curriculum: start with easy (low κ) collocation points,
progressively add harder (high κ) ones.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import numpy as np
from functools import partial

from pinn.network import MLP, HardConstrainedExecution
from pinn.physics import almgren_chriss_residual
from pinn.training import TrainConfig, make_collocation_points
from pinn.visualization import plot_kappa_family

OUTPUT_DIR = Path(__file__).parent / "output" / "phase2"

KAPPA_MIN = 0.1
KAPPA_MAX = 25.0
HORIZON = 1.0
Q = 1000.0


def make_parametric_collocation(n_tau: int, n_kappa: int, kappa_range: tuple[float, float]):
    """Create (τ, κ) collocation grid."""
    tau = make_collocation_points(n_tau, distribution="chebyshev").detach()
    kappas = torch.linspace(kappa_range[0], kappa_range[1], n_kappa)
    tau_grid = tau.repeat(n_kappa, 1)
    kappa_grid = kappas.repeat_interleave(n_tau).reshape(-1, 1)
    return torch.cat([tau_grid, kappa_grid], dim=1).requires_grad_(True)


def parametric_residual(tau_kappa: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """ODE residual for parametric PINN using per-point κ."""
    kappa = tau_kappa[:, 1:2]
    dX = torch.autograd.grad(
        X, tau_kappa, grad_outputs=torch.ones_like(X),
        create_graph=True, retain_graph=True,
    )[0][:, 0:1]

    d2X = torch.autograd.grad(
        dX, tau_kappa, grad_outputs=torch.ones_like(dX),
        create_graph=True, retain_graph=True,
    )[0][:, 0:1]

    kappa_T_sq = (kappa * HORIZON) ** 2
    return (1.0 / kappa_T_sq) * d2X - X


def train_stage(model, colloc, adam_epochs, lbfgs_epochs, lr=1e-3):
    """Train one curriculum stage. Returns loss history."""
    losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, adam_epochs)

    for epoch in range(adam_epochs):
        optimizer.zero_grad()
        X = model(colloc)
        loss = torch.mean(parametric_residual(colloc, X) ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        if epoch % 1000 == 0:
            print(f"    Adam {epoch:>5d}  loss = {loss.item():.2e}", flush=True)

    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=20,
        history_size=50, line_search_fn="strong_wolfe",
    )
    for epoch in range(lbfgs_epochs):
        def closure():
            optimizer.zero_grad()
            X = model(colloc)
            loss = torch.mean(parametric_residual(colloc, X) ** 2)
            loss.backward()
            return loss
        loss = optimizer.step(closure)
        losses.append(loss.item())

    print(f"    Final: {losses[-1]:.2e}", flush=True)
    return losses


def train_parametric():
    """Train the parametric PINN with staged κ-range expansion."""
    # Smaller network + fewer collocation points for CPU-friendly training
    base = MLP(input_dim=2, output_dim=1, hidden_dim=64, num_layers=4)
    model = HardConstrainedExecution(base, initial_inventory=Q)

    # 3-stage curriculum: expand κ range progressively
    # Each stage: (kappa_lo, kappa_hi, n_tau, n_kappa, adam_epochs, lbfgs_epochs)
    stages = [
        (KAPPA_MIN, 3.0, 80, 8, 3_000, 100),
        (KAPPA_MIN, 10.0, 80, 12, 3_000, 100),
        (KAPPA_MIN, KAPPA_MAX, 100, 15, 8_000, 300),
    ]

    all_losses = []
    for i, (k_lo, k_hi, n_tau, n_kappa, adam_ep, lbfgs_ep) in enumerate(stages):
        print(f"\n--- Stage {i+1}/{len(stages)}: κ ∈ [{k_lo}, {k_hi}], "
              f"{n_tau}×{n_kappa}={n_tau*n_kappa} points ---", flush=True)
        colloc = make_parametric_collocation(n_tau, n_kappa, (k_lo, k_hi))
        stage_losses = train_stage(model, colloc, adam_ep, lbfgs_ep)
        all_losses.extend(stage_losses)

    return model, all_losses


def evaluate(model):
    """Evaluate parametric PINN across κ values."""
    test_kappas = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    model.eval()

    print(f"\n{'='*60}")
    print("  PARAMETRIC PINN EVALUATION")
    print(f"{'='*60}")

    all_pass = True
    for kappa in test_kappas:
        tau = torch.linspace(0, 1, 500).reshape(-1, 1)
        kappa_col = torch.full_like(tau, kappa)
        inputs = torch.cat([tau, kappa_col], dim=1)

        with torch.no_grad():
            X_pinn = model(inputs)

        t = tau * HORIZON
        X_exact = Q * torch.sinh(kappa * (HORIZON - t)) / torch.sinh(torch.tensor(kappa * HORIZON))

        max_err = torch.max(torch.abs(X_pinn - X_exact)).item()
        rel_err = max_err / Q
        status = "PASS" if rel_err < 1e-2 else "FAIL"
        if rel_err >= 1e-2:
            all_pass = False
        print(f"  κ = {kappa:>5.1f}  max_err = {max_err:.2e}  rel_err = {rel_err:.2e}  [{status}]")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    def analytical_fn(tau_t, k):
        return Q * torch.sinh(k * (HORIZON - tau_t)) / torch.sinh(torch.tensor(k * HORIZON))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_kappa_family(
        model, analytical_fn,
        kappas=test_kappas,
        horizon=HORIZON,
        initial_inventory=Q,
        save_path=OUTPUT_DIR / "kappa_family.png",
    )
    return all_pass


def main():
    model, losses = train_parametric()
    success = evaluate(model)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_DIR / "parametric_pinn.pt")
    print(f"\n  Model saved to {OUTPUT_DIR / 'parametric_pinn.pt'}")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
