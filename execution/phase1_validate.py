"""Phase 1: Validate PINN against analytical Almgren-Chriss solution.

This is the foundational sanity check: can the PINN learn the solution to
d²X/dt² - κ²X = 0 with X(0)=Q, X(T)=0?

If this fails, nothing downstream will work.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from functools import partial

from pinn.network import MLP, HardConstrainedExecution
from pinn.physics import AlmgrenChrissParams, almgren_chriss_analytical, almgren_chriss_residual
from pinn.training import TrainConfig, train_pinn, train_with_curriculum
from pinn.visualization import plot_trajectory_comparison, plot_loss_curve

OUTPUT_DIR = Path(__file__).parent / "output" / "phase1"

# κ threshold above which curriculum training is used
CURRICULUM_THRESHOLD = 5.0


def make_residual_fn(kappa: float, horizon: float):
    """Create a residual function for a given κ."""
    return partial(almgren_chriss_residual, kappa=kappa, horizon=horizon)


def run_validation(params: AlmgrenChrissParams, tag: str):
    """Train and validate a single PINN against analytical solution."""
    print(f"\n{'='*60}")
    print(f"  {tag}")
    print(f"  κ = {params.kappa:.3f}, Q = {params.initial_inventory}, T = {params.horizon}")
    print(f"{'='*60}")

    base = MLP(input_dim=1, output_dim=1, hidden_dim=128, num_layers=5)
    model = HardConstrainedExecution(base, initial_inventory=params.initial_inventory)

    config = TrainConfig(
        n_collocation=300,
        adam_epochs=10_000,
        adam_lr=1e-3,
        lbfgs_epochs=300,
        log_every=2000,
    )

    if params.kappa > CURRICULUM_THRESHOLD:
        # κ-curriculum: ramp from κ=1 to target in geometric steps
        import numpy as np

        n_stages = max(3, int(np.log2(params.kappa)))
        kappa_schedule = np.geomspace(1.0, params.kappa, n_stages).tolist()
        print(f"  Using κ-curriculum: {[f'{k:.1f}' for k in kappa_schedule]}")

        result = train_with_curriculum(
            model,
            make_residual_fn=lambda k: make_residual_fn(k, params.horizon),
            kappa_schedule=kappa_schedule,
            config=config,
            callback=lambda epoch, loss, kappa: print(
                f"  epoch {epoch:>6d}  κ={kappa:>6.1f}  loss = {loss:.2e}"
            )
            if epoch % config.log_every == 0
            else None,
        )
    else:
        residual_fn = make_residual_fn(params.kappa, params.horizon)
        result = train_pinn(
            model,
            residual_fn,
            config,
            callback=lambda epoch, loss: print(f"  epoch {epoch:>6d}  loss = {loss:.2e}"),
        )

    # Evaluate
    model.eval()
    tau_test = torch.linspace(0, 1, 1000).reshape(-1, 1)
    with torch.no_grad():
        X_pinn = model(tau_test)

    t_test = tau_test * params.horizon
    X_exact = almgren_chriss_analytical(t_test, params)
    max_error = torch.max(torch.abs(X_pinn - X_exact)).item()
    rel_error = max_error / params.initial_inventory

    print(f"\n  Final loss:     {result.final_loss:.2e}")
    print(f"  Max abs error:  {max_error:.2e}")
    print(f"  Max rel error:  {rel_error:.2e}")
    print(f"  Status:         {'PASS' if rel_error < 1e-3 else 'FAIL'}")

    # Save plots
    analytical_fn = lambda tau: almgren_chriss_analytical(tau * params.horizon, params)
    plot_trajectory_comparison(model, analytical_fn, title=tag, save_path=OUTPUT_DIR / f"{tag}.png")
    plot_loss_curve(
        result.losses,
        adam_epochs=len(result.losses) - config.lbfgs_epochs,
        save_path=OUTPUT_DIR / f"{tag}_loss.png",
    )

    return rel_error


def main():
    scenarios = [
        ("low_urgency", AlmgrenChrissParams(
            initial_inventory=1000.0, horizon=1.0, sigma=0.02,
            eta=0.01, theta=0.005, lambda_risk=1e-4, s0=100.0,
        )),
        ("medium_urgency", AlmgrenChrissParams(
            initial_inventory=1000.0, horizon=1.0, sigma=0.02,
            eta=0.01, theta=0.005, lambda_risk=1e-2, s0=100.0,
        )),
        ("high_urgency", AlmgrenChrissParams(
            initial_inventory=1000.0, horizon=1.0, sigma=0.02,
            eta=0.01, theta=0.005, lambda_risk=1.0, s0=100.0,
        )),
    ]

    results = {}
    for tag, params in scenarios:
        rel_error = run_validation(params, tag)
        results[tag] = rel_error

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for tag, err in results.items():
        status = "PASS" if err < 1e-3 else "FAIL"
        if err >= 1e-3:
            all_pass = False
        print(f"  {tag:20s}  rel_error = {err:.2e}  [{status}]")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
