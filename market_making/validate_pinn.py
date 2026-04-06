"""Validate PINN against FD solver for the market-making HJB."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from pinn.market_making import MarketMakerParams, solve_fd, sigmoid
from pinn.mm_pinn import train_mm_pinn, compare_with_fd, MMPINNConfig

OUTPUT_DIR = Path(__file__).parent / "output"


def main():
    # Use moderate parameters where FD is known to work well
    params = MarketMakerParams(
        gamma=0.1, sigma=1.0, A=2.0, kappa=5.0, q_max=3, T=1.0,
    )

    print("Step 1: Solving with FD (ground truth)...")
    fd_result = solve_fd(params, n_z=201, n_t=2000, z_min=-5, z_max=5)
    print(f"  Grid: {len(fd_result['t'])} × {len(fd_result['z'])} × {2*params.q_max+1}")

    print("\nStep 2: Training PINN...")
    config = MMPINNConfig(
        n_colloc_tau=20,
        n_colloc_z=40,
        n_colloc_q=7,  # 2*q_max + 1
        n_terminal_z=80,
        adam_epochs=10_000,
        adam_lr=1e-3,
        lbfgs_epochs=200,
        w_pde=1.0,
        w_terminal=10.0,
        log_every=2000,
    )

    def callback(epoch, loss, loss_pde, loss_terminal):
        print(f"  epoch {epoch:>6d}  loss={loss:.2e}  pde={loss_pde:.2e}  terminal={loss_terminal:.2e}")

    model, losses = train_mm_pinn(params, config, callback)

    print(f"\n  Final loss: {losses[-1]:.2e}")

    print("\nStep 3: Comparing PINN vs FD...")
    errors = compare_with_fd(model, fd_result, params)

    print(f"\n{'='*50}")
    print(f"  PINN vs FD COMPARISON (t=0)")
    print(f"{'='*50}")

    all_pass = True
    for q in sorted(errors.keys()):
        status = "PASS" if errors[q] < 0.1 else "FAIL"
        if errors[q] >= 0.1:
            all_pass = False
        print(f"  q={q:>3d}: max_rel_error = {errors[q]:.4f}  [{status}]")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    # Save model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_DIR / "mm_pinn.pt")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
