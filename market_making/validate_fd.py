"""Validate the FD solver for the modified A-S HJB.

Visualizes the value function, reservation prices, and optimal quotes
across different inventory levels and time-to-expiry.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from pinn.market_making import (
    MarketMakerParams,
    sigmoid,
    solve_fd,
    optimal_quotes,
)

OUTPUT_DIR = Path(__file__).parent / "output"


def plot_value_function(result: dict, save_path: Path):
    """Plot θ(t=0, z, q) for different inventory levels."""
    z = result["z"]
    p = sigmoid(z)
    theta = result["theta"]
    params = result["params"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # θ at t=0 vs price for different q
    ax = axes[0]
    for q in range(-params.q_max, params.q_max + 1):
        ax.plot(p, theta[q][0], label=f"q={q}", linewidth=1.5)
    ax.set_xlabel("Price p")
    ax.set_ylabel("θ(0, p, q)")
    ax.set_title("Value Function at t=0")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # θ at different times for q=0
    ax = axes[1]
    t = result["t"]
    n_times = len(t)
    indices = [0, n_times // 4, n_times // 2, 3 * n_times // 4, -1]
    for idx in indices:
        label = f"t={t[idx]:.2f}"
        ax.plot(p, theta[0][idx], label=label, linewidth=1.5)
    ax.set_xlabel("Price p")
    ax.set_ylabel("θ(t, p, 0)")
    ax.set_title("Value Function (q=0) Over Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Reservation price skew at t=0
    ax = axes[2]
    mid = sigmoid(z)
    for q in range(-params.q_max, params.q_max + 1):
        q_plus = min(q + 1, params.q_max)
        q_minus = max(q - 1, -params.q_max)
        reservation = mid + (theta[q_plus][0] - theta[q_minus][0]) / 2
        ax.plot(p, reservation - mid, label=f"q={q}", linewidth=1.5)
    ax.set_xlabel("Price p")
    ax.set_ylabel("Reservation - Mid")
    ax.set_title("Reservation Price Skew at t=0")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Modified A-S HJB (γ={params.gamma}, σ={params.sigma}, "
        f"A={params.A}, κ={params.kappa}, Q={params.q_max})",
        fontsize=11,
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_optimal_quotes(result: dict, save_path: Path):
    """Plot optimal bid/ask prices at t=0 for different inventory levels."""
    z = result["z"]
    p = sigmoid(z)
    theta = result["theta"]
    params = result["params"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for i, q in enumerate([-2, -1, 0, 1, 2, params.q_max]):
        if abs(q) > params.q_max:
            continue
        ax = axes[i // 3, i % 3]

        # Get theta dict at t=0
        theta_t0 = {qq: theta[qq][0] for qq in theta}
        bid, ask = optimal_quotes(theta_t0, z, q, params.q_max, params)

        ax.fill_between(p, bid, ask, alpha=0.2, color="blue", label="Spread")
        ax.plot(p, bid, "b-", linewidth=1.5, label="Bid")
        ax.plot(p, ask, "r-", linewidth=1.5, label="Ask")
        ax.plot(p, p, "k--", linewidth=1, alpha=0.5, label="Mid")
        ax.set_xlabel("Mid Price p")
        ax.set_ylabel("Quote Price")
        ax.set_title(f"q = {q}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.1, 0.9)
        ax.set_ylim(0, 1)

    fig.suptitle("Optimal Bid/Ask Quotes by Inventory Level", fontsize=12)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def print_summary(result: dict):
    """Print key properties of the solution."""
    z = result["z"]
    p = sigmoid(z)
    theta = result["theta"]
    params = result["params"]
    t = result["t"]

    mid_idx = len(z) // 2  # z≈0, p≈0.5
    print(f"\n{'='*60}")
    print(f"  FD SOLVER RESULTS")
    print(f"{'='*60}")
    print(f"  Grid: {len(t)} time steps × {len(z)} z points × {2*params.q_max+1} q levels")
    print(f"  z range: [{z[0]:.1f}, {z[-1]:.1f}] → p range: [{p[0]:.4f}, {p[-1]:.4f}]")
    print(f"  Base half-spread: {params.base_spread:.4f} ({params.base_spread*100:.1f}¢)")

    print(f"\n  θ(0, p=0.5, q) at t=0:")
    for q in range(-params.q_max, params.q_max + 1):
        print(f"    q={q:>3d}: θ = {theta[q][0, mid_idx]:.4f}")

    print(f"\n  Optimal quotes at t=0, p=0.5:")
    theta_t0 = {q: theta[q][0] for q in theta}
    for q in [-2, -1, 0, 1, 2]:
        if abs(q) <= params.q_max:
            bid, ask = optimal_quotes(theta_t0, z, q, params.q_max, params)
            spread = ask[mid_idx] - bid[mid_idx]
            skew = (ask[mid_idx] + bid[mid_idx]) / 2 - 0.5
            print(f"    q={q:>3d}: bid={bid[mid_idx]:.3f} ask={ask[mid_idx]:.3f} "
                  f"spread={spread:.3f} ({spread*100:.1f}¢) skew={skew:+.4f}")


def main():
    # Moderate parameters for initial validation
    params = MarketMakerParams(
        gamma=0.1,   # moderate risk aversion
        sigma=1.0,   # logit-space volatility
        A=2.0,       # ~2 orders per minute
        kappa=5.0,   # moderate spread sensitivity
        q_max=5,     # max ±5 shares
        T=1.0,       # 1 unit of time (e.g., 1 hour)
    )

    print(f"Solving modified A-S HJB...")
    print(f"  γ={params.gamma}, σ={params.sigma}, A={params.A}, κ={params.kappa}")
    print(f"  Q_max={params.q_max}, T={params.T}")

    result = solve_fd(params, n_z=201, n_t=2000, z_min=-6, z_max=6)

    print_summary(result)
    plot_value_function(result, OUTPUT_DIR / "value_function.png")
    plot_optimal_quotes(result, OUTPUT_DIR / "optimal_quotes.png")

    # Also test with higher risk aversion
    params_risk = MarketMakerParams(
        gamma=0.5, sigma=1.0, A=2.0, kappa=5.0, q_max=5, T=1.0,
    )
    print(f"\n\nSolving with higher risk aversion (γ={params_risk.gamma})...")
    result_risk = solve_fd(params_risk, n_z=201, n_t=2000, z_min=-6, z_max=6)
    print_summary(result_risk)
    plot_value_function(result_risk, OUTPUT_DIR / "value_function_high_gamma.png")
    plot_optimal_quotes(result_risk, OUTPUT_DIR / "optimal_quotes_high_gamma.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
