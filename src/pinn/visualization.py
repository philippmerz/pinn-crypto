"""Plotting utilities for PINN results."""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_trajectory_comparison(
    model,
    analytical_fn,
    title: str = "PINN vs Analytical Almgren-Chriss",
    n_points: int = 500,
    save_path: Path | None = None,
):
    """Plot learned trajectory against analytical solution."""
    model.eval()
    tau = torch.linspace(0, 1, n_points).reshape(-1, 1)

    with torch.no_grad():
        X_pinn = model(tau).numpy()

    X_analytical = analytical_fn(tau).numpy()
    t_np = tau.numpy().flatten()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Trajectory comparison
    ax = axes[0]
    ax.plot(t_np, X_analytical.flatten(), "k-", linewidth=2, label="Analytical")
    ax.plot(t_np, X_pinn.flatten(), "r--", linewidth=2, label="PINN")
    ax.set_xlabel("τ (normalized time)")
    ax.set_ylabel("X(τ) (remaining inventory)")
    ax.set_title("Optimal Execution Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Pointwise error
    ax = axes[1]
    error = np.abs(X_pinn.flatten() - X_analytical.flatten())
    ax.semilogy(t_np, error)
    ax.set_xlabel("τ")
    ax.set_ylabel("|X_pinn - X_analytical|")
    ax.set_title(f"Absolute Error (max={error.max():.2e})")
    ax.grid(True, alpha=0.3)

    # Trading rate (negative derivative of inventory)
    ax = axes[2]
    dt = t_np[1] - t_np[0]
    rate_analytical = -np.gradient(X_analytical.flatten(), dt)
    rate_pinn = -np.gradient(X_pinn.flatten(), dt)
    ax.plot(t_np, rate_analytical, "k-", linewidth=2, label="Analytical")
    ax.plot(t_np, rate_pinn, "r--", linewidth=2, label="PINN")
    ax.set_xlabel("τ")
    ax.set_ylabel("v(τ) (trading rate)")
    ax.set_title("Optimal Trading Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_loss_curve(
    losses: list[float],
    adam_epochs: int,
    save_path: Path | None = None,
):
    """Plot training loss with Adam/L-BFGS phase boundary."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(losses)
    ax.axvline(x=adam_epochs, color="gray", linestyle="--", alpha=0.5, label="Adam → L-BFGS")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared ODE Residual")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_kappa_family(
    model,
    analytical_fn,
    kappas: list[float],
    horizon: float,
    initial_inventory: float,
    n_points: int = 300,
    save_path: Path | None = None,
):
    """Plot solution family across different urgency parameters."""
    model.eval()
    tau = torch.linspace(0, 1, n_points).reshape(-1, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(kappas)))

    for i, kappa in enumerate(kappas):
        kappa_input = torch.full((n_points, 1), kappa)
        model_input = torch.cat([tau, kappa_input], dim=1)

        with torch.no_grad():
            X_pinn = model(model_input).numpy().flatten()

        t_real = tau.numpy().flatten() * horizon
        X_analytical = analytical_fn(tau * horizon, kappa).numpy().flatten()

        axes[0].plot(t_real, X_pinn, color=cmap[i], linewidth=2, label=f"κ={kappa:.1f}")
        axes[1].plot(
            t_real,
            np.abs(X_pinn - X_analytical),
            color=cmap[i],
            linewidth=1.5,
        )

    axes[0].set_xlabel("t")
    axes[0].set_ylabel("X(t) (remaining inventory)")
    axes[0].set_title("Learned Execution Trajectories")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("t")
    axes[1].set_ylabel("|error|")
    axes[1].set_title("Absolute Error vs Analytical")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Parametric PINN: Solution Family over κ", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig
