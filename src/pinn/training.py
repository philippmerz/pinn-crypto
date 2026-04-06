"""PINN training utilities."""

import torch
import torch.nn as nn
from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    n_collocation: int = 300
    adam_epochs: int = 15_000
    adam_lr: float = 1e-3
    lbfgs_epochs: int = 500
    lbfgs_lr: float = 1.0
    lbfgs_max_iter: int = 20
    log_every: int = 1000
    device: str = "cpu"


@dataclass
class TrainResult:
    """Training outcome."""

    losses: list[float] = field(default_factory=list)
    final_loss: float = float("inf")


def make_collocation_points(
    n: int,
    device: str = "cpu",
    distribution: str = "chebyshev",
) -> torch.Tensor:
    """Generate collocation points in [0, 1].

    Chebyshev nodes cluster near boundaries where
    the Almgren-Chriss solution has most curvature at high κ.
    """
    if distribution == "chebyshev":
        i = torch.arange(n, device=device, dtype=torch.float32)
        tau = 0.5 * (1.0 - torch.cos(torch.pi * i / (n - 1)))
    elif distribution == "uniform":
        tau = torch.linspace(0.0, 1.0, n, device=device)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return tau.reshape(-1, 1).requires_grad_(True)


def train_pinn(
    model: nn.Module,
    residual_fn,
    config: TrainConfig = TrainConfig(),
    callback=None,
) -> TrainResult:
    """Two-phase PINN training: Adam for exploration, L-BFGS for refinement.

    Args:
        model: the constrained network (output is inventory X(τ))
        residual_fn: callable(tau, X) -> residual tensor
        config: training hyperparameters
        callback: optional callable(epoch, loss) for logging

    Returns:
        TrainResult with loss history.
    """
    result = TrainResult()
    device = config.device
    model = model.to(device)

    tau = make_collocation_points(config.n_collocation, device)

    # Phase 1: Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=config.adam_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.adam_epochs)

    for epoch in range(config.adam_epochs):
        optimizer.zero_grad()
        X = model(tau)
        residual = residual_fn(tau, X)
        loss = torch.mean(residual**2)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        result.losses.append(loss_val)
        if callback and (epoch % config.log_every == 0 or epoch == config.adam_epochs - 1):
            callback(epoch, loss_val)

    # Phase 2: L-BFGS
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=config.lbfgs_lr,
        max_iter=config.lbfgs_max_iter,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    for epoch in range(config.lbfgs_epochs):

        def closure():
            optimizer.zero_grad()
            X = model(tau)
            residual = residual_fn(tau, X)
            loss = torch.mean(residual**2)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        loss_val = loss.item()
        result.losses.append(loss_val)
        if callback and (epoch % 50 == 0 or epoch == config.lbfgs_epochs - 1):
            callback(config.adam_epochs + epoch, loss_val)

    result.final_loss = result.losses[-1]
    return result


def train_with_curriculum(
    model: nn.Module,
    make_residual_fn,
    kappa_schedule: list[float],
    config: TrainConfig = TrainConfig(),
    callback=None,
) -> TrainResult:
    """Train via κ-curriculum (homotopy continuation).

    Starts with an easy (low κ) problem and gradually increases to the
    target κ. Each stage uses the previous solution as initialization,
    smoothly guiding the network through the loss landscape.

    This is the standard fix for stiff-system failures in PINNs,
    equivalent to parameter continuation in numerical PDE literature.

    Args:
        model: the constrained network
        make_residual_fn: callable(kappa) -> residual_fn(tau, X)
        kappa_schedule: list of κ values, ascending to target
        config: training hyperparameters (applied per stage)
        callback: optional callable(epoch, loss, kappa)
    """
    result = TrainResult()
    global_epoch = 0

    for stage, kappa in enumerate(kappa_schedule):
        # Fewer epochs for early (easy) stages, full budget for final
        is_final = stage == len(kappa_schedule) - 1
        stage_adam = config.adam_epochs if is_final else config.adam_epochs // 3
        stage_lbfgs = config.lbfgs_epochs if is_final else config.lbfgs_epochs // 3

        stage_config = TrainConfig(
            n_collocation=config.n_collocation,
            adam_epochs=stage_adam,
            adam_lr=config.adam_lr,
            lbfgs_epochs=stage_lbfgs,
            log_every=config.log_every,
            device=config.device,
        )

        residual_fn = make_residual_fn(kappa)

        def stage_callback(epoch, loss):
            if callback:
                callback(global_epoch + epoch, loss, kappa)

        stage_result = train_pinn(model, residual_fn, stage_config, stage_callback)
        result.losses.extend(stage_result.losses)
        global_epoch += stage_adam + stage_lbfgs

        if callback:
            callback(global_epoch, stage_result.final_loss, kappa)

    result.final_loss = result.losses[-1]
    return result
