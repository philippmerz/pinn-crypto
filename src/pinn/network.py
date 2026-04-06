"""Core PINN network architectures."""

import torch
import torch.nn as nn
from typing import Callable


class MLP(nn.Module):
    """Standard fully-connected network with smooth activations.

    Uses modified MLP pattern with input encoding on each layer
    (multiplicative gating) for improved gradient flow, following
    Wang et al. (2021) recommendations.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 5,
        activation: Callable[[], nn.Module] = nn.Tanh,
    ):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.activation = activation()

        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Input encoding branches for multiplicative gating
        self.U = nn.Linear(input_dim, hidden_dim)
        self.V = nn.Linear(input_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.activation(self.U(x))
        v = self.activation(self.V(x))

        h = self.activation(self.encoder(x))
        for layer in self.layers:
            h = self.activation(layer(h))
            # Multiplicative gating with input encoding
            h = h * u + (1 - h) * v

        return self.output_layer(h)


class HardConstrainedExecution(nn.Module):
    """Wraps a base network with hard boundary constraints for execution.

    Enforces X(0) = Q (initial inventory) and X(T) = 0 (full liquidation)
    architecturally, so the optimizer only needs to minimize the ODE residual.

    Uses a multiplicative exponential ansatz:
        X(τ) = Q * (1 - τ) * exp(NN(τ) * τ)

    At τ=0: X = Q * 1 * exp(0) = Q  ✓
    At τ=1: X = Q * 0 * exp(...) = 0  ✓

    This is far better than the additive ansatz Q*(1-τ) + τ*(1-τ)*NN(τ)
    for sharp/exponential solutions (high κ), because the exponential
    character is built into the ansatz structure. The network only needs
    to output a moderate scalar (~-20) rather than an extreme correction
    (~-12000), making optimization tractable for stiff problems.
    """

    def __init__(self, base_network: nn.Module, initial_inventory: float):
        super().__init__()
        self.base = base_network
        self.Q = initial_inventory

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """Forward pass with hard constraints.

        Args:
            tau: normalized time in [0, 1], shape (batch, input_dim)
                 First column must be the time coordinate.
        """
        t_coord = tau[:, 0:1]
        raw = self.base(tau)
        return self.Q * (1.0 - t_coord) * torch.exp(raw * t_coord)
