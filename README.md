# pinn-crypto

Physics-Informed Neural Networks for algorithmic trading in crypto markets.

## Premise

Financial markets are governed by approximate but useful differential equations — market-making HJB equations (Avellaneda-Stoikov), optimal execution ODEs (Almgren-Chriss), stochastic volatility SDEs (Heston), and more. Physics-Informed Neural Networks (PINNs) embed these equations directly into the neural network loss function, forcing the learned solution to respect known mathematical structure while fitting to market data.

## Current Focus: Market-Making PINN

Building an Avellaneda-Stoikov market-making PINN that uses the HJB equation for optimal bid/ask placement as a physics constraint. The model outputs optimal quotes given inventory, volatility, and order flow — applicable at any capital level, targeting illiquid crypto pairs where wider spreads compensate for inventory risk.

## Completed: Execution PINN

The `execution/` directory contains a completed proof-of-concept applying PINNs to the Almgren-Chriss optimal execution model. See [execution/README.md](execution/README.md) for full findings, including:

- PINN validated against analytical solution across all urgency regimes
- Parametric PINN learns entire solution family over κ in one network
- Multi-day backtest on 5 days of real BTC-USDT data (240 episodes)
- Key engineering lessons: exponential ansatz for stiff systems, residual normalization, κ-curriculum

**Conclusion:** The execution PINN correctly reproduces the analytical Almgren-Chriss solution but only applies at institutional order sizes where market impact is non-zero. For small accounts and illiquid pairs, market making is the right model.

## Reusable Components

```
src/pinn/
    network.py      — Modified MLP with multiplicative gating, hard constraint ansatz
    physics.py      — Financial differential equations and analytical solutions
    training.py     — Two-phase training (Adam → L-BFGS), parameter curriculum
    data.py         — Binance data pipeline, microstructure features, impact estimation
    visualization.py — Plotting utilities
    backtest.py     — Execution simulation and metrics
    strategies.py   — PINN-based and baseline execution strategies
tests/
    test_network.py    — Architecture and hard constraint tests
    test_physics.py    — Analytical solution and residual tests
    test_strategies.py — Trade sizing regression tests
execution/             — Completed execution PINN experiments and results
```

## Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.2
- NumPy, Matplotlib, Pandas
