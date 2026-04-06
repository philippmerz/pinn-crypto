# pinn-crypto

Physics-Informed Neural Networks for algorithmic trading on prediction markets.

## Premise

Financial markets are governed by approximate but useful differential equations. Physics-Informed Neural Networks (PINNs) embed these equations into the loss function, forcing the learned solution to respect known mathematical structure while fitting to market data. This project tests whether PINNs provide a genuine edge over traditional numerical methods for market making on Polymarket.

## Current Status: Market-Making PINN on Polymarket

### What Works

**FD solver for the modified Avellaneda-Stoikov HJB.** The standard A-S model is modified for prediction markets: bounded [0,1] prices via logit transform, binary terminal settlement, and state-dependent volatility σ·p·(1-p). The FD solver produces correct quotes — symmetric at zero inventory, proper skew direction, and spreads that match real Polymarket markets when κ is calibrated from observed spreads. Solves in <1 second.

**Polymarket data pipeline.** Fetches order books, trades, and market metadata from the CLOB/Gamma/Data APIs. Estimates market-making parameters (logit-space volatility, trade arrival rates, fill rates). Scanner identifies 6 target markets with 2-4¢ spreads and thin depth — our target operating environment.

**Parametric PINN (narrow κ range).** A 6D network (τ, z, q, σ, κ, γ) → θ trained with hybrid supervised + PDE loss achieves 77% of holdout parameter combinations under 10% error when κ ∈ [3, 20]. This validated the cross-market approach and showed the hybrid loss fixes the level-shift problem from the single-market PINN.

### What Doesn't Work (Yet)

**Parametric PINN at realistic κ.** Real Polymarket spreads require κ = 50-150, but extending the PINN's trained range from [3, 20] to [5, 150] collapses holdout accuracy to ~0%. The value function θ changes character dramatically across this range, and the network can't represent both regimes. Increasing supervised data (up to 480 FD solutions) and reducing PDE weight helps training loss but not holdout generalization.

**The core tension.** The modified HJB has Hamiltonian coupling between inventory levels — θ at q±1 appears in the equation for θ at q. This means the unsupervised PDE residual can be satisfied by level-shifted solutions that are wrong in absolute value but correct in structure. Supervised data from FD solutions fixes the drift, but at the cost of needing many FD solves, which undermines the PINN's advantage over FD.

### Next Steps

**Path B: Learn quote spreads, not the value function.** Instead of learning θ(t,z,q) directly (which suffers from level-shift), learn δ_bid and δ_ask — the optimal quote distances from mid-price. These depend on θ *differences* between inventory levels, which the single-market PINN got right (2% error at extreme q). The PDE constraint would operate on reconstructed θ but the network output is the operationally useful quantity. This sidesteps the level-shift problem at the formulation level.

## Completed: Execution PINN

The `execution/` directory contains a proof-of-concept applying PINNs to the Almgren-Chriss optimal execution model on Binance BTC-USDT. See [execution/README.md](execution/README.md) for findings.

Key results:
- PINN validated against analytical solution across all urgency regimes (< 1e-5 relative error)
- Parametric PINN learns entire solution family over κ in one network
- Multi-day backtest on 5 days of real BTC-USDT data (240 episodes) — PINN matches analytical AC exactly, as expected

**Conclusion:** The execution PINN correctly reproduces the analytical solution but only applies at institutional order sizes. For small accounts on prediction markets, market making is the right model.

## Key Engineering Lessons

1. **Exponential ansatz for stiff systems.** The standard additive PINN ansatz fails at high κ. A multiplicative exponential ansatz X(τ) = Q·(1-τ)·exp(NN(τ)·τ) naturally represents sharp decay and solved the stiff-system failure.

2. **Residual normalization.** Dividing the ODE residual by (κT)² keeps the loss O(1) regardless of the urgency parameter. Without this, loss scales with κ² and the optimizer can't make progress.

3. **κ-curriculum (homotopy continuation).** Training at high κ directly fails. Geometrically ramping from low to high κ works reliably — each stage's solution initializes the next.

4. **Hybrid data + PDE loss.** The HJB's Hamiltonian coupling creates a level-shift degeneracy that pure PDE loss can't resolve. Supervised data from FD solutions anchors absolute values. But this creates a tension: more data → less PINN advantage.

5. **Strategy scaling bugs are silent and flattering.** A unit mismatch in the execution PINN caused "18x variance reduction" that was actually zero market exposure. Always verify with regression tests.

## Project Structure

```
src/pinn/
    market_making.py       — Modified A-S HJB: FD solver, terminal conditions, quote extraction
    mm_pinn.py             — Single-market PINN for the HJB
    parametric_mm_pinn.py  — Cross-market parametric PINN (6D input)
    mm_backtest.py         — Market-making simulation engine
    polymarket.py          — Polymarket API: markets, order books, trades, parameter estimation
    fd_dataset.py          — FD training data generation across parameter grid
    network.py             — Modified MLP with multiplicative gating, hard constraint ansatz
    physics.py             — Almgren-Chriss ODE, analytical solutions
    training.py            — Two-phase training (Adam → L-BFGS), parameter curriculum
    data.py                — Binance data pipeline
    backtest.py            — Execution simulation and metrics
    strategies.py          — PINN-based and baseline execution strategies
    visualization.py       — Plotting utilities
market_making/             — MM experiments, FD validation, PINN training, backtests
execution/                 — Completed execution PINN experiments
tests/                     — Unit tests for all modules
```

## Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.2
- NumPy, Matplotlib, Pandas
- py-clob-client (for Polymarket API)
