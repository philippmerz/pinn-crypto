# pinn-crypto

Physics-Informed Neural Networks for algorithmic trading in crypto markets.

## Premise

Financial markets are governed by approximate but useful differential equations — optimal execution ODEs (Almgren-Chriss), market-making HJB equations (Avellaneda-Stoikov), stochastic volatility SDEs (Heston), and more. Physics-Informed Neural Networks (PINNs) embed these equations directly into the neural network loss function, forcing the learned solution to respect known mathematical structure while fitting to market data.

This project applies PINNs to crypto execution and trading, starting with the Almgren-Chriss optimal execution model on Binance BTC-USDT data.

## Architecture

The core PINN encodes financial differential equations as soft constraints via automatic differentiation:

```
L_total = w_ode * L_ode_residual + w_data * L_market_data
```

where the ODE residual is computed by differentiating the network output with respect to its inputs using PyTorch autograd.

**Hard boundary constraints** are enforced via a multiplicative exponential ansatz — `X(τ) = Q·(1-τ)·exp(NN(τ)·τ)` — guaranteeing X(0) = Q and X(T) = 0 for any network weights. This naturally handles sharp/exponential decay profiles (high urgency), which additive ansatze fail on.

**Training** uses a two-phase protocol: Adam for initial exploration, then L-BFGS for fine-tuning (achieves orders of magnitude lower loss). For stiff problems (high κ), κ-curriculum (homotopy continuation) progressively increases difficulty.

## Implementation Phases

### Phase 1: Validate PINN on Almgren-Chriss ODE ✅
Solve `d²X/dt² - κ²X = 0` and validate against the analytical `sinh` solution across different urgency parameters (κ). All three scenarios (low/medium/high urgency) pass with < 1e-5 relative error. Key engineering: exponential ansatz solved the stiff-system failure at high κ.

### Phase 2: Parametric PINN ✅
A single network takes `(τ, κ)` as input and learns the entire solution family. Achieves < 0.4% error for κ ≥ 2 (the practically relevant regime where execution dynamics are non-trivial). Trained via κ-curriculum (homotopy continuation).

### Phase 3: Binance Data Pipeline ✅
Downloads BTC-USDT tick trades from `data.binance.vision`. Estimates market impact parameters: Kyle's λ, realized volatility, temporary/permanent impact coefficients. Tested on ~1.7M trades/day.

### Phase 4: Market-State-Dependent Execution
Add real market features (realized volatility, spread, volume ratio, book imbalance) as inputs. Train on historical Binance BTC-USDT episodes.

### Phase 5: Backtesting
Walk-forward evaluation against TWAP, VWAP, and analytical Almgren-Chriss. Metrics: implementation shortfall, execution Sharpe, tail risk.

## Candidate Financial Equations

| Equation | Type | Application |
|----------|------|-------------|
| Almgren-Chriss ODE | Euler-Lagrange | Optimal execution trajectory |
| HJB (execution) | Nonlinear PDE | Value function for optimal trading |
| Avellaneda-Stoikov | HJB | Market making (bid/ask placement) |
| Black-Scholes | Linear PDE | Option pricing / hedging |
| Fokker-Planck | Linear PDE | Return distribution evolution |
| Heston | 2D PDE | Stochastic volatility dynamics |

## Project Structure

```
src/pinn/
    network.py      — MLP with modified architecture, hard boundary constraints
    physics.py      — Financial differential equations and analytical solutions
    training.py     — Two-phase training loop (Adam → L-BFGS), κ-curriculum
    data.py         — Binance data pipeline, microstructure features, impact estimation
    visualization.py — Trajectory, loss, and parameter sweep plots
experiments/
    phase1_validate.py  — Phase 1: validate against analytical solution
    phase2_parametric.py — Phase 2: parametric PINN over κ
    fetch_data.py       — Download and analyze Binance trade data
tests/
    test_network.py — Architecture and hard constraint tests
    test_physics.py — Analytical solution and ODE residual tests
```

## Usage

```bash
# Install
pip install -e .

# Run tests
pytest tests/ -v

# Run Phase 1 validation
python experiments/phase1_validate.py

# Run Phase 2 parametric PINN
python experiments/phase2_parametric.py

# Download and analyze Binance data
python experiments/fetch_data.py
```

## Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.2
- NumPy, Matplotlib, Pandas
