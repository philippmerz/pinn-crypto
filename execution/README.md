# Execution PINN — Almgren-Chriss Optimal Execution

## Summary

A physics-informed neural network that solves the Almgren-Chriss optimal execution ODE:

$$
\frac{d^2 X}{dt^2} - \kappa^2 X(t) = 0, \qquad X(0) = Q, \qquad X(T) = 0
$$

where $X(t)$ is remaining inventory and $\kappa$ is the urgency parameter balancing market impact against timing risk.

## What We Built

1. **Phase 1 PINN** — Solves the ODE at fixed κ. Validated against the analytical sinh solution with <1e-5 relative error across low/medium/high urgency regimes.

2. **Phase 2 Parametric PINN** — Takes (τ, κ) as input, learns the entire solution family in one network. <0.4% error for κ ≥ 2.

3. **Binance Data Pipeline** — Downloads BTC-USDT tick trades from `data.binance.vision`, estimates market impact parameters (Kyle's λ, temporary/permanent impact, realized volatility).

4. **Backtest Framework** — Walk-forward execution simulation with transient impact model, compared against TWAP, VWAP, and analytical Almgren-Chriss.

## Key Engineering Findings

### Exponential ansatz solves stiff-system failures

The standard PINN ansatz

$$
X(\tau) = Q(1 - \tau) + \tau(1 - \tau)\,\mathrm{NN}(\tau)
$$

fails at high $\kappa$ (stiff ODE) because the correction term has minimal flexibility near $\tau = 0$ where the sharp decay happens.

The fix is a multiplicative exponential ansatz:

$$
X(\tau) = Q(1 - \tau)\exp\left(\mathrm{NN}(\tau)\tau\right)
$$

It naturally represents exponential-like decay. The network only needs to output a moderate scalar ($\approx -20$) rather than an extreme correction ($\approx -12000$).

### Residual normalization is essential

The unnormalized ODE residual

$$
\frac{d^2 X}{dt^2} - \kappa^2 X
$$

scales with $\kappa^2$, making the loss $\approx 10^{10}$ at $\kappa = 20$. Dividing by $(\kappa T)^2$ normalizes the residual to $O(1)$ regardless of $\kappa$.

### κ-curriculum (homotopy continuation)

Training at high κ directly fails due to poor loss landscape. Starting at low κ and geometrically ramping to the target works reliably — each stage's solution initializes the next.

### Strategy scaling bugs are silent and flattering

A unit mismatch between normalized PINN output (Q=1) and actual trade sizes (0.5 BTC) caused the strategy to dump all inventory in the first interval. The resulting "18x variance reduction" was zero market exposure, not skill. Added regression tests: total execution must match target, no single interval >50%.

## Backtest Results (5 days, 240 episodes)

BTC-USDT on Binance, Sep–Nov 2025, 30-min windows, 0.5 BTC per window.

| Strategy | Mean IS (bps) | Std IS (bps) | Exec Sharpe |
|----------|:---:|:---:|:---:|
| TWAP | 1.50 | 13.38 | -0.11 |
| AC/PINN κ=1 | 1.56 | 12.58 | -0.13 |
| AC/PINN κ=12 | 1.61 | 4.33 | -0.38 |
| AC/PINN κ=8 | 1.75 | 5.64 | -0.32 |

- **PINN matches analytical AC exactly** at every κ (correct — same equation, validated)
- **TWAP wins mean IS** but has the highest variance
- **κ=12 is the risk-adjusted sweet spot**: +0.11 bps vs TWAP but 3x lower variance
- **Optimal κ is regime-dependent**: high-vol days favor aggressive execution

## Limitations and Honest Assessment

### The model doesn't apply at small capital

Almgren-Chriss optimizes the tradeoff between market impact and timing risk. At small order sizes (e.g., $20), market impact is zero and the optimal strategy trivially degenerates to "just execute." The model is only relevant when order size is meaningful relative to available liquidity (roughly >0.1% of ADV, which for BTC-USDT is >$600K).

### The model doesn't apply to illiquid pairs

On illiquid pairs with gapped order books, vanishing liquidity, and wide spreads:
- Impact is a step function of book state, not a linear function of trade rate
- Spread crossing cost dominates impact cost
- You can't trade continuously — execution is event-driven
- Adverse selection is severe

The AC model's core assumptions are structurally violated, not just stretched.

### Fixed-κ PINN = analytical solution

A PINN at fixed κ with only the ODE constraint (no data loss) learns exactly the analytical solution. It adds no value over the closed-form. The PINN's potential edge is in:
- **Adaptive κ** via the parametric PINN (Phase 2) responding to market state
- **Non-linear impact** where no closed-form exists (square-root law)
- **Multi-asset execution** where the coupled ODEs have no clean analytical solution

## What's Reusable

The following components transfer directly to other PINN applications:

- `src/pinn/network.py` — Modified MLP with multiplicative gating, hard constraint ansatz
- `src/pinn/training.py` — Two-phase Adam→L-BFGS training, κ-curriculum
- `src/pinn/data.py` — Binance data pipeline
- `src/pinn/backtest.py` — Execution simulation and metrics framework
- `src/pinn/visualization.py` — Plotting utilities
