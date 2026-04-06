# Market-Making PINN — Avellaneda-Stoikov on Polymarket

## Modified Avellaneda-Stoikov HJB for Prediction Markets

### Setup

A market maker quotes bid and ask prices on a binary outcome token with price p ∈ (0, 1). The market maker has:
- Cash position: x(t)
- Inventory: q(t) ∈ {-Q_max, ..., Q_max} (integer shares)
- Terminal time: T (market expiry / horizon)

The token settles at either $0 or $1 at expiry.

### Price Dynamics

The mid-price p(t) is bounded on (0, 1). We model it using a **logit-normal diffusion**:

Working in logit space: z = logit(p) = log(p / (1-p)), so p = sigmoid(z).

```
dz = σ dW
```

where σ is the volatility in logit space. This ensures p stays in (0, 1) since sigmoid maps ℝ → (0, 1).

By Itô's lemma, the price-space dynamics are:
```
dp = σ² p(1-p)(1-2p)/2 · dt + σ p(1-p) · dW
```

Key features:
- Volatility in price space is σ·p·(1-p), which vanishes at boundaries — correct
- There is a mean-reverting drift toward p=0.5 from the Itô correction — a modeling choice
- At p near 0 or 1, the process is "sticky" — also correct for prediction markets nearing resolution

### Order Arrival Model

Following A-S, market orders arrive as Poisson processes with intensity decreasing in distance from mid:
```
λ^a(δ^a) = A · exp(-κ · δ^a)    (buy orders hitting our ask)
λ^b(δ^b) = A · exp(-κ · δ^b)    (sell orders hitting our bid)
```

where δ^a, δ^b are the distances of our ask/bid from the mid-price, A is baseline intensity, and κ is the liquidity parameter.

**Polymarket modification**: distances are in probability space, not price space. Since the tick size is typically $0.01, our quotes snap to the nearest cent.

### Value Function and HJB Equation

The market maker maximizes expected terminal utility:
```
max E[-exp(-γ · (x_T + q_T · V_T))]
```

where V_T is the terminal settlement value:
- V_T = 1 if outcome is YES, V_T = 0 if outcome is NO
- E[V_T | p_T] ≈ p_T (the current price is the market's best estimate)

We work with the **indifference price** formulation. Define:
```
u(t, z, q) = -exp(-γ · (θ(t, z, q)))
```

where θ is the "certainty equivalent" value. The HJB for θ in logit coordinates:

```
∂θ/∂t + (σ²/2) · ∂²θ/∂z² - (γσ²/2) · (∂θ/∂z)² + H_bid + H_ask = 0
```

where the Hamiltonian terms from market making are:

```
H_bid = (A/κ) · exp(-γ·Δ_bid) · (1 - exp(-κ·δ*_bid))
H_ask = (A/κ) · exp(-γ·Δ_ask) · (1 - exp(-κ·δ*_ask))
```

with:
- Δ_bid = θ(t, z, q) - θ(t, z, q+1) - p + δ*_bid  (value change from buying at bid)
- Δ_ask = θ(t, z, q-1) - θ(t, z, q) - (1-p) + δ*_ask  (value change from selling at ask — noting symmetry)

Wait — let me be more careful. In the A-S framework with exponential utility, the optimal spreads satisfy:

```
δ*_bid = (1/κ) · log(1 + κ/γ) + (θ(t,z,q) - θ(t,z,q+1))
δ*_ask = (1/κ) · log(1 + κ/γ) + (θ(t,z,q) - θ(t,z,q-1))
```

The first term is the **base spread** (compensation for adverse selection), identical to standard A-S.
The second term is the **inventory skew** — how much to shift quotes based on current inventory.

### Terminal Condition

At expiry T, the market maker's terminal value per share depends on the outcome:
```
θ(T, z, q) = q · sigmoid(z)
```

This is the expected terminal value: each share is worth p = sigmoid(z), and you hold q shares.

For a risk-averse market maker, the certainty equivalent is:
```
θ(T, z, q) = -(1/γ) · log(E[exp(-γ · q · V_T)])
```

For binary V_T ∈ {0, 1} with P(V_T=1) = p = sigmoid(z):
```
θ(T, z, q) = -(1/γ) · log(p · exp(-γq) + (1-p))
```

### Reservation Price

The **reservation price** (indifference mid-price) is:
```
r(t, z, q) = sigmoid(z) + (θ(t,z,q+1) - θ(t,z,q-1)) / 2
```

Standard A-S gives r = s - q·γ·σ²·(T-t). Our version is more complex because:
- The sigmoid nonlinearity makes the relationship between z and p nonlinear
- The terminal condition is nonlinear in q (not quadratic as in standard A-S)
- The bounded price creates asymmetric risk

### Summary of PDE to Solve

**State**: (t, z, q) where t ∈ [0, T], z ∈ ℝ, q ∈ {-Q, ..., Q}

**PDE** (one equation per inventory level q):
```
∂θ/∂t + (σ²/2) · ∂²θ/∂z² - (γσ²/2) · (∂θ/∂z)²
    + A·exp(1-κ/γ)·[exp(-γ(θ_q - θ_{q+1})) + exp(-γ(θ_q - θ_{q-1}))] / (κ+γ) = 0
```

(Simplified form after substituting optimal spreads.)

**Terminal condition**:
```
θ(T, z, q) = -(1/γ) · log(sigmoid(z) · exp(-γq) + (1 - sigmoid(z)))
```

**Boundary in z**: As z → +∞ (p→1), θ → q (all shares worth $1). As z → -∞ (p→0), θ → 0.

**Boundary in q**: At q = ±Q_max, the corresponding buy/sell Hamiltonian term is zero (can't accumulate more).

### Parameters

| Parameter | Symbol | Typical Range (Polymarket) |
|-----------|--------|---------------------------|
| Risk aversion | γ | 0.01 – 1.0 |
| Logit-space volatility | σ | 0.1 – 5.0 (to be estimated) |
| Order arrival rate | A | 0.1 – 10 per minute |
| Liquidity parameter | κ | 1 – 100 |
| Max inventory | Q_max | 5 – 50 shares |
| Time horizon | T | hours to weeks |
