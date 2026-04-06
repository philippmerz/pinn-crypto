"""Backtest market-making strategies on real Polymarket data.

Compares PINN-based, FD-based, and naive market makers across
multiple prediction markets.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import numpy as np
import pandas as pd

from pinn.polymarket import fetch_active_markets, fetch_order_book, fetch_trades, estimate_parameters
from pinn.market_making import MarketMakerParams
from pinn.mm_backtest import (
    MMSimConfig,
    NaiveSymmetricMM,
    FDBasedMM,
    PINNBasedMM,
    simulate_market_making,
)
from pinn.parametric_mm_pinn import ParametricMMNetwork

OUTPUT_DIR = Path(__file__).parent / "output" / "backtest"
MODEL_DIR = Path(__file__).parent / "output" / "parametric"


def load_pinn_model() -> ParametricMMNetwork:
    meta = torch.load(MODEL_DIR / "metadata.pt", weights_only=False)
    model = ParametricMMNetwork(
        hidden_dim=meta["hidden_dim"],
        num_layers=meta["num_layers"],
    )
    model.load_state_dict(torch.load(MODEL_DIR / "parametric_mm_pinn.pt", weights_only=True))
    model.eval()
    return model


def run_backtest_on_market(
    market_slug: str,
    pinn_model: ParametricMMNetwork,
    horizon_minutes: float = 120.0,
) -> list[dict]:
    """Run all strategies on a single market."""
    print(f"\n  Fetching data for {market_slug}...")

    # Find market
    markets = fetch_active_markets(min_liquidity=500, max_liquidity=100000)
    market = next((m for m in markets if m.slug == market_slug), None)
    if market is None:
        print(f"    Market not found: {market_slug}")
        return []

    # Get data
    book = fetch_order_book(market.yes_token_id)
    trades = fetch_trades(market.condition_id, limit=500)
    if trades.empty or len(trades) < 20:
        print(f"    Not enough trades: {len(trades)}")
        return []

    est = estimate_parameters(trades, book)
    print(f"    Mid: {book.mid:.3f}  Spread: {book.spread*100:.1f}¢  "
          f"σ_logit: {est.sigma_logit:.3f}  Arrival: {est.arrival_rate:.2f}/min  "
          f"Trades: {est.n_trades}")

    sigma = np.clip(est.sigma_logit, 0.3, 3.0)
    gamma = 0.1

    # κ controls spread sensitivity: higher κ = tighter quotes.
    # Calibrate from observed spread: κ ≈ γ / (exp(γ * half_spread) - 1)
    observed_half_spread = est.spread_cents / 200  # cents → probability units
    if observed_half_spread > 0.005:
        kappa = min(150.0, gamma / (np.exp(gamma * observed_half_spread) - 1))
    else:
        kappa = 100.0
    kappa = max(kappa, 5.0)

    from math import log as mlog
    base_hs = (1.0 / gamma) * mlog(1.0 + gamma / kappa)
    print(f"    Calibrated: κ={kappa:.1f}  γ={gamma}  σ={sigma:.3f}  "
          f"base_half_spread={base_hs*100:.2f}¢  (PINN extrapolating: {kappa > 20})")

    config = MMSimConfig(initial_cash=20.0, q_max=5)

    # Build strategies
    params = MarketMakerParams(
        gamma=gamma, sigma=sigma, A=est.arrival_rate, kappa=kappa, q_max=5, T=1.0,
    )

    strategies = {
        "Naive_2c": NaiveSymmetricMM(half_spread=0.02),
        "Naive_3c": NaiveSymmetricMM(half_spread=0.03),
        "FD": FDBasedMM(params),
        "PINN": PINNBasedMM(pinn_model, q_max=5, sigma=sigma, kappa=kappa, gamma=gamma),
    }

    results = []
    for name, strategy in strategies.items():
        result = simulate_market_making(trades, strategy, config, horizon_minutes)
        result["market"] = market_slug
        result["sigma"] = sigma
        result["mid_price"] = book.mid
        result["spread_cents"] = book.spread * 100
        results.append(result)
        print(f"    {name:12s}: PnL=${result['final_pnl']:+.4f}  "
              f"fills={result['total_fills']:>3d} ({result['n_fills_bid']}b/{result['n_fills_ask']}a)  "
              f"inv={result['final_inventory']:+d}  mtm=${result['final_mtm']:.2f}")

    return results


def main():
    print("Loading PINN model...")
    pinn_model = load_pinn_model()

    # Target markets (illiquid with spreads)
    target_slugs = [
        "will-megaeth-perform-an-airdrop-by-june-30-143-229-513-574-212-254",
        "new-playboi-carti-album-before-gta-vi-421",
        "new-rhianna-album-before-gta-vi-926",
    ]

    all_results = []

    print(f"\n{'='*70}")
    print(f"  MARKET-MAKING BACKTEST ON POLYMARKET")
    print(f"{'='*70}")

    for slug in target_slugs:
        results = run_backtest_on_market(slug, pinn_model, horizon_minutes=120.0)
        all_results.extend(results)

    if not all_results:
        print("\nNo results. Markets may be inactive.")
        return 1

    # Summary
    df = pd.DataFrame(all_results)
    summary = df.groupby("strategy").agg(
        mean_pnl=("final_pnl", "mean"),
        total_fills=("total_fills", "sum"),
        mean_fills=("total_fills", "mean"),
        mean_inventory=("final_inventory", lambda x: x.abs().mean()),
        n_markets=("market", "count"),
    ).round(4)

    print(f"\n{'='*70}")
    print(f"  AGGREGATE RESULTS")
    print(f"{'='*70}")
    print(summary.to_string())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "mm_backtest_results.csv", index=False)
    print(f"\n  Saved to {OUTPUT_DIR / 'mm_backtest_results.csv'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
