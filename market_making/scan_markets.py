"""Scan Polymarket for market-making opportunities and estimate parameters."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pinn.polymarket import scan_markets, fetch_active_markets, fetch_order_book, fetch_trades, estimate_parameters


def main():
    print("Scanning Polymarket for market-making opportunities...")
    print("  Filter: liquidity $1K-$50K, spread ≥ 1.5¢\n")

    results = scan_markets(min_liquidity=1000, max_liquidity=50000, min_spread_cents=1.5)

    if not results:
        print("No markets found matching criteria. Trying broader search...")
        results = scan_markets(min_liquidity=500, max_liquidity=100000, min_spread_cents=1.0)

    print(f"\n{'='*80}")
    print(f"  MARKET-MAKING OPPORTUNITIES ({len(results)} found)")
    print(f"{'='*80}")

    for market, book, est in results:
        print(f"\n  {market.question[:70]}")
        print(f"  slug: {market.slug}")
        print(f"  Liquidity: ${market.liquidity:,.0f}  |  24h Volume: ${market.volume_24h:,.0f}")
        print(f"  Mid: {book.mid:.3f}  |  Spread: {book.spread:.3f} ({book.spread*100:.1f}¢)")
        bid_d, ask_d = book.depth_at(3)
        print(f"  Top-3 depth: {bid_d:.0f} bid / {ask_d:.0f} ask shares")
        print(f"  Estimated parameters:")
        print(f"    σ_logit (per min): {est.sigma_logit:.4f}")
        print(f"    Arrival rate: {est.arrival_rate:.2f} trades/min")
        print(f"    Avg trade size: {est.avg_trade_size:.1f} shares")
        print(f"    Trades in sample: {est.n_trades}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
