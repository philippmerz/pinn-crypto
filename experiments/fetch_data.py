"""Download and analyze BTC-USDT trade data from Binance.

Downloads a sample of recent daily trade files and estimates
market impact parameters for Almgren-Chriss calibration.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pinn.data import download_date_range, load_trades, estimate_impact_params, compute_microstructure

OUTPUT_DIR = Path(__file__).parent / "output" / "data"


def main():
    symbol = "BTCUSDT"

    # Download 3 recent days as a sample
    end_date = date(2025, 12, 1)
    start_date = end_date - timedelta(days=2)

    print(f"Downloading {symbol} trades: {start_date} to {end_date}")
    paths = download_date_range(symbol, start_date, end_date)

    if not paths:
        print("No data downloaded. Check network connection and date range.")
        return 1

    # Analyze each day
    all_params = []
    for path in paths:
        print(f"\n--- {path.stem} ---")
        df = load_trades(path)
        print(f"  Trades: {len(df):,}")
        print(f"  Price range: {df['price'].min():.2f} — {df['price'].max():.2f}")
        print(f"  Total volume: {df['qty'].sum():,.2f} BTC")

        params = estimate_impact_params(df, interval_seconds=60)
        all_params.append(params)
        print(f"  Kyle's λ: {params['kyle_lambda']:.6f}")
        print(f"  σ (1-min): {params['sigma']:.6f}")
        print(f"  Avg volume (1-min): {params['avg_volume']:.2f} BTC")
        print(f"  η estimate: {params['eta_estimate']:.6f}")

        # Compute and display microstructure
        features = compute_microstructure(df, interval_seconds=300)
        print(f"  5-min bars: {len(features.mid_price)}")

    # Summary
    import numpy as np

    print(f"\n{'='*60}")
    print("  IMPACT PARAMETER SUMMARY")
    print(f"{'='*60}")
    for key in ["kyle_lambda", "sigma", "avg_volume", "eta_estimate"]:
        values = [p[key] for p in all_params if not np.isnan(p[key])]
        if values:
            print(f"  {key:20s}  mean={np.mean(values):.6f}  std={np.std(values):.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
