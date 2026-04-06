"""Multi-day, multi-κ backtest for rigorous validation.

Downloads multiple days of BTC-USDT data and runs execution backtests
across a range of urgency parameters (κ) to find the optimal operating
point and validate robustness across market regimes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
from datetime import date, timedelta
from functools import partial

from pinn.data import load_trades, download_trades, compute_microstructure, estimate_impact_params
from pinn.backtest import (
    BacktestConfig,
    TWAPStrategy,
    VWAPStrategy,
    AnalyticalACStrategy,
    backtest_on_trades,
    summarize_results,
)
from pinn.strategies import PINNStrategy
from pinn.network import MLP, HardConstrainedExecution
from pinn.physics import almgren_chriss_residual
from pinn.training import TrainConfig, train_pinn, train_with_curriculum

OUTPUT_DIR = Path(__file__).parent / "output" / "multi_backtest"

# --- Configuration ---
SYMBOL = "BTCUSDT"
EXECUTION_WINDOW_MIN = 30
INVENTORY_BTC = 0.5
N_INTERVALS = 30
KAPPAS = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]

# Diverse dates: different market conditions
TEST_DATES = [
    date(2025, 11, 29),  # already downloaded
    date(2025, 11, 30),  # already downloaded
    date(2025, 11, 15),
    date(2025, 10, 15),
    date(2025, 9, 15),
]


def train_pinn_for_kappa(kappa: float) -> HardConstrainedExecution:
    """Train a Phase 1 PINN at the given κ."""
    base = MLP(input_dim=1, output_dim=1, hidden_dim=128, num_layers=5)
    model = HardConstrainedExecution(base, initial_inventory=1.0)
    config = TrainConfig(n_collocation=300, adam_epochs=10_000, lbfgs_epochs=300)

    if kappa > 5.0:
        n_stages = max(3, int(np.log2(kappa)))
        schedule = np.geomspace(1.0, kappa, n_stages).tolist()
        train_with_curriculum(
            model,
            make_residual_fn=lambda k: partial(almgren_chriss_residual, kappa=k, horizon=1.0),
            kappa_schedule=schedule,
            config=config,
        )
    else:
        train_pinn(model, partial(almgren_chriss_residual, kappa=kappa, horizon=1.0), config)

    return model


def load_day(trade_date: date) -> pd.DataFrame | None:
    """Load or download a day of trade data."""
    data_path = Path("data/trades") / SYMBOL / f"{SYMBOL}-trades-{trade_date}.csv"
    if not data_path.exists():
        try:
            download_trades(SYMBOL, trade_date)
        except Exception as e:
            print(f"  Failed to download {trade_date}: {e}")
            return None
    return load_trades(data_path)


def main():
    # --- Download data ---
    print("=" * 70)
    print("  DOWNLOADING DATA")
    print("=" * 70)
    day_data = {}
    for d in TEST_DATES:
        print(f"\n  {d}...", end=" ", flush=True)
        df = load_day(d)
        if df is not None:
            day_data[d] = df
            print(f"{len(df):,} trades, ${df['price'].iloc[0]:,.0f}")
        else:
            print("SKIP")

    if not day_data:
        print("No data available.")
        return 1

    # --- Train PINNs for each κ ---
    print(f"\n{'='*70}")
    print(f"  TRAINING PINNS FOR κ = {KAPPAS}")
    print(f"{'='*70}")
    pinn_models = {}
    for kappa in KAPPAS:
        print(f"  κ = {kappa}...", end=" ", flush=True)
        pinn_models[kappa] = train_pinn_for_kappa(kappa)
        print("done")

    # --- Run backtests ---
    config = BacktestConfig(
        n_intervals=N_INTERVALS,
        temporary_impact_bps=5.0,
        permanent_impact_bps=2.0,
        spread_bps=0.5,
    )

    all_summaries = []

    for trade_date, df in day_data.items():
        print(f"\n{'='*70}")
        print(f"  BACKTESTING: {trade_date}")
        print(f"{'='*70}")

        # Compute volume profile for VWAP
        features = compute_microstructure(df, interval_seconds=60)
        window_profile = np.array([
            features.total_volume[i::N_INTERVALS].mean()
            for i in range(N_INTERVALS)
        ])
        if np.any(np.isnan(window_profile)):
            window_profile = np.ones(N_INTERVALS)
        window_profile /= window_profile.sum()

        # Estimate impact params
        params = estimate_impact_params(df, interval_seconds=60)
        print(f"  σ={params['sigma']:.6f}  λ={params['kyle_lambda']:.4f}  "
              f"vol={params['avg_volume']:.1f} BTC/min")

        # Build strategies for each κ
        strategies = {
            "TWAP": TWAPStrategy(INVENTORY_BTC, N_INTERVALS),
            "VWAP": VWAPStrategy(INVENTORY_BTC, window_profile),
        }

        for kappa in KAPPAS:
            strategies[f"AC_k{kappa}"] = AnalyticalACStrategy(
                INVENTORY_BTC, kappa, N_INTERVALS
            )
            strategies[f"PINN_k{kappa}"] = PINNStrategy(
                pinn_models[kappa], INVENTORY_BTC, N_INTERVALS
            )

        # Run
        results = backtest_on_trades(
            df=df,
            total_inventory_btc=INVENTORY_BTC,
            execution_window_minutes=EXECUTION_WINDOW_MIN,
            strategies=strategies,
            config=config,
        )

        if results:
            summary = summarize_results(results)
            summary["date"] = str(trade_date)
            all_summaries.append(summary)
            print(f"  {len(results)} episodes")

            # Print top strategies
            print(summary[["mean_is_bps", "std_is_bps", "exec_sharpe"]]
                  .head(8).to_string())

    # --- Aggregate across days ---
    if all_summaries:
        combined = pd.concat(all_summaries)

        # Average metrics across days
        agg = combined.reset_index().groupby("strategy").agg(
            mean_is_bps=("mean_is_bps", "mean"),
            std_is_bps=("std_is_bps", "mean"),
            exec_sharpe=("exec_sharpe", "mean"),
            n_days=("date", "nunique"),
        ).round(4).sort_values("mean_is_bps")

        print(f"\n{'='*70}")
        print("  AGGREGATE RESULTS ACROSS ALL DAYS")
        print(f"{'='*70}")
        print(agg.to_string())

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        agg.to_csv(OUTPUT_DIR / "aggregate_summary.csv")
        combined.to_csv(OUTPUT_DIR / "per_day_summary.csv")
        print(f"\n  Saved to {OUTPUT_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
