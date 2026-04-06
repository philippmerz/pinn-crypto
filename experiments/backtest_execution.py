"""Backtest execution strategies on real Binance BTC-USDT data.

Compares PINN-based execution against TWAP, VWAP, and analytical
Almgren-Chriss across multiple execution episodes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from datetime import date

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
from pinn.physics import AlmgrenChrissParams

OUTPUT_DIR = Path(__file__).parent / "output" / "backtest"


def load_or_train_pinn(kappa: float) -> HardConstrainedExecution:
    """Load Phase 1 PINN or train fresh for the given κ."""
    from pinn.training import TrainConfig, train_pinn, train_with_curriculum
    from functools import partial
    from pinn.physics import almgren_chriss_residual

    Q = 1.0  # work in normalized units for the model
    base = MLP(input_dim=1, output_dim=1, hidden_dim=128, num_layers=5)
    model = HardConstrainedExecution(base, initial_inventory=Q)

    config = TrainConfig(
        n_collocation=300,
        adam_epochs=10_000,
        adam_lr=1e-3,
        lbfgs_epochs=300,
    )

    if kappa > 5.0:
        import numpy as np
        n_stages = max(3, int(np.log2(kappa)))
        kappa_schedule = np.geomspace(1.0, kappa, n_stages).tolist()
        print(f"  Training PINN with κ-curriculum: {[f'{k:.1f}' for k in kappa_schedule]}")

        from pinn.training import train_with_curriculum
        result = train_with_curriculum(
            model,
            make_residual_fn=lambda k: partial(almgren_chriss_residual, kappa=k, horizon=1.0),
            kappa_schedule=kappa_schedule,
            config=config,
        )
    else:
        residual_fn = partial(almgren_chriss_residual, kappa=kappa, horizon=1.0)
        result = train_pinn(model, residual_fn, config)

    print(f"  PINN trained: final loss = {result.final_loss:.2e}")
    return model


def main():
    # --- Configuration ---
    SYMBOL = "BTCUSDT"
    TRADE_DATE = date(2025, 11, 29)
    EXECUTION_WINDOW_MIN = 30  # 30-minute execution windows
    INVENTORY_BTC = 0.5  # liquidate 0.5 BTC per window
    N_INTERVALS = 30  # 1-minute intervals within each window
    KAPPA = 5.0  # moderate urgency

    # --- Load data ---
    print(f"Loading {SYMBOL} trades for {TRADE_DATE}...")
    data_path = Path("data/trades") / SYMBOL / f"{SYMBOL}-trades-{TRADE_DATE}.csv"
    if not data_path.exists():
        print("  Downloading...")
        data_path = download_trades(SYMBOL, TRADE_DATE)
    df = load_trades(data_path)
    print(f"  {len(df):,} trades loaded")

    # --- Estimate impact parameters ---
    print("\nEstimating impact parameters...")
    params = estimate_impact_params(df, interval_seconds=60)
    print(f"  Kyle's λ = {params['kyle_lambda']:.4f}")
    print(f"  σ (1-min) = {params['sigma']:.6f}")
    print(f"  Avg volume = {params['avg_volume']:.2f} BTC/min")

    # --- Compute volume profile for VWAP ---
    features = compute_microstructure(df, interval_seconds=60)
    # Average hourly volume profile (24 hours * 60 min)
    hourly_volumes = features.total_volume
    # Create a profile for each 30-min window
    window_profile = np.array([
        hourly_volumes[i::N_INTERVALS].mean()
        for i in range(N_INTERVALS)
    ])
    if np.any(np.isnan(window_profile)):
        window_profile = np.ones(N_INTERVALS)
    window_profile = window_profile / window_profile.sum()

    # --- Train PINN ---
    print(f"\nTraining PINN (κ = {KAPPA})...")
    pinn_model = load_or_train_pinn(KAPPA)

    # --- Build strategies ---
    config = BacktestConfig(
        n_intervals=N_INTERVALS,
        temporary_impact_bps=5.0,
        permanent_impact_bps=2.0,
        spread_bps=0.5,
    )

    strategies = {
        "TWAP": TWAPStrategy(INVENTORY_BTC, N_INTERVALS),
        "VWAP": VWAPStrategy(INVENTORY_BTC, window_profile),
        "AC_Analytical": AnalyticalACStrategy(INVENTORY_BTC, KAPPA, N_INTERVALS),
        "PINN": PINNStrategy(pinn_model, INVENTORY_BTC, N_INTERVALS, kappa=None),
    }

    # --- Run backtest ---
    print(f"\nRunning backtest: {EXECUTION_WINDOW_MIN}-min windows, "
          f"{INVENTORY_BTC} BTC each, {N_INTERVALS} intervals...")

    all_results = backtest_on_trades(
        df=load_trades(data_path),
        total_inventory_btc=INVENTORY_BTC,
        execution_window_minutes=EXECUTION_WINDOW_MIN,
        strategies=strategies,
        config=config,
    )

    print(f"\n  Completed {len(all_results)} episodes")

    # --- Summarize ---
    if all_results:
        summary = summarize_results(all_results)
        print(f"\n{'='*70}")
        print("  BACKTEST RESULTS")
        print(f"{'='*70}")
        print(summary.to_string())

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        summary.to_csv(OUTPUT_DIR / "summary.csv")
        print(f"\n  Results saved to {OUTPUT_DIR / 'summary.csv'}")
    else:
        print("  No episodes completed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
