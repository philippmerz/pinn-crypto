"""Execution backtest engine.

Simulates the execution of a liquidation order using different strategies
(PINN, TWAP, VWAP, analytical Almgren-Chriss) and computes implementation
shortfall metrics.
"""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd


class ExecutionStrategy(Protocol):
    """Interface for execution strategies."""

    def get_trade_rate(
        self,
        t: float,
        remaining: float,
        market_state: dict,
    ) -> float:
        """Return the number of units to trade in this interval.

        Args:
            t: current time (normalized to [0, 1])
            remaining: remaining inventory
            market_state: dict with keys like 'volatility', 'volume', 'spread'

        Returns:
            Number of units to sell (positive = sell).
        """
        ...


class TWAPStrategy:
    """Time-Weighted Average Price: sell at a constant rate."""

    def __init__(self, total_inventory: float, n_intervals: int):
        self.rate = total_inventory / n_intervals

    def get_trade_rate(self, t: float, remaining: float, market_state: dict) -> float:
        return min(self.rate, remaining)


class VWAPStrategy:
    """Volume-Weighted Average Price: sell proportional to expected volume."""

    def __init__(self, total_inventory: float, volume_profile: np.ndarray):
        self.total = total_inventory
        # Normalize volume profile to sum to 1
        self.profile = volume_profile / volume_profile.sum()

    def get_trade_rate(self, t: float, remaining: float, market_state: dict) -> float:
        # t is normalized [0,1], map to profile index
        idx = min(int(t * len(self.profile)), len(self.profile) - 1)
        target = self.total * self.profile[idx]
        return min(target, remaining)


class AnalyticalACStrategy:
    """Analytical Almgren-Chriss: sinh-based optimal trajectory."""

    def __init__(self, total_inventory: float, kappa: float, n_intervals: int):
        self.Q = total_inventory
        self.kappa = kappa
        self.n = n_intervals

    def get_trade_rate(self, t: float, remaining: float, market_state: dict) -> float:
        T = 1.0
        dt = T / self.n
        # Inventory at current time
        X_t = self.Q * np.sinh(self.kappa * (T - t)) / np.sinh(self.kappa * T)
        # Inventory at next time step
        t_next = min(t + dt, T)
        X_next = self.Q * np.sinh(self.kappa * (T - t_next)) / np.sinh(self.kappa * T)
        trade = max(X_t - X_next, 0)
        return min(trade, remaining)


@dataclass
class ExecutionResult:
    """Result of a single execution simulation."""

    strategy_name: str
    fills: list[dict] = field(default_factory=list)
    arrival_price: float = 0.0
    total_inventory: float = 0.0

    @property
    def avg_execution_price(self) -> float:
        if not self.fills:
            return 0.0
        total_qty = sum(f["qty"] for f in self.fills)
        if total_qty == 0:
            return 0.0
        return sum(f["qty"] * f["price"] for f in self.fills) / total_qty

    @property
    def implementation_shortfall(self) -> float:
        """IS = (arrival_price - avg_exec_price) / arrival_price for a sell."""
        if self.arrival_price == 0:
            return 0.0
        return (self.arrival_price - self.avg_execution_price) / self.arrival_price

    @property
    def implementation_shortfall_bps(self) -> float:
        return self.implementation_shortfall * 10_000

    @property
    def total_executed(self) -> float:
        return sum(f["qty"] for f in self.fills)

    @property
    def completion_rate(self) -> float:
        if self.total_inventory == 0:
            return 0.0
        return self.total_executed / self.total_inventory


@dataclass
class BacktestConfig:
    """Backtest parameters."""

    n_intervals: int = 60  # number of execution intervals
    temporary_impact_bps: float = 5.0  # η in basis points per unit participation
    permanent_impact_bps: float = 2.0  # θ in basis points per unit participation
    spread_bps: float = 1.0  # half-spread cost in bps


def simulate_execution(
    strategy: ExecutionStrategy,
    strategy_name: str,
    prices: np.ndarray,
    volumes: np.ndarray,
    total_inventory: float,
    config: BacktestConfig = BacktestConfig(),
) -> ExecutionResult:
    """Simulate execution of a liquidation order.

    Uses a transient impact model:
        exec_price = mid_price - η * (trade_size / interval_volume) - spread/2

    For a sell order, we receive less than mid due to impact and spread.

    Args:
        strategy: the execution strategy
        prices: mid-price series (length = n_intervals)
        volumes: volume series (length = n_intervals)
        total_inventory: total amount to liquidate
        config: backtest parameters
    """
    result = ExecutionResult(
        strategy_name=strategy_name,
        arrival_price=prices[0],
        total_inventory=total_inventory,
    )

    remaining = total_inventory
    n = min(len(prices), config.n_intervals)

    for i in range(n):
        if remaining <= 0:
            break

        t = i / n  # normalized time
        market_state = {
            "price": prices[i],
            "volume": volumes[i],
            "volatility": 0.0,  # could be computed from rolling window
        }

        trade_size = strategy.get_trade_rate(t, remaining, market_state)
        trade_size = max(0, min(trade_size, remaining))

        if trade_size == 0:
            continue

        # Participation rate
        participation = trade_size / max(volumes[i], 1e-10)

        # Temporary impact: price worsens proportional to participation
        temp_impact_pct = config.temporary_impact_bps * 1e-4 * participation

        # Permanent impact: shifts the mid-price for all subsequent intervals
        perm_impact_pct = config.permanent_impact_bps * 1e-4 * participation

        # Spread cost (for crossing the spread)
        spread_cost_pct = config.spread_bps * 1e-4

        # Execution price for a sell (we receive less)
        exec_price = prices[i] * (1 - temp_impact_pct - spread_cost_pct)

        # Apply permanent impact to future prices
        prices[i + 1:] *= (1 - perm_impact_pct)

        result.fills.append({
            "interval": i,
            "t": t,
            "qty": trade_size,
            "price": exec_price,
            "mid_price": prices[i],
            "participation": participation,
        })

        remaining -= trade_size

    # If any inventory remains, force-liquidate at last price with extra penalty
    if remaining > 0:
        penalty_price = prices[-1] * (1 - config.spread_bps * 1e-4 * 5)
        result.fills.append({
            "interval": n - 1,
            "t": 1.0,
            "qty": remaining,
            "price": penalty_price,
            "mid_price": prices[-1],
            "participation": 1.0,
        })

    return result


def run_backtest(
    prices: np.ndarray,
    volumes: np.ndarray,
    total_inventory: float,
    strategies: dict[str, ExecutionStrategy],
    config: BacktestConfig = BacktestConfig(),
) -> dict[str, ExecutionResult]:
    """Run multiple strategies on the same price/volume path."""
    results = {}
    for name, strategy in strategies.items():
        # Each strategy gets its own copy of prices (permanent impact modifies them)
        results[name] = simulate_execution(
            strategy, name, prices.copy(), volumes, total_inventory, config,
        )
    return results


def backtest_on_trades(
    df: pd.DataFrame,
    total_inventory_btc: float,
    execution_window_minutes: int,
    strategies: dict[str, ExecutionStrategy],
    config: BacktestConfig = BacktestConfig(),
    n_episodes: int | None = None,
) -> list[dict[str, ExecutionResult]]:
    """Run walk-forward backtests on real trade data.

    Splits the trading day into non-overlapping execution windows
    and runs all strategies on each window.
    """
    df = df.set_index("time") if "time" in df.columns else df

    # Resample into intervals matching n_intervals within each window
    interval_seconds = (execution_window_minutes * 60) // config.n_intervals
    rule = f"{interval_seconds}s"

    bars = df.resample(rule).agg(
        price=("price", "last"),
        volume=("qty", "sum"),
    ).dropna()

    # Split into non-overlapping windows
    window_size = config.n_intervals
    n_windows = len(bars) // window_size
    if n_episodes is not None:
        n_windows = min(n_windows, n_episodes)

    all_results = []
    for w in range(n_windows):
        start = w * window_size
        end = start + window_size
        window = bars.iloc[start:end]

        if len(window) < window_size:
            break

        prices = window["price"].values.astype(np.float64)
        volumes = window["volume"].values.astype(np.float64)

        results = run_backtest(prices, volumes, total_inventory_btc, strategies, config)
        all_results.append(results)

    return all_results


def summarize_results(all_results: list[dict[str, ExecutionResult]]) -> pd.DataFrame:
    """Summarize backtest results across episodes."""
    rows = []
    for episode_results in all_results:
        for name, result in episode_results.items():
            rows.append({
                "strategy": name,
                "is_bps": result.implementation_shortfall_bps,
                "avg_exec_price": result.avg_execution_price,
                "arrival_price": result.arrival_price,
                "completion_rate": result.completion_rate,
                "n_fills": len(result.fills),
            })

    df = pd.DataFrame(rows)
    summary = df.groupby("strategy").agg(
        mean_is_bps=("is_bps", "mean"),
        std_is_bps=("is_bps", "std"),
        p95_is_bps=("is_bps", lambda x: x.quantile(0.95)),
        mean_completion=("completion_rate", "mean"),
        n_episodes=("is_bps", "count"),
    ).round(4)

    # Execution Sharpe = -mean(IS) / std(IS)
    summary["exec_sharpe"] = (-summary["mean_is_bps"] / summary["std_is_bps"]).round(4)

    return summary.sort_values("mean_is_bps")
