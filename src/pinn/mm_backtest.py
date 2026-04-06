"""Market-making backtest engine for Polymarket.

Simulates a market maker quoting on a binary prediction market.
Replays historical trades against the maker's bid/ask quotes and
tracks PnL, inventory, and risk metrics.
"""

from dataclasses import dataclass, field
from math import log

import numpy as np
import pandas as pd
import torch

from pinn.market_making import MarketMakerParams, sigmoid, solve_fd, optimal_quotes


@dataclass
class MMSimConfig:
    """Market-making simulation configuration."""

    initial_cash: float = 20.0  # starting cash (USD)
    q_max: int = 5  # max absolute inventory
    tick_size: float = 0.01  # price grid resolution
    maker_rebate_pct: float = 0.0  # maker rebate as fraction of fees paid by taker


@dataclass
class MMState:
    """Current state of the market maker."""

    cash: float
    inventory: int = 0
    n_fills_bid: int = 0
    n_fills_ask: int = 0
    n_quotes: int = 0
    pnl_history: list[float] = field(default_factory=list)
    inventory_history: list[int] = field(default_factory=list)
    cash_history: list[float] = field(default_factory=list)

    @property
    def total_fills(self) -> int:
        return self.n_fills_bid + self.n_fills_ask

    def mark_to_market(self, mid_price: float) -> float:
        return self.cash + self.inventory * mid_price


class QuoteStrategy:
    """Base class for market-making quote strategies."""

    def get_quotes(
        self, mid_price: float, inventory: int, time_remaining: float, **kwargs
    ) -> tuple[float | None, float | None]:
        """Return (bid_price, ask_price). None means no quote on that side."""
        raise NotImplementedError


class NaiveSymmetricMM(QuoteStrategy):
    """Naive market maker: fixed spread around mid, no inventory management."""

    def __init__(self, half_spread: float = 0.02):
        self.half_spread = half_spread

    def get_quotes(self, mid_price, inventory, time_remaining, **kwargs):
        q_max = kwargs.get("q_max", 5)
        bid = round(mid_price - self.half_spread, 2) if inventory < q_max else None
        ask = round(mid_price + self.half_spread, 2) if inventory > -q_max else None
        return bid, ask


class FDBasedMM(QuoteStrategy):
    """Market maker using FD-solved value function for a specific parameter set."""

    def __init__(self, params: MarketMakerParams):
        self.params = params
        self.fd_result = solve_fd(params, n_z=201, n_t=2000, z_min=-6, z_max=6)
        self.z_grid = self.fd_result["z"]

    def get_quotes(self, mid_price, inventory, time_remaining, **kwargs):
        q_max = self.params.q_max
        if abs(inventory) >= q_max:
            # Only quote on the reducing side
            pass

        p = np.clip(mid_price, 0.01, 0.99)
        z = np.log(p / (1 - p))

        # Find closest time index (time_remaining as fraction of T)
        tau = 1.0 - time_remaining  # τ=0 at start, τ=1 at expiry
        t_stored = self.fd_result["t"]
        # t_stored goes from 0 to T, we want tau fraction
        t_idx = max(0, min(len(t_stored) - 1,
                           int(tau * (len(t_stored) - 1))))

        theta_t = {q: self.fd_result["theta"][q][t_idx] for q in self.fd_result["theta"]}
        bid_arr, ask_arr = optimal_quotes(theta_t, self.z_grid, inventory, q_max, self.params)

        z_idx = np.argmin(np.abs(self.z_grid - z))
        bid = bid_arr[z_idx]
        ask = ask_arr[z_idx]

        # Snap to tick and enforce bounds
        bid = max(0.01, round(bid / 0.01) * 0.01) if inventory < q_max else None
        ask = min(0.99, round(ask / 0.01) * 0.01) if inventory > -q_max else None

        # Don't quote if bid >= ask (would cross)
        if bid is not None and ask is not None and bid >= ask:
            # Widen: only quote the side that reduces inventory
            if inventory > 0:
                bid = None
            else:
                ask = None

        return bid, ask


class PINNBasedMM(QuoteStrategy):
    """Market maker using the parametric PINN for instant quote computation."""

    def __init__(self, model, q_max: int, sigma: float, kappa: float, gamma: float):
        self.model = model
        self.model.eval()
        self.q_max = q_max
        self.sigma = sigma
        self.kappa = kappa
        self.gamma = gamma
        self.base_spread = (1.0 / gamma) * log(1.0 + gamma / kappa)

    def _eval_theta(self, tau: float, z: float, q: int) -> float:
        q_norm = q / self.q_max
        inp = torch.tensor(
            [[tau, z, q_norm, self.sigma, self.kappa, self.gamma]],
            dtype=torch.float32,
        )
        with torch.no_grad():
            return self.model(inp).item()

    def get_quotes(self, mid_price, inventory, time_remaining, **kwargs):
        q_max = self.q_max
        p = np.clip(mid_price, 0.01, 0.99)
        z = np.log(p / (1 - p))
        tau = 1.0 - time_remaining

        # Indifference pricing from θ differences
        theta_q = self._eval_theta(tau, z, inventory)
        bid, ask = None, None

        if inventory < q_max:
            theta_q_plus = self._eval_theta(tau, z, min(inventory + 1, q_max))
            bid = theta_q_plus - theta_q - self.base_spread
            bid = max(0.01, round(bid / 0.01) * 0.01)

        if inventory > -q_max:
            theta_q_minus = self._eval_theta(tau, z, max(inventory - 1, -q_max))
            ask = theta_q - theta_q_minus + self.base_spread
            ask = min(0.99, round(ask / 0.01) * 0.01)

        if bid is not None and ask is not None and bid >= ask:
            if inventory > 0:
                bid = None
            else:
                ask = None

        return bid, ask


def simulate_market_making(
    trades_df: pd.DataFrame,
    strategy: QuoteStrategy,
    config: MMSimConfig = MMSimConfig(),
    horizon_minutes: float = 60.0,
) -> dict:
    """Simulate market making on historical trade data.

    For each trade in the history:
    1. Compute the maker's current quotes
    2. Check if the trade would fill against our quotes
    3. Update cash and inventory accordingly

    A trade fills our bid if trade.side == 'SELL' and trade.price <= our_bid
    A trade fills our ask if trade.side == 'BUY' and trade.price >= our_ask
    """
    state = MMState(cash=config.initial_cash)

    if trades_df.empty:
        return _build_result(state, strategy.__class__.__name__, 0.5)

    timestamps = trades_df["timestamp"].astype(float).values
    prices = trades_df["price"].astype(float).values
    sides = trades_df["side"].values
    sizes = trades_df["size"].astype(float).values

    t_start = timestamps[0]
    t_end = t_start + horizon_minutes * 60
    total_duration = t_end - t_start

    for i in range(len(trades_df)):
        if timestamps[i] > t_end:
            break

        trade_price = prices[i]
        trade_side = sides[i]
        trade_size = min(sizes[i], 1.0)  # limit fill size to 1 share per trade

        time_remaining = max(0, (t_end - timestamps[i]) / total_duration)
        mid_price = trade_price  # use trade price as mid estimate

        # Get our quotes
        bid, ask = strategy.get_quotes(
            mid_price, state.inventory, time_remaining, q_max=config.q_max,
        )
        state.n_quotes += 1

        # Check for fills
        filled = False
        if trade_side == "SELL" and bid is not None and trade_price <= bid:
            # We buy at our bid
            if state.inventory < config.q_max and state.cash >= bid:
                state.cash -= bid
                state.inventory += 1
                state.n_fills_bid += 1
                filled = True

        elif trade_side == "BUY" and ask is not None and trade_price >= ask:
            # We sell at our ask
            if state.inventory > -config.q_max:
                state.cash += ask
                state.inventory -= 1
                state.n_fills_ask += 1
                filled = True

        # Record state
        mtm = state.mark_to_market(trade_price)
        state.pnl_history.append(mtm - config.initial_cash)
        state.inventory_history.append(state.inventory)
        state.cash_history.append(state.cash)

    final_mid = prices[min(len(prices) - 1, i)] if len(prices) > 0 else 0.5
    return _build_result(state, strategy.__class__.__name__, final_mid)


def _build_result(state: MMState, strategy_name: str, final_mid: float) -> dict:
    final_mtm = state.mark_to_market(final_mid)
    pnl = final_mtm - (state.cash_history[0] if state.cash_history else state.cash)
    pnl_arr = np.array(state.pnl_history) if state.pnl_history else np.array([0.0])

    return {
        "strategy": strategy_name,
        "final_pnl": pnl,
        "final_cash": state.cash,
        "final_inventory": state.inventory,
        "final_mtm": final_mtm,
        "n_fills_bid": state.n_fills_bid,
        "n_fills_ask": state.n_fills_ask,
        "total_fills": state.total_fills,
        "n_quotes": state.n_quotes,
        "max_inventory": max(state.inventory_history) if state.inventory_history else 0,
        "min_inventory": min(state.inventory_history) if state.inventory_history else 0,
        "pnl_std": float(np.std(pnl_arr)),
        "max_drawdown": float(np.min(pnl_arr)) if len(pnl_arr) > 0 else 0.0,
    }
