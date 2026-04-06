"""Polymarket data pipeline for prediction market making.

Fetches market metadata, order books, trade history, and price data
from the Polymarket Gamma/CLOB/Data APIs. Estimates market-making
parameters (volatility, arrival rates, fill rates).
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"
DATA_URL = "https://data-api.polymarket.com"
TIMEOUT = 15


@dataclass
class Market:
    """A Polymarket prediction market."""

    question: str
    slug: str
    condition_id: str
    yes_token_id: str
    no_token_id: str
    outcomes: list[str]
    liquidity: float
    volume_24h: float
    yes_price: float
    no_price: float
    end_date: str | None = None

    @property
    def mid_price(self) -> float:
        return self.yes_price

    @property
    def spread_estimate(self) -> float:
        """Rough spread from YES/NO price disagreement."""
        return max(0, 1.0 - self.yes_price - self.no_price)


@dataclass
class OrderBook:
    """Snapshot of a market's order book."""

    token_id: str
    bids: list[tuple[float, float]]  # (price, size) descending by price
    asks: list[tuple[float, float]]  # (price, size) ascending by price
    tick_size: float
    min_order_size: float
    timestamp: float = 0.0

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 1.0

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    def depth_at(self, levels: int = 5) -> tuple[float, float]:
        """Total size within top N levels on each side."""
        bid_depth = sum(s for _, s in self.bids[:levels])
        ask_depth = sum(s for _, s in self.asks[:levels])
        return bid_depth, ask_depth


def fetch_active_markets(
    min_liquidity: float = 0,
    max_liquidity: float = float("inf"),
    limit: int = 100,
) -> list[Market]:
    """Fetch active markets, optionally filtered by liquidity range."""
    resp = requests.get(
        f"{GAMMA_URL}/markets",
        params={"limit": limit, "active": "true", "closed": "false"},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()

    markets = []
    for m in resp.json():
        liq = float(m.get("liquidity") or 0)
        if not (min_liquidity <= liq <= max_liquidity):
            continue

        clob_ids = m.get("clobTokenIds", "[]")
        if isinstance(clob_ids, str):
            clob_ids = json.loads(clob_ids)
        if len(clob_ids) < 2:
            continue

        outcomes = m.get("outcomes", "[]")
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)

        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            prices = json.loads(prices)
        if len(prices) < 2:
            continue

        markets.append(Market(
            question=m.get("question", ""),
            slug=m.get("slug", ""),
            condition_id=m.get("conditionId", ""),
            yes_token_id=clob_ids[0],
            no_token_id=clob_ids[1],
            outcomes=outcomes,
            liquidity=liq,
            volume_24h=float(m.get("volume24hr") or 0),
            yes_price=float(prices[0]),
            no_price=float(prices[1]),
            end_date=m.get("endDate"),
        ))

    return sorted(markets, key=lambda m: m.liquidity)


def fetch_order_book(token_id: str) -> OrderBook:
    """Fetch current order book for a token."""
    resp = requests.get(
        f"{CLOB_URL}/book",
        params={"token_id": token_id},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()

    bids = sorted(
        [(float(b["price"]), float(b["size"])) for b in data.get("bids", [])],
        key=lambda x: -x[0],
    )
    asks = sorted(
        [(float(a["price"]), float(a["size"])) for a in data.get("asks", [])],
        key=lambda x: x[0],
    )

    return OrderBook(
        token_id=token_id,
        bids=bids,
        asks=asks,
        tick_size=float(data.get("tick_size", 0.01)),
        min_order_size=float(data.get("min_order_size", 5)),
        timestamp=time.time(),
    )


def fetch_trades(condition_id: str, limit: int = 500) -> pd.DataFrame:
    """Fetch recent trades for a market."""
    all_trades = []
    cursor = None

    while len(all_trades) < limit:
        params = {"market": condition_id, "limit": min(100, limit - len(all_trades))}
        if cursor:
            params["cursor"] = cursor

        resp = requests.get(f"{DATA_URL}/trades", params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        trades = resp.json()

        if not trades:
            break
        all_trades.extend(trades)

        # Check for pagination cursor
        if len(trades) < 100:
            break
        cursor = trades[-1].get("id")

    if not all_trades:
        return pd.DataFrame()

    df = pd.DataFrame(all_trades)
    if "timestamp" in df.columns:
        df["time"] = pd.to_datetime(df["timestamp"].astype(float), unit="s")
    if "price" in df.columns:
        df["price"] = df["price"].astype(float)
    if "size" in df.columns:
        df["size"] = df["size"].astype(float)

    return df.sort_values("timestamp").reset_index(drop=True)


@dataclass
class MarketMakingEstimates:
    """Estimated parameters for the A-S HJB from market data."""

    sigma_logit: float  # volatility in logit space
    arrival_rate: float  # trades per minute
    avg_trade_size: float  # average trade size in shares
    spread_cents: float  # observed spread in cents
    bid_depth: float  # average top-5 bid depth
    ask_depth: float  # average top-5 ask depth
    mid_price: float
    n_trades: int


def estimate_parameters(
    trades_df: pd.DataFrame,
    book: OrderBook,
) -> MarketMakingEstimates:
    """Estimate market-making parameters from trades and order book.

    Volatility is estimated in logit space: σ = std(Δlogit(p)) / sqrt(Δt).
    Arrival rate is estimated from trade frequency.
    """
    if trades_df.empty:
        return MarketMakingEstimates(
            sigma_logit=0, arrival_rate=0, avg_trade_size=0,
            spread_cents=book.spread * 100, bid_depth=0, ask_depth=0,
            mid_price=book.mid, n_trades=0,
        )

    prices = trades_df["price"].values
    timestamps = trades_df["timestamp"].astype(float).values

    # Filter to valid prices (avoid logit of 0 or 1)
    valid = (prices > 0.01) & (prices < 0.99)
    prices = prices[valid]
    timestamps = timestamps[valid]

    if len(prices) < 5:
        return MarketMakingEstimates(
            sigma_logit=0, arrival_rate=0, avg_trade_size=0,
            spread_cents=book.spread * 100, bid_depth=0, ask_depth=0,
            mid_price=book.mid, n_trades=len(prices),
        )

    # Logit-space volatility
    logit_prices = np.log(prices / (1 - prices))
    d_logit = np.diff(logit_prices)
    d_time = np.diff(timestamps)
    d_time = np.maximum(d_time, 1.0)  # avoid division by zero

    # σ per second in logit space, then convert to per minute
    sigma_per_sec = np.std(d_logit / np.sqrt(d_time))
    sigma_per_min = sigma_per_sec * np.sqrt(60)

    # Arrival rate (trades per minute)
    total_time_min = (timestamps[-1] - timestamps[0]) / 60
    arrival_rate = len(prices) / max(total_time_min, 1)

    # Average trade size
    avg_size = trades_df["size"].astype(float).mean() if "size" in trades_df.columns else 0

    # Book depth
    bid_depth, ask_depth = book.depth_at(5)

    return MarketMakingEstimates(
        sigma_logit=sigma_per_min,
        arrival_rate=arrival_rate,
        avg_trade_size=avg_size,
        spread_cents=book.spread * 100,
        bid_depth=bid_depth,
        ask_depth=ask_depth,
        mid_price=book.mid,
        n_trades=len(prices),
    )


def scan_markets(
    min_liquidity: float = 1000,
    max_liquidity: float = 50000,
    min_spread_cents: float = 1.5,
) -> list[tuple[Market, OrderBook, MarketMakingEstimates]]:
    """Scan for market-making opportunities: illiquid markets with wide spreads."""
    markets = fetch_active_markets(min_liquidity, max_liquidity)
    results = []

    for market in markets:
        try:
            book = fetch_order_book(market.yes_token_id)
            if book.spread * 100 < min_spread_cents:
                continue
            if len(book.bids) < 3 or len(book.asks) < 3:
                continue

            trades = fetch_trades(market.condition_id, limit=200)
            estimates = estimate_parameters(trades, book)
            results.append((market, book, estimates))
            time.sleep(0.2)  # rate limiting
        except Exception as e:
            print(f"  Skip {market.slug}: {e}")
            continue

    return results
