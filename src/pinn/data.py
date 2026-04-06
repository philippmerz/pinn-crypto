"""Binance data pipeline for BTC-USDT trade and market microstructure data."""

import io
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
BASE_URL = "https://data.binance.vision/data/spot/daily"

TRADE_COLUMNS = [
    "trade_id",
    "price",
    "qty",
    "quote_qty",
    "time",
    "is_buyer_maker",
    "is_best_match",
]


def download_trades(symbol: str, trade_date: date, dest: Path | None = None) -> Path:
    """Download daily trade data from Binance data portal.

    Returns path to the extracted CSV file.
    """
    dest = dest or DATA_DIR / "trades" / symbol
    dest.mkdir(parents=True, exist_ok=True)

    csv_path = dest / f"{symbol}-trades-{trade_date}.csv"
    if csv_path.exists():
        return csv_path

    date_str = trade_date.isoformat()
    url = f"{BASE_URL}/trades/{symbol}/{symbol}-trades-{date_str}.zip"

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        zf.extract(names[0], dest)
        extracted = dest / names[0]
        if extracted != csv_path:
            extracted.rename(csv_path)

    return csv_path


def load_trades(path: Path) -> pd.DataFrame:
    """Load a trade CSV into a DataFrame with proper types."""
    df = pd.read_csv(path, names=TRADE_COLUMNS)
    df["time"] = pd.to_datetime(df["time"], unit="us")
    df["price"] = df["price"].astype(np.float64)
    df["qty"] = df["qty"].astype(np.float64)
    df["quote_qty"] = df["quote_qty"].astype(np.float64)
    # Trade sign: +1 for buyer-initiated (taker buys), -1 for seller-initiated
    df["sign"] = np.where(df["is_buyer_maker"], -1, 1)
    return df


def download_date_range(
    symbol: str,
    start: date,
    end: date,
    dest: Path | None = None,
) -> list[Path]:
    """Download trades for a date range. Skips already-downloaded files."""
    paths = []
    current = start
    while current <= end:
        try:
            path = download_trades(symbol, current, dest)
            paths.append(path)
            print(f"  {current}: OK ({path.stat().st_size / 1e6:.1f} MB)")
        except requests.HTTPError as e:
            print(f"  {current}: SKIP ({e})")
        current += timedelta(days=1)
    return paths


@dataclass
class MicrostructureFeatures:
    """Market microstructure features computed from trade data."""

    interval_seconds: int
    time: np.ndarray
    mid_price: np.ndarray
    realized_vol: np.ndarray
    signed_volume: np.ndarray
    total_volume: np.ndarray
    trade_count: np.ndarray
    vwap: np.ndarray
    kyle_lambda: float
    price_change: np.ndarray


def compute_microstructure(
    df: pd.DataFrame,
    interval_seconds: int = 60,
    vol_window: int = 20,
) -> MicrostructureFeatures:
    """Compute market microstructure features from tick trades.

    Aggregates trades into fixed-time intervals and computes:
    - Mid-price proxy (trade-weighted)
    - Realized volatility (rolling)
    - Signed order flow (for Kyle's lambda estimation)
    - VWAP
    """
    df = df.copy()
    df["signed_qty"] = df["qty"] * df["sign"]
    df = df.set_index("time")
    rule = f"{interval_seconds}s"

    bars = df.resample(rule).agg(
        price_last=("price", "last"),
        price_first=("price", "first"),
        signed_volume=("signed_qty", "sum"),
        total_volume=("qty", "sum"),
        trade_count=("trade_id", "count"),
        quote_volume=("quote_qty", "sum"),
    ).dropna()

    # VWAP = total quote volume / total base volume
    bars["vwap"] = bars["quote_volume"] / bars["total_volume"]

    mid_price = bars["vwap"].values
    returns = np.diff(np.log(mid_price))
    price_change = np.diff(mid_price)

    # Realized volatility: rolling std of log returns, annualized
    vol = pd.Series(returns).rolling(vol_window).std().values
    # Pad to match length
    vol = np.concatenate([[np.nan] * (len(mid_price) - len(vol)), vol])

    # Kyle's lambda: ΔP = λ * OI + ε
    signed_vol = bars["signed_volume"].values
    if len(price_change) > 10:
        # Simple OLS regression
        oi = signed_vol[:-1]  # order imbalance
        dp = price_change  # price change
        valid = ~(np.isnan(oi) | np.isnan(dp))
        if valid.sum() > 10:
            oi_v, dp_v = oi[valid], dp[valid]
            kyle_lambda = np.sum(oi_v * dp_v) / np.sum(oi_v**2)
        else:
            kyle_lambda = np.nan
    else:
        kyle_lambda = np.nan

    return MicrostructureFeatures(
        interval_seconds=interval_seconds,
        time=bars.index.values,
        mid_price=mid_price,
        realized_vol=vol,
        signed_volume=signed_vol,
        total_volume=bars["total_volume"].values,
        trade_count=bars["trade_count"].values,
        vwap=bars["vwap"].values,
        kyle_lambda=kyle_lambda,
        price_change=np.concatenate([[0.0], price_change]),
    )


def estimate_impact_params(
    df: pd.DataFrame,
    interval_seconds: int = 60,
) -> dict[str, float]:
    """Estimate Almgren-Chriss impact parameters from trade data.

    Returns dict with keys:
    - kyle_lambda: permanent impact coefficient (bps per unit OI)
    - sigma: volatility (per interval)
    - avg_volume: average volume per interval
    - eta_estimate: temporary impact estimate (from spread proxy)
    """
    features = compute_microstructure(df, interval_seconds)

    # Volatility: std of returns per interval
    returns = np.diff(np.log(features.mid_price))
    sigma = np.nanstd(returns)

    avg_volume = np.nanmean(features.total_volume)

    # Temporary impact proxy: use realized volatility * sqrt(participation rate)
    # This follows the square-root law: impact ≈ σ * sqrt(Q/V)
    # For a unit participation rate: η ≈ σ / sqrt(V)
    eta_estimate = sigma / np.sqrt(avg_volume) if avg_volume > 0 else np.nan

    return {
        "kyle_lambda": features.kyle_lambda,
        "sigma": sigma,
        "avg_volume": avg_volume,
        "eta_estimate": eta_estimate,
        "n_intervals": len(features.mid_price),
    }
