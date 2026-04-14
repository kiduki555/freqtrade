"""FT-019: Cross-asset peer basket data provider.

Provides peer basket returns for correlation_stress and breadth_dispersion
factor computation. Pure computation — Freqtrade DataProvider passed in.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Empirical ceiling for crypto cross-sectional return std.
# Calibrated on 2020-2024 top-10 pairs; in extreme regimes (FTX collapse,
# COVID crash) cross-std exceeds this, pinning dispersion at 1.0.
_BREADTH_STD_CEILING: float = 0.10

# Default peer basket (top-10 by volume, excluding the symbol itself)
DEFAULT_PEERS: list[str] = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "AVAX/USDT",
    "DOT/USDT",
    "LINK/USDT",
    "ATOM/USDT",
]


def get_peer_list(symbol: str, peers: list[str] | None = None) -> list[str]:
    """Return peer basket excluding the symbol itself."""
    basket = peers if peers is not None else DEFAULT_PEERS
    return [p for p in basket if p != symbol]


def compute_peer_returns(
    peer_closes: dict[str, pd.Series],
    window: int = 24,
) -> pd.DataFrame:
    """Compute rolling log returns for each peer.

    Args:
        peer_closes: {pair: close_price_series} for each peer.
        window: Rolling window for returns (default 24h).

    Returns:
        DataFrame with columns = peer names, values = rolling log returns.
    """
    returns: dict[str, pd.Series] = {}
    for pair, close in peer_closes.items():
        lr = np.log(close / close.shift(window))
        returns[pair] = lr
    return pd.DataFrame(returns)


def compute_correlation_stress(
    peer_returns: pd.DataFrame,
    window: int = 72,
) -> pd.Series:
    """Compute average pairwise rolling correlation -> stress metric.

    Higher correlation = higher stress (markets moving together = less
    diversification).

    Args:
        peer_returns: DataFrame of peer log returns.
        window: Rolling correlation window (default 72h = 3 days).

    Returns:
        Series of correlation_stress values in [0, 1].
    """
    if peer_returns.shape[1] < 2:
        return pd.Series(np.nan, index=peer_returns.index)

    cols = peer_returns.columns.tolist()
    corr_sum = pd.Series(0.0, index=peer_returns.index)
    count = 0

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairwise_corr = (
                peer_returns[cols[i]]
                .rolling(window)
                .corr(peer_returns[cols[j]])
            )
            corr_sum = corr_sum + pairwise_corr.fillna(0)
            count += 1

    avg_corr = corr_sum / max(count, 1)
    # Map from [-1, 1] correlation to [0, 1] stress
    stress = (avg_corr + 1) / 2
    return stress.clip(0, 1)


def compute_breadth_dispersion(
    peer_returns: pd.DataFrame,
    window: int = 24,
) -> pd.Series:
    """Compute breadth dispersion: cross-sectional std of returns.

    Higher dispersion = markets diverging = higher breadth stress.

    Args:
        peer_returns: DataFrame of peer log returns.
        window: Rolling window for cross-sectional std.

    Returns:
        Series of breadth_dispersion values in [0, 1].
    """
    if peer_returns.shape[1] < 2:
        return pd.Series(np.nan, index=peer_returns.index)

    # Cross-sectional standard deviation at each timestamp
    cross_std = peer_returns.std(axis=1)
    # Smooth with rolling mean
    smoothed = cross_std.rolling(window, min_periods=1).mean()
    # Normalize to [0, 1] using empirical ceiling
    normalized = (smoothed / _BREADTH_STD_CEILING).clip(0, 1)
    return normalized


def fetch_peer_closes(
    dp: object,
    symbol: str,
    timeframe: str = "1h",
    peers: list[str] | None = None,
) -> dict[str, pd.Series]:
    """Fetch close prices for peer basket from Freqtrade DataProvider.

    This is the ONLY function that touches Freqtrade's DataProvider.

    Args:
        dp: Freqtrade DataProvider instance (typed as object to avoid import).
        symbol: Current trading pair (excluded from peers).
        timeframe: Candle timeframe.
        peers: Custom peer list (default: DEFAULT_PEERS).

    Returns:
        {pair: close_series} dict for available peers.
    """
    peer_list = get_peer_list(symbol, peers)
    closes: dict[str, pd.Series] = {}

    for pair in peer_list:
        try:
            df = dp.get_pair_dataframe(pair, timeframe)  # type: ignore[union-attr]
            if df is not None and len(df) > 0 and "close" in df.columns:
                closes[pair] = df["close"]
            else:
                logger.warning("No data for peer %s", pair)
        except Exception:
            logger.warning("Failed to fetch peer %s", pair, exc_info=True)

    return closes
