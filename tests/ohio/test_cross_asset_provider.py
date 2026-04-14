"""Tests for FT-019: cross_asset_provider — pure computation layer."""
from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from freqtrade.ohio.adapters.freqtrade.cross_asset_provider import (
    DEFAULT_PEERS,
    compute_breadth_dispersion,
    compute_correlation_stress,
    compute_peer_returns,
    fetch_peer_closes,
    get_peer_list,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_close_series(n: int = 200, seed: int = 42) -> pd.Series:
    """Random-walk close price series."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, n)
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(prices, dtype=float)


def _make_correlated_closes(n: int = 200, n_peers: int = 3) -> dict[str, pd.Series]:
    """All peers share the same returns (perfectly correlated)."""
    base = _make_close_series(n, seed=0)
    return {f"PEER{i}/USDT": base.copy() for i in range(n_peers)}


def _make_uncorrelated_closes(n: int = 200, n_peers: int = 3) -> dict[str, pd.Series]:
    """Each peer has independent random returns."""
    return {
        f"PEER{i}/USDT": _make_close_series(n, seed=i * 7 + 1)
        for i in range(n_peers)
    }


def _make_divergent_closes(n: int = 200) -> dict[str, pd.Series]:
    """Two peers moving in opposite directions."""
    t = np.arange(n, dtype=float)
    up = 100 * np.exp(0.001 * t)
    down = 100 * np.exp(-0.001 * t)
    return {
        "UP/USDT": pd.Series(up, dtype=float),
        "DOWN/USDT": pd.Series(down, dtype=float),
    }


# ---------------------------------------------------------------------------
# 1-3 get_peer_list
# ---------------------------------------------------------------------------

class TestGetPeerList:
    def test_excludes_current_symbol(self) -> None:
        result = get_peer_list("BTC/USDT")
        assert "BTC/USDT" not in result
        assert len(result) == len(DEFAULT_PEERS) - 1

    def test_custom_list(self) -> None:
        custom = ["AAA/USDT", "BBB/USDT", "CCC/USDT"]
        result = get_peer_list("BBB/USDT", peers=custom)
        assert result == ["AAA/USDT", "CCC/USDT"]

    def test_symbol_not_in_list(self) -> None:
        result = get_peer_list("DOGE/USDT")
        assert len(result) == len(DEFAULT_PEERS)


# ---------------------------------------------------------------------------
# 4-6 compute_peer_returns
# ---------------------------------------------------------------------------

class TestComputePeerReturns:
    def test_shape_matches_input(self) -> None:
        closes = _make_uncorrelated_closes(100, 3)
        df = compute_peer_returns(closes, window=5)
        assert df.shape[0] == 100
        assert df.shape[1] == 3

    def test_values_are_log_returns(self) -> None:
        n, window = 50, 5
        close = pd.Series(np.linspace(100, 150, n), dtype=float)
        closes = {"X/USDT": close}
        df = compute_peer_returns(closes, window=window)
        # Manual check for a specific row
        idx = 10
        expected = math.log(close.iloc[idx] / close.iloc[idx - window])
        assert pytest.approx(df["X/USDT"].iloc[idx], rel=1e-6) == expected

    def test_handles_nan_in_close(self) -> None:
        close = _make_close_series(50)
        close.iloc[10] = np.nan
        df = compute_peer_returns({"A/USDT": close}, window=3)
        # NaN propagates but no exception
        assert df.shape == (50, 1)


# ---------------------------------------------------------------------------
# 7-10 compute_correlation_stress
# ---------------------------------------------------------------------------

class TestComputeCorrelationStress:
    def test_returns_0_1_range(self) -> None:
        closes = _make_uncorrelated_closes(200, 4)
        ret = compute_peer_returns(closes, window=5)
        stress = compute_correlation_stress(ret, window=20)
        valid = stress.dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_perfectly_correlated_high_stress(self) -> None:
        closes = _make_correlated_closes(200, 3)
        ret = compute_peer_returns(closes, window=5)
        stress = compute_correlation_stress(ret, window=20)
        # Perfect correlation → stress should approach 1.0
        tail = stress.iloc[-30:]
        assert tail.dropna().mean() > 0.9

    def test_uncorrelated_mid_stress(self) -> None:
        closes = _make_uncorrelated_closes(500, 4)
        ret = compute_peer_returns(closes, window=5)
        stress = compute_correlation_stress(ret, window=50)
        tail = stress.iloc[-100:]
        mean_stress = tail.dropna().mean()
        # Uncorrelated → avg corr ~0 → stress ~0.5
        assert 0.2 < mean_stress < 0.8

    def test_single_peer_returns_nan(self) -> None:
        closes = {"ONLY/USDT": _make_close_series(100)}
        ret = compute_peer_returns(closes, window=5)
        stress = compute_correlation_stress(ret, window=20)
        assert stress.isna().all()


# ---------------------------------------------------------------------------
# 11-14 compute_breadth_dispersion
# ---------------------------------------------------------------------------

class TestComputeBreadthDispersion:
    def test_returns_0_1_range(self) -> None:
        closes = _make_uncorrelated_closes(200, 4)
        ret = compute_peer_returns(closes, window=5)
        disp = compute_breadth_dispersion(ret, window=10)
        valid = disp.dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_identical_returns_zero_dispersion(self) -> None:
        closes = _make_correlated_closes(200, 3)
        ret = compute_peer_returns(closes, window=5)
        disp = compute_breadth_dispersion(ret, window=10)
        # Identical returns → cross-sectional std = 0
        valid = disp.dropna()
        assert valid.max() < 0.01

    def test_divergent_returns_high_dispersion(self) -> None:
        closes = _make_divergent_closes(200)
        ret = compute_peer_returns(closes, window=5)
        disp = compute_breadth_dispersion(ret, window=10)
        tail = disp.iloc[-30:]
        assert tail.dropna().mean() > 0.01

    def test_single_peer_returns_nan(self) -> None:
        closes = {"ONLY/USDT": _make_close_series(100)}
        ret = compute_peer_returns(closes, window=5)
        disp = compute_breadth_dispersion(ret, window=10)
        assert disp.isna().all()


# ---------------------------------------------------------------------------
# 15-16 fetch_peer_closes with fake DataProvider
# ---------------------------------------------------------------------------

class _FakeDataProvider:
    """Minimal DataProvider stub backed by a dict of DataFrames."""

    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self._data = data

    def get_pair_dataframe(self, pair: str, timeframe: str) -> pd.DataFrame:
        if pair not in self._data:
            raise KeyError(f"No data for {pair}")
        return self._data[pair]


class TestFetchPeerCloses:
    def test_returns_close_series(self) -> None:
        data = {
            "ETH/USDT": pd.DataFrame({"close": [1, 2, 3]}),
            "BNB/USDT": pd.DataFrame({"close": [4, 5, 6]}),
        }
        dp = _FakeDataProvider(data)
        result = fetch_peer_closes(dp, "BTC/USDT", peers=["ETH/USDT", "BNB/USDT"])
        assert set(result.keys()) == {"ETH/USDT", "BNB/USDT"}
        assert list(result["ETH/USDT"]) == [1, 2, 3]

    def test_skips_unavailable_peers(self) -> None:
        data = {
            "ETH/USDT": pd.DataFrame({"close": [1, 2, 3]}),
        }
        dp = _FakeDataProvider(data)
        result = fetch_peer_closes(
            dp, "BTC/USDT", peers=["ETH/USDT", "MISSING/USDT"]
        )
        assert "ETH/USDT" in result
        assert "MISSING/USDT" not in result


# ---------------------------------------------------------------------------
# 17 No Freqtrade imports (AST check)
# ---------------------------------------------------------------------------

def test_no_freqtrade_imports_in_source() -> None:
    """cross_asset_provider must not import from freqtrade (except ohio sub-pkg)."""
    src = Path(inspect.getfile(get_peer_list)).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("freqtrade"), (
                    f"Forbidden import: {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("freqtrade"):
                # Allow ohio sub-package imports
                assert "ohio" in node.module, (
                    f"Forbidden import from: {node.module}"
                )


# ---------------------------------------------------------------------------
# 18 DEFAULT_PEERS sanity
# ---------------------------------------------------------------------------

def test_default_peers_has_10_entries() -> None:
    assert len(DEFAULT_PEERS) == 10
    # All should be X/USDT pairs
    for p in DEFAULT_PEERS:
        assert p.endswith("/USDT"), f"Unexpected pair format: {p}"
