"""Tests for Hedge parameters in OhioConfig."""
from freqtrade.ohio.config.defaults import OhioConfig


def test_ohio_config_has_hedge_params():
    cfg = OhioConfig()
    assert cfg.hedge_eta == 0.1
    assert cfg.hedge_temperature == 2.0
    assert cfg.hedge_weight_floor == 0.05


def test_ohio_config_hedge_params_custom():
    cfg = OhioConfig(hedge_eta=0.2, hedge_temperature=3.0, hedge_weight_floor=0.10)
    assert cfg.hedge_eta == 0.2
    assert cfg.hedge_temperature == 3.0
    assert cfg.hedge_weight_floor == 0.10
