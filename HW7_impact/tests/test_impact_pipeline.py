from __future__ import annotations

import numpy as np
import pandas as pd

from impact_estimation.impact_config import ImpactConfig
from impact_estimation.impact_core import (
    add_markouts,
    collapse_multihit_trades,
    fit_power_model,
    fit_sqrt_model,
    infer_side_signs,
    run_exchange_and_pooled,
    trim_outliers_and_bin,
)


def test_collapse_multihit_qty_vwap():
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime([
                "2024-01-01T00:00:00.001Z",
                "2024-01-01T00:00:00.0014Z",
                "2024-01-01T00:00:00.002Z",
            ]),
            "exchange": ["X", "X", "X"],
            "side": ["BUY", "BUY", "SELL"],
            "qty": [1.0, 3.0, 2.0],
            "price": [10.0, 14.0, 20.0],
        }
    )
    out = collapse_multihit_trades(df, "1ms")
    buy = out[out["side"] == "BUY"].iloc[0]
    assert np.isclose(buy["qty"], 4.0)
    assert np.isclose(buy["price"], (10 * 1 + 14 * 3) / 4)


def test_infer_side_signs_best_mapping():
    n = 60
    ts = pd.date_range("2024-01-01", periods=n, freq="s", tz="UTC")
    price = np.linspace(100, 160, n)
    side = np.array(["X", "Y"] * (n // 2))
    side[-10:] = "X"
    df = pd.DataFrame({"ts": ts, "price": price, "side": side, "qty": 1.0, "exchange": "E"})
    mapping = infer_side_signs(df, M=1)
    assert set(mapping.values()) == {1, -1}


def test_trim_outliers_and_bin_assignment():
    df = pd.DataFrame({"V": np.arange(1, 101), "y": 0.01, "ts": pd.Timestamp("2024-01-01"), "qbin": ""})
    out = trim_outliers_and_bin(df, q_outlier=0.95, q_bins=((0.5, 0.8), (0.8, 0.95)))
    assert out["V"].max() <= df["V"].quantile(0.95)
    assert out["qbin"].str.contains("Q").all()


def test_fit_sqrt_model_synthetic():
    rng = np.random.default_rng(0)
    V = rng.uniform(1, 100, 6000)
    mu0_true = 0.01
    y = mu0_true * np.sqrt(V)
    df = pd.DataFrame({"V": V, "y": y})
    res = fit_sqrt_model(df, hac_lags=5)
    assert res["mu0"] >= 0
    assert np.isclose(res["mu0"], mu0_true, atol=5e-4)


def test_fit_power_model_recovery():
    rng = np.random.default_rng(1)
    V = rng.uniform(1, 300, 5000)
    mu1_true, mu2_true = 0.004, 0.6
    y = mu1_true * np.power(V, mu2_true) + rng.normal(0, 1e-4, size=V.shape[0])
    df = pd.DataFrame({"V": V, "y": y})
    res = fit_power_model(df)
    assert res["success"]
    assert np.isclose(res["mu1"], mu1_true, rtol=0.2)
    assert np.isclose(res["mu2"], mu2_true, atol=0.08)


def test_end_to_end_exchange_and_pooled_non_empty():
    n = 18000
    ts = pd.date_range("2024-01-01", periods=n, freq="s", tz="UTC")
    ex = np.where(np.arange(n) % 2 == 0, "EX1", "EX2")
    side = np.where(np.arange(n) % 3 == 0, "BUY", "SELL")
    qty = 1 + (np.arange(n) % 100)
    price = 100 + np.cumsum(np.where(side == "BUY", 0.01, -0.005))
    df = pd.DataFrame({"ts": ts, "exchange": ex, "side": side, "qty": qty, "price": price})

    cfg = ImpactConfig(M=1, min_n=500, q_bins=((0.2, 0.5), (0.5, 0.8), (0.8, 0.95)))
    m = infer_side_signs(df, M=cfg.M)
    markouts = add_markouts(df, M=cfg.M, side_sign=m)
    assert not markouts.empty

    out = run_exchange_and_pooled(df, cfg)
    assert not out.empty
    assert set(out["label"]).issuperset({"ALL"})
