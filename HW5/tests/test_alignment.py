from __future__ import annotations

import numpy as np
import pandas as pd

from hw5.pipeline import compute_panel


def _toy_data(n: int = 60) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    idx = pd.date_range("2020-01-01", periods=n, freq="W-WED")
    m = pd.Series(np.sin(np.arange(n) / 5) * 0.01, index=idx)
    eq = pd.DataFrame({"AAA": 0.4 * m + 0.001 * np.arange(n)}, index=idx)
    cds = pd.DataFrame({"AAA": 0.7 * eq["AAA"] + 0.2 * (0.7 * eq["AAA"])}, index=idx)
    return cds, eq, m


def test_no_lookahead_coeffs_unchanged_when_future_altered() -> None:
    cds, eq, m = _toy_data()
    p1 = compute_panel(cds, eq, m, boxcar_window=16, ew_window=16, ew_half_life=12.0)

    cds2 = cds.copy()
    cds2.iloc[-1, 0] += 5.0
    p2 = compute_panel(cds2, eq, m, boxcar_window=16, ew_window=16, ew_half_life=12.0)

    merged = p1.merge(p2, on=["date", "ticker"], suffixes=("_a", "_b"))
    early = merged.iloc[:-1]
    assert np.allclose(early["beta_equity_a"], early["beta_equity_b"], equal_nan=True)
    assert np.allclose(early["mu_boxcar_a"], early["mu_boxcar_b"], equal_nan=True)


def test_q_lag_alignment_definition() -> None:
    cds, eq, m = _toy_data()
    p = compute_panel(cds, eq, m, boxcar_window=16, ew_window=16, ew_half_life=12.0)
    g = p[p["ticker"] == "AAA"].set_index("date")
    lhs = g["q_boxcar"]
    rhs = g["rho"] - g["mu_boxcar"].shift(1) * g["c"].shift(1)
    assert np.allclose(lhs, rhs, equal_nan=True)
