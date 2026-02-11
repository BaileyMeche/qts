from __future__ import annotations

import numpy as np
import pandas as pd

from hw5.metrics import metrics_by_ticker, pooled_metrics
from hw5.pipeline import compute_panel


def test_pipeline_smoke_shapes() -> None:
    idx = pd.date_range("2020-01-01", periods=80, freq="W-WED")
    m = pd.Series(np.random.default_rng(0).normal(0, 0.02, len(idx)), index=idx)
    eq = pd.DataFrame(
        {
            "AAA": 0.8 * m + np.random.default_rng(1).normal(0, 0.01, len(idx)),
            "BBB": 0.5 * m + np.random.default_rng(2).normal(0, 0.01, len(idx)),
        },
        index=idx,
    )
    cds = 0.3 * eq + np.random.default_rng(3).normal(0, 0.02, eq.shape)
    panel = compute_panel(cds, eq, m, boxcar_window=16, ew_window=16, ew_half_life=12.0)
    assert {"q_boxcar", "q_ew", "rho", "c"}.issubset(panel.columns)
    assert panel["ticker"].nunique() == 2
    mt = metrics_by_ticker(panel)
    pm = pooled_metrics(panel)
    assert not mt.empty
    assert not pm.empty
