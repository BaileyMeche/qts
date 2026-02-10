# HW5 Predictive Regression

## Run
From repo root:

```bash
python -m hw5.run_all
```

This expects execution from `HW5/` as working directory (or package import path includes `HW5`).

## Data
- Reads CDS from `HW5/Liq5YCDS.delim`.
- Uses `HW5/equity_adj_close.parquet` if present.
- If cache is missing, attempts to download adjusted closes for CDS tickers + `SPY` via yfinance and caches it.

## Outputs (`HW5/output/`)
- `panel_results.csv`: full per-ticker weekly panel with coefficients and residuals (`f`, `rho`, `c`, `mu`, `q`).
- `metrics_per_ticker.csv`: RMSE, OOS R², and tail stats per ticker/model.
- `metrics_pooled.csv`: pooled model comparison.
- `event_windows.csv`: weeks with largest pooled RMSE gaps (EW - boxcar).
- `rolling_rmse.png`, `error_gap.png`, `tail_qq.png`.
- `robustness/robustness_sweep.csv` and robustness heatmaps.
- `summary.md`: short generated summary scaffold.

## Tests
```bash
pytest HW5/tests -q
```

Covers:
- rolling OLS synthetic correctness,
- EW weighted-slope formula,
- no-lookahead invariance,
- `q_t` lag alignment,
- pipeline smoke metrics.
