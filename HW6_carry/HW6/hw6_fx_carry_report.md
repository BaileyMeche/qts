# HW6 FX Carry (GBP Funding) — Execution Report

## 1) Data inputs and coverage
- Notebook: `HW6/HW6_FX_Carry_Trade.ipynb`.
- Input folders: `HW6/data/` (EM curves) and `HW6/data_clean/` (OIS + FX).
- Sample used for backtest: **2019-01-02 to 2025-12-24** (365 weekly observations).
- Weekly sampling uses **Wednesdays (`W-WED`)** with ±2-day tolerance and preference for delayed observations (`t, t+1, t+2, t-1, t-2`).

## 2) Methodology and USD cash-flow accounting
- Strategy evaluates each EM currency weekly with USD asset notional of **$10MM**.
- Funding leg borrows **$8MM equivalent in GBP** at **OIS + 50bp** (5x leverage proxy).
- Lending leg buys a **5Y par bond** in local currency (quarterly coupons, coupon = entry `s5`).
- Entry condition: trade only if `s5_lend >= s5_fund + 50bp`.
- Since GBP swap pillars were not present in provided curve files, `s5_fund` threshold uses **UK OIS proxy** only for the entry filter; borrowing accrual remains OIS+50bp.
- One-week MTM: repriced lending leg with new local curve and tenor `5 - 1/52`; borrowing leg uses FX move plus one-week accrual with ΔV=0 assumption in funding bond.

## 3) Curve bootstrap / pricing implementation
- Linear interpolation over observed curve tenors.
- Recursive quarterly discount-factor bootstrap from par-rate identity.
- Par-bond sanity check included in notebook (`par price ≈ 1.0`).

## 4) Per-currency performance
| ccy | mean_weekly | vol_weekly | sharpe_weekly | max_drawdown | active_weeks |
|---|---|---|---|---|---|
| BRL | -0.004406 | 0.119596 | -0.036840 | -0.994292 | 365 |
| ZAR | -0.003718 | 0.084829 | -0.043825 | -0.979875 | 365 |
| NGN | -0.011380 | 0.199911 | -0.056925 | -1.146011 | 365 |
| PKR | -0.023254 | 0.115817 | -0.200783 | -0.999989 | 345 |
| TRY | -0.042414 | 0.211014 | -0.201001 | -1.000414 | 365 |

## 5) Portfolio performance (equal-weight over active positions)
| Metric | Value |
|---|---:|
| Mean weekly return | -0.016921 |
| Weekly volatility | 0.080687 |
| Weekly Sharpe | -0.209713 |
| Approx annualized return | -0.879905 |
| Approx annualized vol | 0.581846 |
| Max drawdown | -0.999568 |
| Average active positions | 4.945205 |

_Figure omitted from deliverables (binary PNG removed)._

## 6) Correlation across currency strategy returns
| ccy | BRL | NGN | PKR | TRY | ZAR |
|---|---|---|---|---|---|
| BRL | 1.000 | 0.027 | 0.071 | 0.105 | 0.392 |
| NGN | 0.027 | 1.000 | 0.188 | 0.077 | 0.084 |
| PKR | 0.071 | 0.188 | 1.000 | 0.077 | 0.107 |
| TRY | 0.105 | 0.077 | 0.077 | 1.000 | 0.060 |
| ZAR | 0.392 | 0.084 | 0.107 | 0.060 | 1.000 |

_Figure omitted from deliverables (binary PNG removed)._

## 7) Contribution diagnostics (best/worst)
### Best Sharpe contributors (standalone)
| ccy | mean_weekly | vol_weekly | sharpe_weekly | max_drawdown | active_weeks |
|---|---|---|---|---|---|
| BRL | -0.004406 | 0.119596 | -0.036840 | -0.994292 | 365 |
| ZAR | -0.003718 | 0.084829 | -0.043825 | -0.979875 | 365 |

### Worst drawdown contributors (standalone)
| ccy | mean_weekly | vol_weekly | sharpe_weekly | max_drawdown | active_weeks |
|---|---|---|---|---|---|
| NGN | -0.011380 | 0.199911 | -0.056925 | -1.146011 | 365 |
| TRY | -0.042414 | 0.211014 | -0.201001 | -1.000414 | 365 |

### Drop-one portfolio Sharpe diagnostic
| ccy | drop_one_sharpe | delta_vs_base |
|---|---|---|
| NGN | -0.227483 | -0.017770 |
| BRL | -0.221356 | -0.011642 |
| ZAR | -0.216449 | -0.006736 |
| PKR | -0.170402 | 0.039311 |
| TRY | -0.131668 | 0.078045 |

Interpretation: removing TRY or PKR improves portfolio Sharpe most; removing NGN/BRL worsens Sharpe.

## 8) Robustness checks implemented
- Wednesday alignment with tolerance and deterministic tie-break.
- Data-schema robustness for OIS/FX/curve file layouts; FX quote orientation check via GBP sanity range.
- Interpolation fallback where tenors are incomplete.

## 9) Carry intuition
- Carry comes from rate differential, but short-horizon FX moves can dominate and induce crash-like losses.
- Negative realized Sharpe here suggests FX depreciation and repricing outweighed coupon carry.

## 10) Limitations
- `Carry_Concept__FTSE.pdf` and `Zero_And_Spot_Curves.ipynb` were not found in this repo snapshot, so equivalent methodology was implemented from first principles.
- GBP 5Y swap series unavailable; OIS proxy used for threshold only.
- Linear interpolation and sparse curve points may bias short-horizon MTM.
