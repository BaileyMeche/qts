# HW6 FX Carry (GBP funding) — Final Report

## 1) Assignment mechanics implemented
- Weekly traded cross-currency positions with USD asset notional of $10MM and GBP funding notional of $8MM (5x leverage proxy).
- Borrowing rate is OIS + 50bp each week.
- Entry rule enforced exactly as: hold if `s5_lend >= s5_fund + 50bp`, where `s5_fund` is a GBP 5Y rate extracted from local BoE OIS spot-curve archive (not overnight SONIA).
- Lending leg uses a 5Y par bond with quarterly coupons and coupon fixed at entry `s5_lend`; one-week MTM reprices on new curve with all payment times shifted by `1/52`.
- Accounting is in USD for both lending and funding legs.

## 2) Data and coverage
- Assignment specification used: `FXCarryBasedOnGBP.pdf`.
- Curves data read from `HW6/data/` (also searched `../../Data/` per instruction).
- OIS/FX data read from `HW6/data_clean/`.
- GBP 5Y funding series constructed from `data_clean/boe_ois_daily_archive_raw.parquet` using the `UK OIS spot curve` block, aligned to SONIA business-date index by source-file year windows.
- Backtest weekly window: 2019-01-02 to 2025-12-24 (364 weeks), Wednesdays with ±2-day tolerance (prefer delay).

## 3) Method details (Zero/Spot style conventions)
- Term-structure handling follows a standard par-curve bootstrap pattern: linear interpolation over pillars, recursive discount-factor solve on quarterly grid, fixed-coupon PV from discount factors.
- Repricing uses explicit payment schedule: entry `[0.25, ..., 5.00]`, exit `[0.25-1/52, ..., 5.00-1/52]` after dropping non-positive times.
- Validation cell confirms each payment time decreased by `1/52` within floating tolerance.

## 4) Diagnostics and guardrails
- FX direction sanity: GBP USD-per-GBP constrained to reasonable range and inversion guarded; large daily FX jumps (>20%) were counted and printed.
- Par-bond sanity checks on random weeks/currencies produced `par_price ≈ 1.0` on own curve.
- Active diagnostics table:
| ccy | active_frac |
|---|---|
| BRL | 0.145604 |
| NGN | 0.219780 |
| PKR | 0.229651 |
| TRY | 0.219780 |
| ZAR | 0.192308 |
| ALL_AVG_ACTIVE_POS | 0.994505 |

## 5) Per-currency results
| ccy | mean_weekly | vol_weekly | ann_sharpe | active_frac | pnl_sum_usd | max_dd_wealth | active_weeks |
|---|---|---|---|---|---|---|---|
| ZAR | 0.021989 | 0.091897 | 1.725490 | 0.192308 | 3078496.943509 | -0.373887 | 70 |
| BRL | 0.008237 | 0.135203 | 0.439346 | 0.145604 | 873165.543808 | -0.638547 | 53 |
| NGN | 0.002907 | 0.129803 | 0.161516 | 0.219780 | 465178.515834 | -0.555280 | 80 |
| PKR | 0.001429 | 0.126561 | 0.081413 | 0.229651 | 225761.220578 | -0.539442 | 79 |
| TRY | -0.024020 | 0.168761 | -1.026371 | 0.219780 | -3843214.857666 | -0.973415 | 80 |

## 6) Portfolio results
| metric | value |
|---|---:|
| annualized return (52*mean) | -0.007631 |
| annualized vol | 0.281031 |
| annualized Sharpe | -0.027153 |
| max drawdown (wealth) | -0.461001 |

_Figure omitted from deliverables (binary PNG filtered out)._

_Figure omitted from deliverables (binary PNG filtered out)._

## 7) Cross-currency correlation and drop-one diagnostics
| ccy | BRL | NGN | PKR | TRY | ZAR |
|---|---|---|---|---|---|
| BRL | 1.000 | -0.147 | 0.079 | 0.022 | 0.515 |
| NGN | -0.147 | 1.000 | 0.157 | 0.175 | 0.022 |
| PKR | 0.079 | 0.157 | 1.000 | 0.135 | 0.102 |
| TRY | 0.022 | 0.175 | 0.135 | 1.000 | 0.199 |
| ZAR | 0.515 | 0.022 | 0.102 | 0.199 | 1.000 |

_Figure omitted from deliverables (binary PNG filtered out)._

Drop-one diagnostics:
| ccy_removed | ann_sharpe_drop_one | max_dd_drop_one | delta_sharpe | delta_dd |
|---|---|---|---|---|
| ZAR | -0.153720 | -0.559956 | -0.126566 | -0.098954 |
| BRL | -0.087383 | -0.461001 | -0.060230 | 0.000000 |
| PKR | -0.061044 | -0.470277 | -0.033891 | -0.009276 |
| NGN | -0.033646 | -0.526476 | -0.006493 | -0.065475 |
| TRY | 0.275621 | -0.374737 | 0.302774 | 0.086264 |

Interpretation: currencies with most negative `delta_sharpe` are strongest contributors to portfolio Sharpe; currencies with most positive `delta_dd` help reduce drawdown when removed.

## 8) Market factor analysis (required)
- No external factor files were found in `data_clean/` matching VIX/DXY/SPX/MSCI/UST/rates/factors patterns; proxy factors were built internally and labeled as proxies.
- Proxy definitions:
  - `proxy_usd_broad`: average of USD strength vs GBP (`-Δlog(USD_per_GBP)`) and EMFX basket move.
  - `proxy_emfx_basket_ret`: equal-weight weekly log-return basket of EM FX USD/CCY series.
  - `proxy_rates_d_gbp5y`: weekly change in GBP 5Y funding series.
  - `proxy_rates_d_sonia`: weekly change in SONIA overnight.

Correlations with portfolio return:
| index | corr_with_port |
|---|---|
| proxy_usd_broad | -0.047165 |
| proxy_emfx_basket_ret | 0.058133 |
| proxy_rates_d_gbp5y | 0.019783 |
| proxy_rates_d_sonia | 0.033843 |

Regression output (univariate + multivariate):
| model | factor | alpha | beta | t_beta | r2 | n |
|---|---|---|---|---|---|---|
| univariate | proxy_usd_broad | -0.000473 | -0.238050 | -0.897138 | 0.002225 | 363 |
| univariate | proxy_emfx_basket_ret | 0.000302 | 0.174578 | 1.106400 | 0.003379 | 363 |
| univariate | proxy_rates_d_gbp5y | -0.000216 | 0.008539 | 0.375953 | 0.000391 | 363 |
| univariate | proxy_rates_d_sonia | -0.000271 | 1.483105 | 0.643379 | 0.001145 | 363 |
| multivariate | const | -0.000267 | -0.000267 | -0.127394 | 0.018429 | 363 |
| multivariate | proxy_usd_broad | -0.000267 | -0.769631 | -2.226651 | 0.018429 | 363 |
| multivariate | proxy_emfx_basket_ret | -0.000267 | 0.472970 | 2.301592 | 0.018429 | 363 |
| multivariate | proxy_rates_d_gbp5y | -0.000267 | 0.019586 | 0.792510 | 0.018429 | 363 |
| multivariate | proxy_rates_d_sonia | -0.000267 | 1.532638 | 0.629424 | 0.018429 | 363 |

These results are broadly consistent with carry intuition: short-horizon FX and risk-off shocks can dominate pure carry accrual, and factor explanatory power is modest in this sample.

## 9) Robustness and limitations
- Robustness checks included: alignment tolerance, FX inversion checks, explicit MTM time-shift verification, and par-pricing sanity checks.
- Limitation: `Zero_And_Spot_Curves.ipynb` and `Carry_Concept__FTSE.pdf` were not present in this repository snapshot, so methodology/framing was implemented from assignment requirements and available local data only.
- GBP 5Y is reconstructed from BoE raw spot-curve archive structure; mapping of raw values to business dates uses source-file year windows and should be treated as a data-engineering approximation.