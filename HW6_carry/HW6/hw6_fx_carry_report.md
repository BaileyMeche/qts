# HW6 FX Carry Report

## Spec Recap

- Weekly USD 10MM lending, GBP 8MM borrowing at OIS+50bp, entry filter uses GBP 5Y funding rate: s5_lend >= s5_fund + 50bp.

- Lending MTM shifts all coupon/principal times by 1/52 and reprices on exit curve (dirty pricing via shifted cashflows).

## Data & Coverage

- EM curves sourced from ../../Data/ with fallback ./data (manifest has exact files).

- OIS/FX/funding curve raw from ./data_clean/.

- Weekly window: 2019-08-07 to 2024-05-29, N=4.

## Funding 5Y Construction

- GBP 5Y funding series constructed from local BoE raw OIS spot-curve file (not overnight proxy).

- Overnight OIS is used only for borrowing accrual +50bp.

## Methodology

- Par-curve bootstrap and discounting follow Zero/Spot curve conventions (linear interpolation + recursive DF solve).

- Cashflow schedule shift validation: times_exit_full = times_entry - 1/52 with assertion in notebook.

## Results

| ccy | weeks_available | weeks_traded | active_frac | mean_weekly_ret_cond_active | vol_weekly_ret_cond_active | mean_weekly_ret_uncond | vol_weekly_ret_uncond | sharpe_weekly_cond_active | sharpe_ann_cond_active | pnl_sum_usd | max_dd_wealth |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ZAR | 4 | 4 | 1.000000 | -0.102862 | 0.707374 | -0.102862 | 0.707374 | -0.145415 | -1.048600 | -1630638.371442 | -0.999999 |
| PKR | 4 | 4 | 1.000000 | -0.105383 | 0.674912 | -0.105383 | 0.674912 | -0.156143 | -1.125966 | -3559741.119065 | -0.999999 |
| NGN | 4 | 4 | 1.000000 | -0.370258 | 0.577496 | -0.370258 | 0.577496 | -0.641144 | -4.623354 | -8818172.324659 | -0.999999 |
| TRY | 4 | 4 | 1.000000 | -0.431071 | 0.485688 | -0.431071 | 0.485688 | -0.887547 | -6.400192 | -10394276.481219 | -1.000000 |
| BRL | 4 | 4 | 1.000000 | -0.663799 | 0.307992 | -0.663799 | 0.307992 | -2.155249 | -15.541722 | -6175872.887402 | -1.000000 |

| count | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|
| 4.000000 | -0.334675 | 0.457617 | -0.999999 | -0.422065 | -0.189021 | -0.101631 | 0.039343 |
| 4.000000 | 0.192646 | 0.385290 | 0.000001 | 0.000001 | 0.000001 | 0.192646 | 0.770580 |
| 4.000000 | -0.749999 | 0.500000 | -0.999999 | -0.999999 | -0.999999 | -0.749999 | 0.000000 |
| 4.000000 | 5.000000 | 0.000000 | 5.000000 | 5.000000 | 5.000000 | 5.000000 | 5.000000 |

_Figure omitted from deliverables (binary PNG removed)._

_Figure omitted from deliverables (binary PNG removed)._

_Figure omitted from deliverables (binary PNG removed)._

_Figure omitted from deliverables (binary PNG removed)._

## Drop-One Diagnostics

| portfolio | sharpe_weekly | sharpe_ann | max_dd_wealth |
|---|---|---|---|
| full | -0.731342 | -5.273785 | -0.999999 |
| ex_BRL | -0.491630 | -3.545197 | -0.999999 |
| ex_NGN | -0.710109 | -5.120670 | -0.999999 |
| ex_PKR | -0.892174 | -6.433556 | -0.999999 |
| ex_TRY | -0.655194 | -4.724673 | -0.999999 |
| ex_ZAR | -0.898139 | -6.476573 | -0.999999 |

## Market Factors

| index | corr_with_port |
|---|---|
| usd_proxy | 0.993354 |
| em_fx_basket | -0.618784 |
| rates_proxy_on | 0.641254 |
| rates_proxy_5y | 0.644501 |

| model | factor | alpha | beta | t_beta_hac4 | r2 | n |
|---|---|---|---|---|---|---|
| univariate | usd_proxy | -0.130720 | 15.316309 | 50.343581 | 0.986753 | 3 |
| univariate | em_fx_basket | -0.569252 | -0.715527 | -3.053683 | 0.382893 | 3 |
| univariate | rates_proxy_on | -0.575616 | 13.753722 | 3.236653 | 0.411206 | 3 |
| univariate | rates_proxy_5y | -0.577892 | 13.947713 | 3.264792 | 0.415381 | 3 |
| multivariate | const | -0.214111 | -0.214111 | -0.000693 | 0.999250 | 3 |
| multivariate | usd_proxy | -0.214111 | 13.570312 | 0.001439 | 0.999250 | 3 |
| multivariate | em_fx_basket | -0.214111 | 0.171875 | 0.000005 | 0.999250 | 3 |
| multivariate | rates_proxy_5y | -0.214111 | 6.750000 | 0.000027 | 0.999250 | 3 |

Carry framing: factor signs are consistent with carry-vs-crash intuition where USD/risk shocks can dominate carry accrual.

## Robustness & Guardrails

| series | missing_frac |
|---|---|
| ois_on | 0.000000 |
| gbp5_fund | 0.000000 |
| fx_BRL | 0.000000 |
| fx_GBP | 0.000000 |
| fx_NGN | 0.000000 |
| fx_PKR | 0.000000 |
| fx_TRY | 0.000000 |
| fx_ZAR | 0.000000 |
| curve_BRL | 0.000000 |
| curve_NGN | 0.000000 |
| curve_PKR | 0.000000 |
| curve_TRY | 0.000000 |
| curve_ZAR | 0.000000 |

| ccy | weeks_traded | active_frac | spread_mean | spread_p5 | spread_p50 | spread_p95 |
|---|---|---|---|---|---|---|
| BRL | 4 | 1.000000 | 0.054475 | 0.050813 | 0.053537 | 0.059452 |
| NGN | 4 | 1.000000 | 0.116023 | 0.106963 | 0.116818 | 0.123969 |
| PKR | 4 | 1.000000 | 0.118089 | 0.104946 | 0.118284 | 0.130958 |
| TRY | 4 | 1.000000 | 0.196872 | 0.138591 | 0.199384 | 0.251636 |
| ZAR | 4 | 1.000000 | 0.076794 | 0.068080 | 0.076939 | 0.085304 |
| PORTFOLIO | 20 | nan | nan | nan | 5.000000 | nan |
