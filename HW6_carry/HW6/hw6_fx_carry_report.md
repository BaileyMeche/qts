# HW6 FX Carry Report

## 1) Spec recap

- Weekly strategy; lend $10MM in EM 5Y par bond, fund $8MM in GBP at OIS+50bp, equity $2MM.

- Entry filter uses true GBP 5Y funding series: trade if s5_lend >= s5_fund + 50bp.

- MTM uses explicit quarterly schedule shifted by 1/52 at exit; no separate accrual added to lending leg.

- USD accounting for all PnL and portfolio aggregation.

## 2) Data & coverage

- Weekly sample: 2019-01-02 to 2025-12-24, N=364.

- Inputs loaded from ./data_clean/ and curves from ../../Data/ or ./data fallback.

## 3) Methodology

- Curve pricing uses linear interpolation + recursive discount bootstrap from par rates (Zero/Spot style).

- Cashflow shift validation shown in notebook output (entry and exit times).

- GBP 5Y funding built from local BoE OIS spot-curve archive (not overnight proxy).

## 4) Results

| ccy | mean_weekly | vol_weekly | ann_sharpe | active_frac | pnl_sum_usd | max_dd_wealth | active_weeks |\n|---|---|---|---|---|---|---|---|\n| ZAR | 0.021989 | 0.091897 | 1.725490 | 0.192308 | 3078496.943509 | -0.373887 | 70 |\n| BRL | 0.008237 | 0.135203 | 0.439346 | 0.145604 | 873165.543808 | -0.638547 | 53 |\n| NGN | 0.002907 | 0.129803 | 0.161516 | 0.219780 | 465178.515834 | -0.555280 | 80 |\n| PKR | 0.001429 | 0.126561 | 0.081413 | 0.229651 | 225761.220578 | -0.539442 | 79 |\n| TRY | -0.024020 | 0.168761 | -1.026371 | 0.219780 | -3843214.857666 | -0.973415 | 80 |

| metric | value |

|---|---:|

| ann return | -0.007631 |

| ann vol | 0.281031 |

| ann sharpe | -0.027153 |

| max drawdown | -0.461001 |

_Figure omitted from deliverables (binary PNG removed)._

_Figure omitted from deliverables (binary PNG removed)._

_Figure omitted from deliverables (binary PNG removed)._

_Figure omitted from deliverables (binary PNG removed)._

## 5) Correlation and drop-one

| ccy | BRL | NGN | PKR | TRY | ZAR |\n|---|---|---|---|---|---|\n| BRL | 1.000 | -0.147 | 0.079 | 0.022 | 0.515 |\n| NGN | -0.147 | 1.000 | 0.157 | 0.175 | 0.022 |\n| PKR | 0.079 | 0.157 | 1.000 | 0.135 | 0.102 |\n| TRY | 0.022 | 0.175 | 0.135 | 1.000 | 0.199 |\n| ZAR | 0.515 | 0.022 | 0.102 | 0.199 | 1.000 |

| ccy_removed | ann_sharpe_drop_one | max_dd_drop_one | delta_sharpe | delta_dd |\n|---|---|---|---|---|\n| ZAR | -0.153720 | -0.559956 | -0.126566 | -0.098954 |\n| BRL | -0.087383 | -0.461001 | -0.060230 | 0.000000 |\n| PKR | -0.061044 | -0.470277 | -0.033891 | -0.009276 |\n| NGN | -0.033646 | -0.526476 | -0.006493 | -0.065475 |\n| TRY | 0.275621 | -0.374737 | 0.302774 | 0.086264 |

## 6) Market Factors

- No external factor files found; used labeled proxy factors.

| index | corr_with_port |\n|---|---|\n| proxy_usd_broad | -0.047165 |\n| proxy_em_risk | 0.058133 |\n| proxy_d_ois | 0.033843 |\n| proxy_d_gbp5y | 0.019783 |

| model | factor | alpha | beta | t_beta | r2 | n |\n|---|---|---|---|---|---|---|\n| univariate | proxy_usd_broad | -0.000473 | -0.238050 | -0.897138 | 0.002225 | 363 |\n| univariate | proxy_em_risk | 0.000302 | 0.174578 | 1.106400 | 0.003379 | 363 |\n| univariate | proxy_d_ois | -0.000271 | 1.483105 | 0.643379 | 0.001145 | 363 |\n| univariate | proxy_d_gbp5y | -0.000216 | 0.008539 | 0.375953 | 0.000391 | 363 |\n| multivariate | const | -0.000267 | -0.000267 | -0.127394 | 0.018429 | 363 |\n| multivariate | proxy_usd_broad | -0.000267 | -0.769631 | -2.226651 | 0.018429 | 363 |\n| multivariate | proxy_em_risk | -0.000267 | 0.472970 | 2.301592 | 0.018429 | 363 |\n| multivariate | proxy_d_ois | -0.000267 | 1.532638 | 0.629424 | 0.018429 | 363 |\n| multivariate | proxy_d_gbp5y | -0.000267 | 0.019586 | 0.792510 | 0.018429 | 363 |

Carry/crash interpretation: FX/risk proxies explain part of variation; carry can be dominated by adverse FX shocks in risk-off episodes.

## 7) Robustness & limitations

- FX inversion checks, >20% move flags, weekly missingness diagnostics, par PV sanity, and drawdown assertion included.

- Limitation: available local GBP curve data is reconstructed from BoE raw archive format and may not provide full pillar metadata.

