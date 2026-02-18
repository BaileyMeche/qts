# HW6 — FX Carry Strategy Report

## Spec implemented (matches PDF)
- Weekly entry/exit on W-WED; nearest trading day within ±2 days for FX/OIS.
- Funding GBP: borrow at OIS+50bp on 4/5 notional (5x leverage).
- Lending: buy 5Y par bond, coupon fixed at entry 5Y swap rate, quarterly coupons.
- Filter: trade if (lend 5Y swap − fund 5Y swap) ≥ 50bp.
- MTM: reprice remaining CFs one week later on new curve (bootstrapped ZC from par curve).
- All flows in USD; returns shown as equity return (P&L / $2MM) and notional-normalized.

## Data + interpolation audit
- Curves densified to business days via time interpolation (max gap 45 bdays).
- Weekly curves sampled within ±7 days.
- Enforced: curve must cover **1Y and 5Y** at entry and exit.
- See: `curve_time_interpolation_audit.csv`, `alignment_staleness_summary.csv`, `entry_par_check.csv`.

## Performance
### Variable-capital portfolio (avg across active positions)
| n_weeks | mean_w | vol_w | sharpe_w | mean_ann | vol_ann | sharpe_ann | terminal_wealth | max_drawdown |
|---|---|---|---|---|---|---|---|---|
| 365 | -0.006287 | 0.118949 | -0.052851 | -0.326902 | 0.857751 | -0.381115 | 0.006521 | -0.998644 |

### Fixed-allocation portfolio (inactive=0)
| n_weeks | mean_w | vol_w | sharpe_w | mean_ann | vol_ann | sharpe_ann | terminal_wealth | max_drawdown |
|---|---|---|---|---|---|---|---|---|
| 365 | -0.001835 | 0.044655 | -0.041098 | -0.095433 | 0.322014 | -0.296365 | 0.355795 | -0.842076 |

## Correlations, risk factors, contributions
- Correlations: `currency_corr_all_weeks.csv` and `currency_corr_active_overlap.csv`
- Factor regression (HAC): `market_factor_regs.csv`
- Sharpe variance shares: `sharpe_variance_contrib_fixed.csv`
- Worst-10-week drawdown contributors: `drawdown_contrib_worst10w.csv`

## Figures
- `figures/wealth_fixed.png`
- `figures/drawdown_fixed.png`
- `figures/cum_pnl_by_ccy.png`
- `figures/corr_heatmap_all.png`
