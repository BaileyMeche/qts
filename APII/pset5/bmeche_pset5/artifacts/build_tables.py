from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
res = root / 'data' / 'Results'
art = root / 'artifacts'
art.mkdir(exist_ok=True)

mse = pd.read_csv(res / 'NN_vs_RF_MSE_woLAB.csv')
mse.to_csv(art / 'mse_comparison_table.csv', index=False)
mse['pct_improvement_nn_vs_rf'] = (mse['MSE_RF'] - mse['MSE_NN']) / mse['MSE_RF']
mse[['horizon','pct_improvement_nn_vs_rf']].to_csv(art / 'mse_improvement.csv', index=False)

def md(df: pd.DataFrame) -> str:
    cols = df.columns.tolist()
    lines = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    for _, r in df.iterrows():
        lines.append('| ' + ' | '.join(str(r[c]) for c in cols) + ' |')
    return '\n'.join(lines)

main = mse[['horizon','MSE_Analyst','MSE_RF','MSE_NN','NN_minus_RF','RF_minus_Analyst','NN_minus_Analyst']]
imp = mse[['horizon','pct_improvement_nn_vs_rf']]
(art / 'mse_comparison_table.md').write_text(
    '# MSE Comparison (woLAB)\n\n' + md(main) + '\n\n## Percent Improvement (NN vs RF)\n\n' + md(imp) + '\n'
)

nn = pd.read_parquet(res / 'NN_wo_lookahead_raw.parquet', engine='pyarrow')
(art / 'summary.txt').write_text(
    f"min_YearMonth: {nn['YearMonth'].min()}\n"
    f"max_YearMonth: {nn['YearMonth'].max()}\n"
    f"forecasted_rows: {len(nn)}\n"
    'analyst_forecasts_available: yes\n'
)
print('wrote artifacts tables')
