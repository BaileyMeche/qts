# PS5 Part 2 Write-up (Man vs. Machine Learning Revisited)

## 1) Replication setup
- Working directory: `APII/pset5/Man-versus-Machine-Learning-Revisited`
- Python: 3.12.12 (see `RUNLOG.txt`)
- Neural-network files integrated under:
  - `code/functions/nn_forecast.py`
  - `code/run_nn_forecast.py`
  - `code/07_NeuralNetwork.ipynb`
- Executed via CLI and notebook automation attempts (`jupyter nbconvert --execute`).

## 2) Deviations and constraints
- The provided `Man-versus-Machine-Learning-Revisited` folder was empty (missing upstream notebooks and `requirements.txt`).
- Network/proxy restrictions prevented downloading packages/data (`pip` and GitHub requests returned 403 proxy tunnel errors).
- Because upstream files were unavailable, WRDS and shared-dropbox fallbacks could not be executed in this environment.

## 3) Results
The required upstream artifacts were not present:
- `data/Results/df_train_new.parquet`
- `data/Results/RF_wo_lookahead_raw_005.parquet`

Therefore, NN training/evaluation could not be completed with real numeric outputs in this environment. Placeholder result tables are provided in `artifacts/` to preserve expected submission schema.

## 4) Reproduction steps (intended)
```bash
cd APII/pset5/Man-versus-Machine-Learning-Revisited
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --execute --to notebook --inplace code/00_DataDownload.ipynb
jupyter nbconvert --execute --to notebook --inplace code/01_Preprocess.ipynb
jupyter nbconvert --execute --to notebook --inplace code/02_EarningsForecasts.ipynb
cd code
python run_nn_forecast.py --mode woLAB --evaluate
```

## 5) Exact unblock action needed
To finish with real results, place the following files under `data/Results/` (or run upstream notebooks in a WRDS-enabled setup):
1. `df_train_new.parquet`
2. `RF_wo_lookahead_raw_005.parquet`

Then run:
```bash
cd code
python run_nn_forecast.py --mode woLAB --evaluate
```
This will produce:
- `data/Results/NN_wo_lookahead_raw.parquet`
- `data/Results/NN_vs_RF_MSE_woLAB.csv`
