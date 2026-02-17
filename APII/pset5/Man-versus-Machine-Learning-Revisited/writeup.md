# PS5 Part 2 Write-up (Execution Attempt)

## Setup and execution
I ran the pipeline from repository root `Man-versus-Machine-Learning-Revisited/` and executed the required environment checks and data validation steps. Commands and outputs are recorded in `artifacts/RUNLOG.txt`.

Executed commands (high level):
1. `python --version`
2. `pip install -r requirements.txt` (file missing)
3. `pip install numpy pandas pyarrow fastparquet scikit-learn jupyter nbconvert matplotlib tqdm torch`
4. Data inventory checks for `data/WRDS`, `data/Macro`, `data/Other`.

## Deviations from defaults
- `requirements.txt` is not present in this workspace copy, so explicit package installation was attempted.
- `fastparquet` installation failed due proxy/network restrictions (403), though `numpy/pandas/pyarrow` are already installed.
- The pipeline cannot proceed past Step 2 because `data/WRDS` is empty.

## Results
Real NN vs RF results could not be produced in this workspace because preprocessing inputs are absent.

See placeholder table in `artifacts/mse_comparison_table.md`. Numeric non-NA values require the WRDS raw inputs and successful execution of:
- `code/01_Preprocess.ipynb`
- `code/02_EarningsForecasts.ipynb`
- `code/run_nn_forecast.py --mode woLAB --evaluate`

## Interpretation
No empirical interpretation is possible without generated RF and NN outputs. Once WRDS inputs are available, the table can be populated and interpreted horizon-by-horizon (q1, q2, q3, y1, y2) using MSE differences and percent improvement.

## Repro commands
```bash
python --version
pip install -r requirements.txt
pip install numpy pandas pyarrow fastparquet scikit-learn jupyter nbconvert matplotlib tqdm torch
find data/WRDS -maxdepth 2 -type f | head
jupyter nbconvert --execute --to notebook --inplace code/01_Preprocess.ipynb
jupyter nbconvert --execute --to notebook --inplace code/02_EarningsForecasts.ipynb
cd code && python run_nn_forecast.py --mode woLAB --evaluate
```
