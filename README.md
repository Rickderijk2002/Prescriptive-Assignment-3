

# JM0100 – Assignment 3: Predictive Maintenance

Group project for the course Prescriptive Algorithms (2025–2026). The project consists of two parts: a prediction task and an optimization task, applied to a maintenance scheduling problem for airplane engines.

## What this project does

**Prediction Task:** An XGBoost model is trained on run-to-failure engine data to predict the Remaining Useful Lifetime (RUL) of 100 engines currently in operation. Predictions are validated using 5-fold cross-validation and compared against an external consultancy benchmark using a statistical test.

**Optimization Task:** A Genetic Algorithm (GA) built with DEAP uses the predicted RUL values to schedule maintenance teams across engines, minimising penalty costs over a 30-day planning horizon.

## Files

| File | Description |
|---|---|
| `DataTrain.csv` | Training data — 100 engines, run-to-failure |
| `DataSchedule.csv` | Engines currently in use — RUL to be predicted |
| `RUL_consultancy_predictions_A3.csv` | External RUL benchmark for comparison |
| `D-XXX.ipynb` | Main notebook containing all code |
| `D-XXX.pdf` | Report |

## How to run

Clone the repository and make sure the data files are in the same directory as the notebook. All file paths are relative. Install dependencies with:

```
pip install xgboost deap scikit-learn pandas numpy matplotlib scipy
```

Then run the notebook top to bottom.

## Group members

Stefan, Rick, Tycho
