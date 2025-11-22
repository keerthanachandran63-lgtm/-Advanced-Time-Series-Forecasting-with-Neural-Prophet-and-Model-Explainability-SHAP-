Advanced Time Series Forecasting with NeuralProphet + Surrogate SHAP Explainability (Option A)

Contents:
- data_generation.py        : generates multi-variate synthetic dataset (price, promotion, holiday)
- train_neuralprophet.py   : trains NeuralProphet with train/validation split, saves model, forecasts, metrics
- optuna_tune.py           : example Optuna study to tune key NeuralProphet hyperparameters
- surrogate_shap.py        : trains XGBoost surrogate on features, computes SHAP TreeExplainer, saves values and plots
- report.md                : automated report template describing results
- run_all.sh               : convenience script to run pipeline in order
