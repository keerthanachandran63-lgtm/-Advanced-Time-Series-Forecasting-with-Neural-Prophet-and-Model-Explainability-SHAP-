# Example Optuna tuning script for NeuralProphet
# NOTE: Running this script may take significant time; this file is provided as an automated tuner reference.
import optuna
import pandas as pd
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_error
import json

def objective(trial):
    df = pd.read_csv('/mnt/data/neuralprophet_surrogate_project/dataset.csv', parse_dates=['ds'])
    split_idx = int(len(df)*0.7)
    train = df.iloc[:split_idx].copy()
    val = df.iloc[split_idx:].copy()

    n_lags = trial.suggest_int('n_lags', 7, 60)
    epochs = trial.suggest_int('epochs', 20, 100)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    n_forecasts = 1

    m = NeuralProphet(n_lags=n_lags, n_forecasts=n_forecasts, learning_rate=lr, epochs=epochs)
    m.add_future_regressor('price')
    m.add_future_regressor('promotion')
    m.add_future_regressor('holiday')

    m.fit(train, validation_df=val, freq='D', progress='none')
    future = m.make_future_dataframe(val, periods=0)
    forecast = m.predict(future)
    if 'yhat1' in forecast.columns:
        yhat = forecast['yhat1'].values[:len(val)]
    else:
        yhat = forecast[[c for c in forecast.columns if c.startswith('yhat')][0]].values[:len(val)]
    mae = mean_absolute_error(val['y'].values, yhat[:len(val)])
    return mae

if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)  # small number for example
    study.trials_dataframe().to_csv('/mnt/data/neuralprophet_surrogate_project/optuna_trials.csv', index=False)
    json.dump(study.best_params, open('/mnt/data/neuralprophet_surrogate_project/optuna_best_params.json','w'))
    print('Optuna tuning complete. Best params saved.')
