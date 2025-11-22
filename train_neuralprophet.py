import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import json

def train_and_forecast(data_path='/mnt/data/neuralprophet_surrogate_project/dataset.csv',
                       out_dir='/mnt/data/neuralprophet_surrogate_project',
                       n_lags=30, n_forecasts=1, epochs=60, lr=0.01, val_pct=0.2, periods_forecast=90):
    df = pd.read_csv(data_path, parse_dates=['ds'])
    # train/validation split by time
    split_idx = int(len(df)*(1-val_pct) - periods_forecast)  # hold last 'periods_forecast' for final out-of-sample
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:-periods_forecast].copy()
    test_df = df.iloc[-periods_forecast:].copy()

    m = NeuralProphet(n_lags=n_lags, n_forecasts=n_forecasts, learning_rate=lr, epochs=epochs)
    # add regressors
    m.add_future_regressor('price')
    m.add_future_regressor('promotion')
    m.add_future_regressor('holiday')

    # fit using train and validation via 'val_df' passed as validation_df
    metrics = m.fit(train_df, freq='D', validation_df=val_df, progress='none')

    # save model
    m.save(out_dir + '/np_model.np')

    # create future dataframe: use full history + regressors for future
    future = m.make_future_dataframe(df, periods=periods_forecast)
    forecast = m.predict(future)
    # save forecast and metrics
    forecast.to_csv(out_dir + '/forecast_np_full.csv', index=False)

    # Evaluate on test set using last forecasted steps only: align predictions to test dates
    # NeuralProphet's forecast contains historic and future; select last periods_forecast rows matching test ds
    pred_future = forecast[forecast['ds'].isin(test_df['ds'])].copy()
    if 'yhat1' in pred_future.columns:
        yhat_col = 'yhat1'
    elif 'yhat' in pred_future.columns:
        yhat_col = 'yhat'
    else:
        # fallback: try first column starting with 'yhat'
        yhat_col = [c for c in pred_future.columns if c.startswith('yhat')][0]

    y_true = test_df['y'].values
    y_pred = pred_future[yhat_col].values[:len(y_true)]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    with open(out_dir + '/np_metrics.txt','w') as f:
        f.write(f"MAE: {mae}\nRMSE: {rmse}\n")

    # Save a small dataframe of actual vs predicted
    pd.DataFrame({'ds': test_df['ds'].values, 'actual': y_true, 'predicted': y_pred}).to_csv(out_dir + '/forecast_np_test.csv', index=False)

    # Save config used
    json.dump({'n_lags': n_lags, 'n_forecasts': n_forecasts, 'epochs': epochs, 'lr': lr}, open(out_dir + '/np_config.json','w'))

    print('Training complete. Metrics saved to', out_dir + '/np_metrics.txt')

if __name__ == '__main__':
    train_and_forecast()
