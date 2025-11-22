
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

def run_arima():
    df = pd.read_csv('/mnt/data/final_project/dataset.csv')

    train = df['y'][:-90]
    test = df['y'][-90:]

    model = ARIMA(train, order=(5,1,3))
    model_fit = model.fit()

    preds = model_fit.forecast(90)
    preds = pd.Series(preds)

    mae = mean_absolute_error(test, preds)
    rmse = np.sqrt(mean_squared_error(test, preds))

    pd.DataFrame({
        'actual': test.values,
        'predicted': preds.values
    }).to_csv('/mnt/data/final_project/forecast_arima.csv', index=False)

    with open('/mnt/data/final_project/arima_metrics.txt','w') as f:
        f.write(f"MAE: {mae}\nRMSE: {rmse}")

if __name__ == '__main__':
    run_arima()
