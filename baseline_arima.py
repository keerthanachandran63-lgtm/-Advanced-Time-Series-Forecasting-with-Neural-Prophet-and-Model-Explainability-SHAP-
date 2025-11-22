
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def run_arima():
    df = pd.read_csv('/mnt/data/project/dataset.csv')
    model = ARIMA(df['y'], order=(5,1,2))
    model_fit = model.fit()
    preds = model_fit.forecast(60)
    pd.DataFrame({'y_pred': preds}).to_csv('/mnt/data/project/forecast_arima.csv', index=False)

if __name__ == '__main__':
    run_arima()
