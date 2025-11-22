
from neuralprophet import NeuralProphet
import pandas as pd

def train_model():
    df = pd.read_csv('/mnt/data/project/dataset.csv')
    m = NeuralProphet(n_lags=14, n_forecasts=1)
    m.add_future_regressor(name='exog')
    model = m.fit(df, freq='D')
    m.save('/mnt/data/project/np_model.np')
    future = m.make_future_dataframe(df, periods=60, n_historic_predictions=False)
    forecast = m.predict(future)
    forecast.to_csv('/mnt/data/project/forecast_np.csv', index=False)

if __name__ == '__main__':
    train_model()
