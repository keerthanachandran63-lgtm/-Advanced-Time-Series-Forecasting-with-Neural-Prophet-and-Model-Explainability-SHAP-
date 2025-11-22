
from neuralprophet import NeuralProphet
import pandas as pd

def train_model():
    df = pd.read_csv('/mnt/data/final_project/dataset.csv')

    m = NeuralProphet(
        n_lags=30,
        n_forecasts=1,
        learning_rate=0.01,
        epochs=50
    )

    m.add_future_regressor('price')
    m.add_future_regressor('promotion')
    m.add_future_regressor('holiday')

    model = m.fit(df, freq='D')
    m.save('/mnt/data/final_project/np_model.np')

    future = m.make_future_dataframe(df, periods=90)
    forecast = m.predict(future)
    forecast.to_csv('/mnt/data/final_project/forecast_np.csv', index=False)

if __name__ == '__main__':
    train_model()
