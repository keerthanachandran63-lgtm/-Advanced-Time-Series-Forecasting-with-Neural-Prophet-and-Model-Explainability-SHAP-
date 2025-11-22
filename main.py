
from neuralprophet import NeuralProphet
import pandas as pd
import pickle, os

df = pd.read_csv("dataset.csv")
m = NeuralProphet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

m = m.add_lagged_regressor(["price","promotion","holiday"])
m = m.add_future_regressor(["price","promotion","holiday"])

train_size = int(len(df)*0.85)
train_df = df[:train_size]

model = m.fit(train_df, freq="D")
future = m.make_future_dataframe(df, periods=60, n_historic_predictions=True)
forecast = m.predict(future)

os.makedirs("artifacts", exist_ok=True)
forecast.to_csv("artifacts/forecast.csv", index=False)

with open("artifacts/neuralprophet_model.pkl","wb") as f:
    pickle.dump(model,f)

print("Training complete.")
