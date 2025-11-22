
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv("dataset.csv")
model = ARIMA(df['y'], order=(5,1,2))
fit = model.fit()
forecast = fit.predict(len(df), len(df)+60)

print("ARIMA forecast complete.")
