
import numpy as np
import pandas as pd

def generate_data():
    np.random.seed(42)
    dates = pd.date_range(start="2015-01-01", periods=2000, freq="D")

    trend = np.linspace(50, 120, len(dates))
    season1 = 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)
    season2 = 3 * np.sin(np.arange(len(dates)) * 2 * np.pi / 30)

    price = 200 + 10*np.sin(np.arange(len(dates))/20) + np.random.randn(len(dates))*5
    promotion = np.random.choice([0,1], size=len(dates), p=[0.85,0.15])
    holiday = np.random.choice([0,1], size=len(dates), p=[0.95,0.05])

    noise = np.random.randn(len(dates)) * 3
    y = trend + season1 + season2 + 0.3*price - 5*promotion + 20*holiday + noise

    df = pd.DataFrame({
        'ds': dates,
        'y': y,
        'price': price,
        'promotion': promotion,
        'holiday': holiday
    })
    df.to_csv('/mnt/data/final_project/dataset.csv', index=False)

if __name__ == '__main__':
    generate_data()
