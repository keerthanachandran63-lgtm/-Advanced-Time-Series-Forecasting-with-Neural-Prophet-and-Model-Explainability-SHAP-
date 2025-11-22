
import numpy as np
import pandas as pd

def generate_data():
    np.random.seed(42)
    dates = pd.date_range(start="2015-01-01", periods=2000, freq="D")
    trend = np.linspace(10, 50, len(dates))
    season = 5 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)
    exog = np.random.randn(len(dates)) * 3
    noise = np.random.randn(len(dates)) * 2
    y = trend + season + 0.8 * exog + noise

    df = pd.DataFrame({'ds': dates, 'y': y, 'exog': exog})
    df.to_csv('/mnt/data/project/dataset.csv', index=False)

if __name__ == '__main__':
    generate_data()
