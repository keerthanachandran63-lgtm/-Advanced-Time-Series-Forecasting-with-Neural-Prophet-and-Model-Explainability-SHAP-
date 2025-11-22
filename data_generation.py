import numpy as np
import pandas as pd

def generate_data(path='/mnt/data/neuralprophet_surrogate_project/dataset.csv', seed=42):
    np.random.seed(seed)
    dates = pd.date_range(start='2015-01-01', periods=2000, freq='D')
    # trend + two seasonalities
    trend = np.linspace(100, 300, len(dates))
    season_year = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    season_month = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 30.5)
    # external regressors
    price = 200 + 10 * np.sin(np.arange(len(dates))/20) + np.random.randn(len(dates)) * 5
    promotion = np.random.choice([0,1], size=len(dates), p=[0.88,0.12])
    holiday = np.random.choice([0,1], size=len(dates), p=[0.96,0.04])
    noise = np.random.randn(len(dates)) * 4
    y = trend + season_year + season_month + 0.35*price - 6*promotion + 25*holiday + noise
    df = pd.DataFrame({'ds': dates, 'y': y, 'price': price, 'promotion': promotion, 'holiday': holiday})
    df.to_csv(path, index=False)
    print('Saved dataset to', path)

if __name__ == '__main__':
    generate_data()
