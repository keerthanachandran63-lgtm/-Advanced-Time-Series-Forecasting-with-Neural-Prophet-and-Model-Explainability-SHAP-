
# Advanced Time Series Forecasting with NeuralProphet & SHAP

## 1. Dataset
Simulated multi-variate dataset with:
- Trend
- Weekly + yearly seasonality
- External regressors: price, promotion, holiday
- Noise + holiday spikes

## 2. Model
NeuralProphet with:
- Yearly + weekly seasonality
- Lagged regressors
- Future regressors

## 3. Explainability
SHAP summary identifies:
- Promotion → highest positive impact
- Holiday → next strongest
- Price → negative impact

## 4. Baseline Comparison
Evaluated vs ARIMA(5,1,2). NeuralProphet captures seasonality + regressors better.

## 5. Deliverables
- Dataset
- Forecast artifacts
- SHAP code
- ARIMA baseline
