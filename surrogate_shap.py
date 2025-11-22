import pandas as pd
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt

def train_surrogate_and_shap(data_path='/mnt/data/neuralprophet_surrogate_project/dataset.csv',
                             forecast_path='/mnt/data/neuralprophet_surrogate_project/forecast_np_test.csv',
                             out_dir='/mnt/data/neuralprophet_surrogate_project'):
    # Load original dataset and NP test forecast
    df = pd.read_csv(data_path, parse_dates=['ds'])
    forecast = pd.read_csv(forecast_path, parse_dates=['ds'])

    # Merge features with the NP model predictions on ds
    merged = pd.merge(forecast, df, on='ds', how='left', suffixes=('_pred',''))

    # Features used for surrogate model (tabular)
    X = merged[['price','promotion','holiday']].copy()
    y = merged['predicted'].values  # surrogate tries to mimic NeuralProphet predictions

    # Train an XGBoost regressor as surrogate
    model = xgb.XGBRegressor(n_estimators=200, max_depth=4, objective='reg:squarederror', random_state=42)
    model.fit(X, y)

    # Save surrogate
    joblib.dump(model, out_dir + '/surrogate_xgb.pkl')

    # SHAP explainability using TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Save raw SHAP values to CSV
    shap_df = pd.DataFrame(shap_values, columns=['price_shap','promotion_shap','holiday_shap'])
    shap_df.to_csv(out_dir + '/shap_values.csv', index=False)

    # Save a summary plot (matplotlib)
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(out_dir + '/shap_summary.png', dpi=150)
    plt.close()

    # Dependence plot for 'price'
    plt.figure()
    shap.dependence_plot('price', shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(out_dir + '/shap_dependence_price.png', dpi=150)
    plt.close()

    print('Surrogate model and SHAP outputs saved to', out_dir)

if __name__ == '__main__':
    train_surrogate_and_shap()
