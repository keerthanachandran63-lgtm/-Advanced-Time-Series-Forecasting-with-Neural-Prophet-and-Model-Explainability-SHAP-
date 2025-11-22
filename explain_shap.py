
import shap
import pandas as pd
import xgboost as xgb

def run_shap():
    df = pd.read_csv('/mnt/data/final_project/dataset.csv')

    X = df[['price','promotion','holiday']]
    y = df['y']

    model = xgb.XGBRegressor()
    model.fit(X, y)

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    shap_df = pd.DataFrame(shap_values.values, columns=['price_shap','promotion_shap','holiday_shap'])
    shap_df.to_csv('/mnt/data/final_project/shap_values.csv', index=False)

if __name__ == '__main__':
    run_shap()
