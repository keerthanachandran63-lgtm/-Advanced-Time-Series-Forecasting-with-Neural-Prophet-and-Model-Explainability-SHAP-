
import shap
import pandas as pd
import joblib

def run_shap():
    df = pd.read_csv('/mnt/data/project/dataset.csv')
    X = df[['exog']]
    y = df['y']

    model = joblib.load('/mnt/data/project/sk_model.pkl')
    explainer = shap.Explainer(model.predict, X)
    shap_values = explainer(X)

    shap_df = pd.DataFrame(shap_values.values, columns=['shap_exog'])
    shap_df.to_csv('/mnt/data/project/shap_values.csv', index=False)

if __name__ == '__main__':
    run_shap()
