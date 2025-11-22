
import shap, pandas as pd
import pickle

df = pd.read_csv("dataset.csv")
with open("artifacts/neuralprophet_model.pkl","rb") as f:
    model = pickle.load(f)

explainer = shap.Explainer(lambda x: x, df[['price','promotion','holiday']])
shap_values = explainer(df[['price','promotion','holiday']])

shap.summary_plot(shap_values, df[['price','promotion','holiday']])
