import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.pipeline import Pipeline

warnings.simplefilter("ignore")

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Load X_train baseline
file_path = os.path.join(current_dir, 'X_train.csv')
dataset = pd.read_csv(file_path)
if 'Unnamed: 0' in dataset.columns:
    dataset = dataset.drop(['Unnamed: 0'], axis=1)

# Load model and explainer from local joblib files
@st.cache_resource
def load_model():
    path = os.path.join(current_dir, 'finalized_loan_model.joblib')
    return joblib.load(path)

@st.cache_resource
def load_explainer():
    path = os.path.join(current_dir, 'shap_explainer_loan.joblib')
    return joblib.load(path)

model     = load_model()
explainer = load_explainer()

# Model Configuration
MODEL_INFO = {
    "keys": ['int_rate', 'fico_range_low', 'dti', 'revol_util'],
    "inputs": [
        {"name": "int_rate",       "label": "Interest Rate (%)",         "min": 5.0,   "max": 30.0,  "default": 13.5, "step": 0.1},
        {"name": "fico_range_low", "label": "FICO Score",                "min": 580.0, "max": 850.0, "default": 700.0,"step": 1.0},
        {"name": "dti",            "label": "Debt-to-Income Ratio (%)",  "min": 0.0,   "max": 45.0,  "default": 18.0, "step": 0.1},
        {"name": "revol_util",     "label": "Revolving Utilisation (%)", "min": 0.0,   "max": 100.0, "default": 45.0, "step": 0.1},
    ]
}

# Prediction
def call_model(input_df):
    try:
        pred_val = model.predict(input_df)[0]
        mapping  = {0: "Fully Paid (Low Risk)", 1: "Charged Off (High Risk)"}
        return mapping.get(int(pred_val), str(pred_val)), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

# SHAP Explanation
def display_explanation(input_df):
    preprocessing = Pipeline(steps=model.steps[:-1])
    input_transformed = preprocessing.transform(input_df)
    feature_names = dataset.columns.tolist()
    input_transformed = pd.DataFrame(input_transformed, columns=feature_names)

    shap_values = explainer.shap_values(input_transformed)

    st.subheader("Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    sv = shap_values[0]
    order = np.argsort(np.abs(sv))[::-1][:10]
    feats = [feature_names[i] for i in order]
    vals  = sv[order]
    colors = ["#dc2626" if v > 0 else "#16a34a" for v in vals]
    ax.barh(feats[::-1], vals[::-1], color=colors[::-1], edgecolor='white')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on default probability)")
    ax.set_title("Top 10 Feature Contributions", fontweight='bold')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#dc2626', label='Increases risk'),
                       Patch(color='#16a34a', label='Decreases risk')],
              loc='lower right', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    top_feature = pd.Series(np.abs(sv), index=feature_names).idxmax()
    st.info(f"**Business Insight:** The most influential factor in this prediction was **{top_feature}**.")

# Streamlit UI
st.set_page_config(page_title="Loan Default Predictor", page_icon="🏦", layout="wide")
st.title("🏦 Loan Default Prediction")
st.caption("Machine Learning in Finance - Milestone 4 | XGBoost Model | LendingClub Dataset")

with st.form("pred_form"):
    st.subheader("Applicant Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['label'],
                min_value=float(inp['min']),
                max_value=float(inp['max']),
                value=float(inp['default']),
                step=float(inp['step'])
            )

    submitted = st.form_submit_button("Run Prediction")

# Build full input row from X_train baseline + user inputs
original = dataset.iloc[0:1].copy()
for key, val in user_inputs.items():
    if key in original.columns:
        original[key] = val
input_df = original[dataset.columns]

if submitted:
    res, status = call_model(input_df)
    if status == 200:
        if "High Risk" in res:
            st.error(f"Prediction: **{res}**")
        else:
            st.success(f"Prediction: **{res}**")
        st.metric("Prediction Result", res)
        display_explanation(input_df)
    else:
        st.error(res)
