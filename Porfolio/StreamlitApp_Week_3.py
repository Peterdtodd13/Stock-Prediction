import os
import sys
import warnings
import tempfile
import tarfile

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

import shap

warnings.simplefilter("ignore")

# Fix path for Streamlit Cloud (ensure 'src' is findable)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import extract_features  # noqa: E402

# Access Streamlit secrets
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]


@st.cache_resource
def get_session(_aws_id, _aws_secret, _aws_token):
    return boto3.Session(
        aws_access_key_id=_aws_id,
        aws_secret_access_key=_aws_secret,
        aws_session_token=_aws_token,
        region_name="us-east-1",
    )


session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# Data & Model Configuration
df_features = extract_features()

MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": "explainer.shap",
    "pipeline": "finalized_model.tar.gz",
    # These are the exact feature names the app expects from the form and the model.
    "keys": ["NVDA", "TSLA", "DEXCAUS", "DEXMXUS", "SP500", "DJIA", "NASDAQCOM"],
}

MODEL_INFO["inputs"] = [
    {"name": k, "type": "number", "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01}
    for k in MODEL_INFO["keys"]
]


def load_pipeline(_session, bucket, key_prefix):
    """(Optional) Load a local sklearn pipeline artifact from S3."""
    s3_client = _session.client("s3")
    filename = MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"{key_prefix}/{os.path.basename(filename)}",
    )

    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith(".joblib")][0]

    return joblib.load(joblib_file)


def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client("s3")

    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)

    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)


def call_model_api(input_df: pd.DataFrame):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer(),
    )

    try:
        raw_pred = predictor.predict(input_df)
        pred_val = pd.DataFrame(raw_pred).values[-1][0]
        return round(float(pred_val), 4), 200
    except Exception as e:
        return f"Error: {str(e)}", 500


def display_explanation(input_df: pd.DataFrame, _session, _aws_bucket: str):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(
        _session,
        _aws_bucket,
        posixpath.join("explainer", explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name),
    )
    shap_values = explainer(input_df)

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)

    top_feature = shap_values[0].feature_names[0]
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")


# Streamlit UI
st.set_page_config(page_title="ML Deployment", layout="wide")
st.title("👨‍💻 ML Deployment")

with st.form("pred_form"):
    st.subheader("Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                inp["name"].replace("_", " ").upper(),
                min_value=float(inp["min"]),
                max_value=float(inp["max"]),
                value=float(inp["default"]),
                step=float(inp["step"]),
            )

    submitted = st.form_submit_button("Run Prediction")


if submitted:
    missing = [k for k in MODEL_INFO["keys"] if k not in user_inputs]
    if missing:
        st.error(f"Missing inputs: {missing}")
        st.stop()

    # Build the new row in the exact order expected by the model
    data_row = [float(user_inputs[k]) for k in MODEL_INFO["keys"]]
    new_row_df = pd.DataFrame([data_row], columns=MODEL_INFO["keys"])

    # Ensure df_features columns align (and reorder if needed)
    base_df = df_features.copy()
    base_df = base_df.reindex(columns=MODEL_INFO["keys"])

    input_df = pd.concat([base_df, new_row_df], ignore_index=True)

    res, status = call_model_api(input_df)
    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(input_df, session, aws_bucket)
    else:
        st.error(res)
