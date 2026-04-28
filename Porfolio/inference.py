import joblib
import os
import pandas as pd
import json
import numpy as np
import sys
from io import BytesIO, StringIO

model_dir = os.environ.get('SM_MODEL_DIR')
if model_dir not in sys.path:
    sys.path.append(model_dir)


def model_fn(model_dir):
    """Load the finalized loan default pipeline from the model directory."""
    path = os.path.join(model_dir, 'finalized_loan_model.joblib')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    model = joblib.load(path)
    print("Model loaded successfully.")
    return model


def input_fn(request_body, request_content_type):
    """Parse the incoming request into a DataFrame."""
    print(f"Receiving data of type: {request_content_type}")

    if request_content_type == 'application/x-npy':
        data = np.load(BytesIO(request_body), allow_pickle=True)
        return pd.DataFrame(data)

    elif request_content_type == 'application/json':
        return pd.read_json(StringIO(request_body))

    elif request_content_type == 'text/csv':
        return pd.read_csv(StringIO(request_body))

    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_df, model):
    """Run the pipeline and return class predictions."""
    print("Running prediction pipeline...")
    return model.predict(input_df)


def output_fn(prediction, content_type):
    """Serialise the prediction array back to JSON."""
    print("Formatting output...")
    res = prediction.tolist() if isinstance(prediction, (np.ndarray, np.generic)) else prediction
    return json.dumps(res), "application/json"
