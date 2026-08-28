import json
import os
import joblib
import numpy as np
import pandas as pd


def model_fn(model_dir):
  """Loads the Scikit-Learn model artifact."""
  return joblib.load(os.path.join(model_dir, "property_valuation_model.pkl"))


def input_fn(request_body, request_content_type):
  """Parses JSON input payload."""
  if request_content_type == "application/json":
    return json.loads(request_body)
  raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, artifact):
  """Runs inference and computes confidence bounds."""
  model = artifact["model"]
  features = artifact["features"]

  if "parking_availability" not in input_data:
    input_data["parking_availability"] = 1

  input_df = pd.DataFrame([input_data])[features]
  log_pred = model.predict(input_df)
  base_price_usd = float(np.expm1(log_pred)[0])
  mae_val = 18541.37

  return {
      "status": "success",
      "base_price_usd": round(base_price_usd, 2),
      "lower_bound_usd": round(max(0, base_price_usd - mae_val), 2),
      "upper_bound_usd": round(base_price_usd + mae_val, 2),
  }


def output_fn(prediction, accept):
  """Returns JSON formatted response."""
  if accept == "application/json":
    return json.dumps(prediction), accept
  raise ValueError(f"Unsupported accept type: {accept}")
