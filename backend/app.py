# backend/app.py
"""
Flask backend to load your saved MLP model + scaler + label encoder
and serve predictions for uploaded CSVs or JSON feature dicts.

Assumes saved models are in ./saved_models:
 - mlp_multiclass_model.keras
 - scaler.pkl
 - label_encoder.pkl
"""

from flask import Flask, request, jsonify, send_from_directory
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from model_utils import prepare_features_for_prediction

app = Flask(__name__, static_folder="../frontend", static_url_path="/")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
MODEL_PATH = os.path.join(MODEL_DIR, "mlp_multiclass_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
LE_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# load model and preprocessors once at start
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(LE_PATH)

@app.route("/")
def index():
    return send_from_directory("../frontend", "index.html")

@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    """
    Accepts multipart/form-data with file field 'file' (CSV)
    CSV should contain one or more rows of flow features with same column names used in training.
    """
    if 'file' not in request.files:
        return jsonify({"error": "no file uploaded (use form field 'file')"}), 400

    f = request.files['file']
    try:
        df = pd.read_csv(f)
    except Exception as e:
        return jsonify({"error": f"failed to read CSV: {e}"}), 400

    try:
        X = prepare_features_for_prediction(df, scaler)  # returns numpy array
    except Exception as e:
        return jsonify({"error": f"feature preparation failed: {e}"}), 400

    probs = model.predict(X)  # shape (n, num_classes)
    preds = np.argmax(probs, axis=1)
    classes = label_encoder.inverse_transform(preds)

    results = []
    for i in range(len(preds)):
        results.append({
            "pred_index": int(preds[i]),
            "pred_class": str(classes[i]),
            "confidence": float(np.max(probs[i]))
        })

    return jsonify({"predictions": results})

@app.route("/predict_json", methods=["POST"])
def predict_json():
    """
    Accepts a JSON body of either:
      - a list of feature dicts: [{col1:val1, col2:val2, ...}, ...]
      - or a single feature dict
    """
    content = request.get_json()
    if content is None:
        return jsonify({"error": "no JSON body"}), 400

    # normalize to list of dicts
    if isinstance(content, dict):
        items = [content]
    elif isinstance(content, list):
        items = content
    else:
        return jsonify({"error": "JSON must be dict or list of dicts"}), 400

    try:
        df = pd.DataFrame(items)
        X = prepare_features_for_prediction(df, scaler)
    except Exception as e:
        return jsonify({"error": f"failed to prepare features: {e}"}), 400

    probs = model.predict(X)
    preds = np.argmax(probs, axis=1)
    classes = label_encoder.inverse_transform(preds)
    results = []
    for i in range(len(preds)):
        results.append({
            "pred_index": int(preds[i]),
            "pred_class": str(classes[i]),
            "confidence": float(np.max(probs[i]))
        })
    return jsonify({"predictions": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
