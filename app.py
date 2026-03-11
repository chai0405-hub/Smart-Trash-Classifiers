import os
import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# ---------------------------------------------------
# Flask App Initialization
# ---------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------
# Model Paths
# ---------------------------------------------------
MODEL_PATH = "trash_classifier_model.pkl"
ENCODER_PATH = "label_encoder.pkl"

# ---------------------------------------------------
# Load Model and Encoder
# ---------------------------------------------------
try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("✅ Model and encoder loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", str(e))
    model = None
    label_encoder = None

# ---------------------------------------------------
# Helper Function: Align Input Features
# ---------------------------------------------------
def prepare_features(input_json):
    """
    Convert incoming JSON to dataframe
    and align with model feature names.
    """

    df = pd.DataFrame([input_json])

    # Convert categorical columns
    df = pd.get_dummies(df)

    # Align with training features
    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)

        for col in expected_features:
            if col not in df.columns:
                df[col] = 0

        df = df[expected_features]

    return df


# ---------------------------------------------------
# Root Route
# ---------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Smart Trash AI API is running",
        "status": "OK"
    })


# ---------------------------------------------------
# Health Check
# ---------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "model_loaded": model is not None,
        "encoder_loaded": label_encoder is not None
    })


# ---------------------------------------------------
# Prediction Route
# ---------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:

        data = request.get_json()

        if data is None:
            return jsonify({
                "error": "Invalid input. JSON expected."
            }), 400

        # Prepare features
        df = prepare_features(data)

        # Model prediction
        prediction = model.predict(df)

        # Convert numeric label to text
        label = label_encoder.inverse_transform(prediction)

        return jsonify({
            "input": data,
            "predicted_waste_type": label[0]
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# ---------------------------------------------------
# Batch Prediction
# ---------------------------------------------------
@app.route("/predict_batch", methods=["POST"])
def predict_batch():

    try:

        data = request.get_json()

        if not isinstance(data, list):
            return jsonify({
                "error": "Expected a list of JSON objects"
            }), 400

        df = pd.DataFrame(data)

        df = pd.get_dummies(df)

        if hasattr(model, "feature_names_in_"):
            expected = model.feature_names_in_

            for col in expected:
                if col not in df.columns:
                    df[col] = 0

            df = df[expected]

        predictions = model.predict(df)

        labels = label_encoder.inverse_transform(predictions)

        results = []

        for i, row in enumerate(data):
            results.append({
                "input": row,
                "prediction": labels[i]
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# ---------------------------------------------------
# Example Input Route
# ---------------------------------------------------
@app.route("/example", methods=["GET"])
def example():
    return jsonify({
        "example_request": {
            "weight": 120,
            "moisture": 20,
            "metal_detected": 0,
            "plastic_detected": 1,
            "organic_level": 10
        }
    })


# ---------------------------------------------------
# Run Flask App
# ---------------------------------------------------
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )