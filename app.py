import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load model files
# -----------------------------
model = joblib.load("trash_classifier_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_columns = joblib.load("model_features.pkl")

st.title("♻️ Smart Trash Classifier")

st.write("Enter waste characteristics to predict the material type.")

# -----------------------------
# User inputs
# -----------------------------
weight = st.number_input("Weight (grams)", min_value=0.0)
moisture = st.number_input("Moisture (%)", min_value=0.0)
size = st.number_input("Size (cm)", min_value=0.0)

is_contaminated = st.selectbox("Is Contaminated?", [0,1])
recyclable = st.selectbox("Recyclable?", [0,1])

recommended_bin = st.selectbox(
    "Recommended Bin",
    ["organic","plastic","metal","cardboard"]
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Waste Type"):

    input_data = pd.DataFrame([{
        "weight_g": weight,
        "moisture_pct": moisture,
        "size_cm": size,
        "is_contaminated": is_contaminated,
        "recyclable": recyclable,
        "recommended_bin": recommended_bin
    }])

    # Convert categorical features
    input_data = pd.get_dummies(input_data)

    # Align columns with training features
    input_data = input_data.reindex(columns=feature_columns, fill_value=0)

    # Predict
    prediction = model.predict(input_data)

    predicted_label = label_encoder.inverse_transform(prediction)

    st.success(f"Predicted Waste Type: {predicted_label[0]}")