import streamlit as st
import pandas as pd
import joblib

model = joblib.load("trash_classifier_model.pkl")
encoder = joblib.load("label_encoder.pkl")
features = joblib.load("model_features.pkl")

st.title("Smart Trash Classifier")

weight = st.number_input("Weight (g)")
moisture = st.number_input("Moisture (%)")
size = st.number_input("Size (cm)")
contaminated = st.selectbox("Is Contaminated", [0,1])
recyclable = st.selectbox("Recyclable", [0,1])
bin_type = st.selectbox("Recommended Bin", ["plastic","metal","organic","cardboard"])

if st.button("Predict"):

    data = pd.DataFrame([{
        "weight_g": weight,
        "moisture_pct": moisture,
        "size_cm": size,
        "is_contaminated": contaminated,
        "recyclable": recyclable,
        "recommended_bin": bin_type
    }])

    data = pd.get_dummies(data)

    # Align with training columns
    data = data.reindex(columns=features, fill_value=0)

    prediction = model.predict(data)

    result = encoder.inverse_transform(prediction)

    st.success(f"Predicted Waste Type: {result[0]}")