import streamlit as st
import joblib
import pandas as pd

model = joblib.load("trash_classifier_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("Smart Trash Classifier")

weight = st.number_input("Weight")
moisture = st.number_input("Moisture")
metal = st.number_input("Metal Detected")
plastic = st.number_input("Plastic Detected")
organic = st.number_input("Organic Level")

if st.button("Predict"):
    data = pd.DataFrame([{
        "weight": weight,
        "moisture": moisture,
        "metal_detected": metal,
        "plastic_detected": plastic,
        "organic_level": organic
    }])

    prediction = model.predict(data)
    result = encoder.inverse_transform(prediction)

    st.success(f"Predicted Waste Type: {result[0]}")