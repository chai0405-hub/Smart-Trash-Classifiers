import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load model files
# -----------------------------
model = joblib.load("trash_classifier_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_columns = joblib.load("model_features.pkl")

# -----------------------------
# Page Title
# -----------------------------
st.set_page_config(page_title="Smart Trash Classifier")

st.title("♻️ Smart Trash Classifier")

# -----------------------------
# Banner Image
# -----------------------------
st.image("images/banner.jpg", use_container_width=True)

st.write("Predict the type of waste and see how it should be sorted.")

# -----------------------------
# Example Waste Types
# -----------------------------
st.subheader("Example Waste Types")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.image("images/plastic.jpg", caption="Plastic")

with col2:
    st.image("images/metal.jpg", caption="Metal")

with col3:
    st.image("images/cardboard.jpg", caption="Cardboard")

with col4:
    st.image("images/organic.jpg", caption="Organic")

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Enter Waste Details")

weight = st.number_input("Weight (grams)", min_value=0.0)
moisture = st.number_input("Moisture (%)", min_value=0.0)
size = st.number_input("Size (cm)", min_value=0.0)

is_contaminated = st.selectbox("Is Contaminated?", [0, 1])
recyclable = st.selectbox("Recyclable?", [0, 1])

recommended_bin = st.selectbox(
    "Recommended Bin",
    ["organic", "plastic", "metal", "cardboard"]
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

    predicted_label = label_encoder.inverse_transform(prediction)[0]

    st.success(f"Predicted Waste Type: {predicted_label}")

    # -----------------------------
    # Show predicted waste image
    # -----------------------------
    image_path = f"images/{predicted_label}.jpg"

    st.image(image_path, caption=f"{predicted_label} Waste", width=350)

    # -----------------------------
    # Show input summary
    # -----------------------------
    st.subheader("Input Summary")

    st.write(f"Weight: {weight} g")
    st.write(f"Moisture: {moisture} %")
    st.write(f"Size: {size} cm")

    if recyclable == 1:
        st.success("This waste can be recycled.")
    else:
        st.warning("This waste is not recyclable.")