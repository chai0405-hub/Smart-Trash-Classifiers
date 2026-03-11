import pandas as pd
import joblib

# Load trained model and label encoder
model = joblib.load("trash_classifier_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Example input data (change values based on your sensors)
input_data = {
    "weight": [120],
    "moisture": [30],
    "metal_detected": [0],
    "plastic_detected": [1],
    "organic_level": [10]
}

# Convert to dataframe
df = pd.DataFrame(input_data)

# Convert categorical columns same way as training
df = pd.get_dummies(df)

# Predict
prediction = model.predict(df)

# Convert back to label
material = label_encoder.inverse_transform(prediction)

print("Predicted Waste Type:", material[0])