import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv("smart_trash_classifier_dataset.csv")

# -----------------------------
# Target column
# -----------------------------
target = "material_type"

# -----------------------------
# Split features and label
# -----------------------------
X = data.drop(columns=[target])
y = data[target]

# -----------------------------
# Convert categorical columns
# -----------------------------
X = pd.get_dummies(X)

# -----------------------------
# Encode labels
# -----------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# -----------------------------
# Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train model (smaller size)
# -----------------------------
model = RandomForestClassifier(
    n_estimators=50,     # fewer trees = smaller model
    max_depth=10,        # limits tree size
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluate model
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# -----------------------------
# Save model (compressed)
# -----------------------------
joblib.dump(model, "trash_classifier_model.pkl", compress=3)

# Save label encoder
joblib.dump(label_encoder, "label_encoder.pkl")

print("Model saved as trash_classifier_model.pkl")
print("Label encoder saved as label_encoder.pkl")