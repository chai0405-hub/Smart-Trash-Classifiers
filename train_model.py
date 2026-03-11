import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("smart_trash_classifier_dataset.csv")

# Target column
target = "material_type"

# Split features and target
X = df.drop(columns=[target])
y = df[target]

# Convert categorical columns
X = pd.get_dummies(X)

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model files
joblib.dump(model, "trash_classifier_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("Model files created successfully.")