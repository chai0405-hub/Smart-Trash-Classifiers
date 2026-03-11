import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
data = pd.read_csv("smart_trash_classifier_dataset.csv")

# 2. Target column
target = "material_type"

# 3. Separate features and label
X = data.drop(columns=[target])
y = data[target]

# 4. Convert categorical feature columns to numbers
X = pd.get_dummies(X)

# 5. Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# 7. Train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# 8. Predictions
y_pred = model.predict(X_test)

# 9. Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 10. Save model
joblib.dump(model, "trash_classifier_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\nModel saved successfully!")