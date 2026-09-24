import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
df = pd.read_csv("Healthcare-Diabetes.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# Features and target
X = df.drop(columns=["Id", "Outcome"])
y = df["Outcome"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# Train model
model.fit(X_train, y_train)


# Prediction
y_pred = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Random Forest Accuracy:", accuracy)


# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Save model
joblib.dump(model, "diabetes_model.pkl")


# Save feature names
feature_names = X.columns.tolist()
joblib.dump(feature_names, "feature_names.pkl")


print("\nModel saved successfully!")
print("Created: diabetes_model.pkl")
print("Created: feature_names.pkl")