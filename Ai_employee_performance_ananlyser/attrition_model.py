
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load the dataset
df = pd.read_csv("employee_data.csv")

# Select input features
features = [
    "performance_score",
    "overtime_hours",
    "training_hours",
    "attendance"
]

X = df[features]
y = df["attrition_risk"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train the classification model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predict risk labels
y_pred = model.predict(X_test)

# Evaluate the model
print("EMPLOYEE ATTRITION RISK MODEL")
print("-" * 35)
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred, zero_division=0))

# Save predictions
results = X_test.copy()
results["actual_risk"] = y_test
results["predicted_risk"] = y_pred

results.to_csv("attrition_predictions.csv", index=False)

print("\nSaved: attrition_predictions.csv")
