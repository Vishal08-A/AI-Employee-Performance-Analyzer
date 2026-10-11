
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("employee_data.csv")

# 2. Select input features
features = [
    "task_completion",
    "quality_score",
    "attendance",
    "teamwork",
    "initiative",
    "communication"
]

X = df[features]
y = df["performance_score"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 4. Train Random Forest Regression model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# 5. Predict performance scores
y_pred = model.predict(X_test)

# 6. Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("EMPLOYEE PERFORMANCE MODEL")
print("-" * 35)
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²:   {r2:.3f}")

# 7. Feature importance
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nFEATURE IMPORTANCE")
print(importance.to_string(index=False))

# 8. Save test predictions
results = X_test.copy()
results["actual_score"] = y_test
results["predicted_score"] = y_pred.round(2)

results.to_csv("performance_predictions.csv", index=False)

print("\nSaved: performance_predictions.csv")
