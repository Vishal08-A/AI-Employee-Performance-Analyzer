#```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Load employee data
df = pd.read_csv("employee_data.csv")

# 2. Select KPI features
features = [
    "task_completion",
    "quality_score",
    "attendance",
    "teamwork",
    "initiative",
    "communication"
]

X = df[features]

# 3. Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Create three clusters
model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["employee_cluster"] = model.fit_predict(X_scaled)

# 5. Analyze cluster characteristics
summary = df.groupby("employee_cluster")[features].mean().round(2)

print("EMPLOYEE SEGMENTATION")
print("-" * 35)

print("\nEmployees in each cluster:")
print(df["employee_cluster"].value_counts().sort_index())

print("\nAverage KPI scores by cluster:")
print(summary)

# 6. Save results
df.to_csv("employee_clusters.csv", index=False)
summary.to_csv("cluster_summary.csv")

print("\nSaved: employee_clusters.csv")
print("Saved: cluster_summary.csv")
