
import os
import pandas as pd

print("PROJECT VALIDATION")
print("=" * 40)

# 1. Check required files
required_files = [
    "employee_data.csv",
    "performance_predictions.csv",
    "attrition_predictions.csv",
    "employee_clusters.csv",
    "cluster_summary.csv",
    "dashboard.py"
]

print("\n1. FILE CHECK")
for file in required_files:
    status = "FOUND" if os.path.exists(file) else "MISSING"
    print(f"{file}: {status}")

# 2. Check employee dataset
print("\n2. DATASET CHECK")
df = pd.read_csv("employee_data.csv")

print("Employee records:", len(df))
print("Duplicate rows:", df.duplicated().sum())
print("Missing values:", df.isnull().sum().sum())

# 3. Check performance scores
print("\n3. PERFORMANCE SCORE CHECK")
if "performance_score" in df.columns:
    print("Minimum score:", round(df["performance_score"].min(), 2))
    print("Maximum score:", round(df["performance_score"].max(), 2))
    print("Missing scores:", df["performance_score"].isnull().sum())
else:
    print("performance_score column is missing")

# 4. Check clustering output
print("\n4. CLUSTERING CHECK")
if os.path.exists("employee_clusters.csv"):
    clusters = pd.read_csv("employee_clusters.csv")

    print("Cluster column exists:",
          "employee_cluster" in clusters.columns)

    if "employee_cluster" in clusters.columns:
        print("\nEmployee count by cluster:")
        print(clusters["employee_cluster"].value_counts().sort_index())

# 5. Final status
print("\nVALIDATION CHECK COMPLETE")

