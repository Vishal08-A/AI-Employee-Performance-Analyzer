
import pandas as pd
import os

print("=" * 50)
print("AI EMPLOYEE PERFORMANCE ANALYZER")
print("=" * 50)

# 1. Load employee dataset
file_path = "employee_data.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(
        "employee_data.csv not found. "
        "Place it in the project folder."
    )

df = pd.read_csv(file_path)

print("\n1. DATASET OVERVIEW")
print("-" * 30)
print("Total employees:", len(df))
print("Total columns:", len(df.columns))
print("Duplicate rows:", df.duplicated().sum())
print("Missing values:", df.isnull().sum().sum())

# 2. Check required KPI columns
kpi_weights = {
    "task_completion": 0.25,
    "quality_score": 0.20,
    "attendance": 0.10,
    "teamwork": 0.15,
    "initiative": 0.15,
    "communication": 0.15
}

required_columns = list(kpi_weights.keys())

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required KPI columns: {missing_columns}"
    )

# 3. Calculate weighted performance score
df["performance_score"] = sum(
    df[column] * weight
    for column, weight in kpi_weights.items()
)

# 4. Assign performance categories
df["performance_category"] = pd.cut(
    df["performance_score"],
    bins=[0, 60, 75, 85, 100],
    labels=[
        "Needs Improvement",
        "Average",
        "Good",
        "Excellent"
    ],
    include_lowest=True
)

print("\n2. PERFORMANCE ANALYSIS")
print("-" * 30)
print(
    "Average performance score:",
    round(df["performance_score"].mean(), 2)
)

print("\nPerformance category distribution:")
print(df["performance_category"].value_counts())

# 5. Generate synthetic attrition-risk labels
# These are rule-generated indicators, not actual resignations.
df["attrition_risk"] = (
    (df["overtime_hours"] > 30)
    | (df["performance_score"] < 60)
    | (df["training_hours"] < 20)
).astype(int)

print("\n3. SYNTHETIC ATTRITION-RISK ANALYSIS")
print("-" * 30)
print(df["attrition_risk"].value_counts().rename(
    index={0: "Lower risk", 1: "Higher risk"}
))

# 6. Department-level performance summary
department_summary = (
    df.groupby("department", observed=True)
    .agg(
        employee_count=("employee_id", "count"),
        average_performance=("performance_score", "mean"),
        average_attendance=("attendance", "mean"),
        average_quality=("quality_score", "mean")
    )
    .round(2)
)

print("\n4. DEPARTMENT PERFORMANCE SUMMARY")
print("-" * 30)
print(department_summary)

# 7. Department-level synthetic risk summary
risk_summary = (
    df.groupby("department", observed=True)
    .agg(
        employee_count=("employee_id", "count"),
        high_risk_labels=("attrition_risk", "sum")
    )
)

risk_summary["high_risk_percentage"] = (
    risk_summary["high_risk_labels"]
    / risk_summary["employee_count"] * 100
).round(2)

print("\n5. DEPARTMENT RISK SUMMARY")
print("-" * 30)
print(risk_summary)

# 8. Save enriched dataset and summaries
df.to_csv("employee_data.csv", index=False)
department_summary.to_csv("department_summary.csv")
risk_summary.to_csv("department_risk_summary.csv")

print("\n6. OUTPUT FILES")
print("-" * 30)
print("Saved: employee_data.csv")
print("Saved: department_summary.csv")
print("Saved: department_risk_summary.csv")

print("\nAnalysis completed successfully.")
