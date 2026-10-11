import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create output folder
os.makedirs("visualizations", exist_ok=True)

# Load employee data
df = pd.read_csv("employee_data.csv")

sns.set_theme(style="whitegrid")

# 1. Average performance by department
department_performance = (
    df.groupby("department")["performance_score"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 5))
department_performance.plot(kind="bar")
plt.title("Average Employee Performance by Department")
plt.xlabel("Department")
plt.ylabel("Average Performance Score")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("visualizations/department_performance.png", dpi=300)
plt.close()

# 2. KPI correlation heatmap
kpi_columns = [
    "task_completion",
    "quality_score",
    "attendance",
    "teamwork",
    "initiative",
    "communication",
]

plt.figure(figsize=(9, 6))
sns.heatmap(
    df[kpi_columns].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1,
)
plt.title("Employee KPI Correlation Heatmap")
plt.tight_layout()
plt.savefig("visualizations/kpi_correlation_heatmap.png", dpi=300)
plt.close()

# 3. Performance score distribution
plt.figure(figsize=(9, 5))
sns.histplot(
    data=df,
    x="performance_score",
    bins=15,
    kde=True,
)
plt.title("Distribution of Employee Performance Scores")
plt.xlabel("Performance Score")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.savefig("visualizations/performance_distribution.png", dpi=300)
plt.close()

print("Visualizations created successfully!")
print("1. visualizations/department_performance.png")
print("2. visualizations/kpi_correlation_heatmap.png")
print("3. visualizations/performance_distribution.png")
