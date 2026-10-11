
import streamlit as st
import pandas as pd
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Employee Performance Analytics",
    page_icon="📊",
    layout="wide"
)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("employee_data.csv")

df = load_data()

# Dashboard title
st.title("📊 Employee Performance Analytics")
st.caption("HR Analytics | 150 Synthetic Employee Records")

# Sidebar filters
st.sidebar.header("Filters")

departments = ["All"] + sorted(df["department"].unique().tolist())
selected_department = st.sidebar.selectbox(
    "Select Department", departments
)

categories = ["All"] + sorted(
    df["performance_category"].dropna().unique().tolist()
)
selected_category = st.sidebar.selectbox(
    "Select Performance Category", categories
)

# Apply filters
filtered_df = df.copy()

if selected_department != "All":
    filtered_df = filtered_df[
        filtered_df["department"] == selected_department
    ]

if selected_category != "All":
    filtered_df = filtered_df[
        filtered_df["performance_category"] == selected_category
    ]

# KPI cards
col1, col2, col3, col4 = st.columns(4)

col1.metric("Employees", len(filtered_df))

col2.metric(
    "Average Performance",
    f"{filtered_df['performance_score'].mean():.2f}"
    if not filtered_df.empty else "N/A"
)

col3.metric(
    "Excellent Performers",
    int((filtered_df["performance_category"] == "Excellent").sum())
)

col4.metric(
    "Synthetic High-Risk Labels",
    int((filtered_df["attrition_risk"] == 1).sum())
)

st.divider()

# Performance category distribution
left, right = st.columns(2)

with left:
    st.subheader("Performance Category Distribution")

    category_counts = (
        filtered_df["performance_category"]
        .value_counts()
        .reindex(
            ["Needs Improvement", "Average", "Good", "Excellent"],
            fill_value=0
        )
        .rename_axis("Category")
        .reset_index(name="Employees")
    )

    fig_category = px.bar(
        category_counts,
        x="Category",
        y="Employees",
        color="Category",
        title="Employees by Performance Level",
        text="Employees"
    )

    st.plotly_chart(fig_category, use_container_width=True)

with right:
    st.subheader("Department Performance")

    department_summary = (
        filtered_df.groupby("department", as_index=False)
        .agg(
            average_performance=("performance_score", "mean"),
            employee_count=("employee_id", "count")
        )
    )

    fig_department = px.bar(
        department_summary,
        x="department",
        y="average_performance",
        color="department",
        text_auto=".2f",
        title="Average Performance by Department"
    )

    st.plotly_chart(fig_department, use_container_width=True)

# Attrition-risk analysis
st.subheader("Synthetic Attrition-Risk Analysis")

risk_summary = (
    filtered_df["attrition_risk"]
    .value_counts()
    .reindex([0, 1], fill_value=0)
    .rename_axis("Risk Label")
    .reset_index(name="Employees")
)

risk_summary["Risk Label"] = risk_summary["Risk Label"].map({
    0: "Low Risk Label",
    1: "High Risk Label"
})

fig_risk = px.pie(
    risk_summary,
    names="Risk Label",
    values="Employees",
    hole=0.45,
    title="Distribution of Synthetic Risk Labels"
)

st.plotly_chart(fig_risk, use_container_width=True)

# Department-wise business analysis
st.divider()
st.header("Department-wise Business Analysis")

department_analysis = (
    filtered_df.groupby("department", as_index=False)
    .agg(
        employee_count=("employee_id", "count"),
        average_performance=("performance_score", "mean"),
        average_attendance=("attendance", "mean"),
        average_overtime=("overtime_hours", "mean"),
        average_quality=("quality_score", "mean"),
        average_teamwork=("teamwork", "mean")
    )
    .round(2)
)

st.subheader("Department KPI Summary")
st.dataframe(
    department_analysis,
    use_container_width=True,
    hide_index=True
)

# Highest and lowest performing departments
if not department_analysis.empty:
    highest = department_analysis.loc[
        department_analysis["average_performance"].idxmax()
    ]
    lowest = department_analysis.loc[
        department_analysis["average_performance"].idxmin()
    ]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Top Department",
        highest["department"],
        f"{highest['average_performance']:.2f} average score"
    )

    col2.metric(
        "Lowest Department",
        lowest["department"],
        f"{lowest['average_performance']:.2f} average score"
    )

    performance_gap = (
        highest["average_performance"]
        - lowest["average_performance"]
    )

    col3.metric(
        "Performance Gap",
        f"{performance_gap:.2f} points"
    )

# Department-wise synthetic risk labels
st.subheader("Synthetic Risk Labels by Department")

department_risk = (
    filtered_df.groupby("department", as_index=False)
    .agg(
        employee_count=("employee_id", "count"),
        high_risk_labels=("attrition_risk", "sum")
    )
)

department_risk["high_risk_percentage"] = (
    department_risk["high_risk_labels"]
    / department_risk["employee_count"] * 100
).round(2)

fig_department_risk = px.bar(
    department_risk,
    x="department",
    y="high_risk_percentage",
    text="high_risk_percentage",
    color="department",
    title="High-Risk Labels by Department (%)",
    labels={
        "department": "Department",
        "high_risk_percentage": "High-Risk Labels (%)"
    }
)

st.plotly_chart(
    fig_department_risk,
    use_container_width=True
)


# Employee details
st.subheader("Employee Details")

display_columns = [
    "employee_id",
    "name",
    "department",
    "performance_score",
    "performance_category",
    "attrition_risk"
]

st.dataframe(
    filtered_df[display_columns].sort_values(
        "performance_score", ascending=False
    ),
    use_container_width=True,
    hide_index=True
)

# Download filtered records
csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Filtered Employee Data",
    data=csv_data,
    file_name="filtered_employee_data.csv",
    mime="text/csv"
)

# Employee Clustering Analysis
st.header("👥 Employee Clustering Analysis")

cluster_file = "employee_clusters.csv"

try:
    cluster_df = pd.read_csv(cluster_file)

    st.subheader("Employees by Cluster")

    cluster_counts = (
        cluster_df["employee_cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    cluster_counts.columns = ["Cluster", "Employee Count"]

    fig_cluster = px.bar(
        cluster_counts,
        x="Cluster",
        y="Employee Count",
        title="Number of Employees in Each Cluster",
        text="Employee Count"
    )

    st.plotly_chart(fig_cluster, use_container_width=True)

    st.subheader("Average KPI Scores by Cluster")

    cluster_summary = pd.read_csv("cluster_summary.csv", index_col=0)

    st.dataframe(cluster_summary, use_container_width=True)

except FileNotFoundError:
    st.warning(
        "Clustering files are missing. "
        "Run clustering_model.py first."
    )
