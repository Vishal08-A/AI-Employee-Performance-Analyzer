# AI Employee Performance Analyzer

**Task ID:** AI-RH-006  
**Student Code:** DAS011288

## 1. Project Overview

The AI Employee Performance Analyzer is a Python-based HR analytics project that analyzes employee performance using key performance indicators (KPIs). It calculates performance scores, categorizes employees, analyzes synthetic attrition-risk labels, and groups employees using clustering techniques.

The project includes an interactive dashboard to help users explore employee performance and department-level trends.

## 2. Objectives

- Analyze employee KPI data.
- Calculate weighted employee performance scores.
- Categorize employees based on performance.
- Build a machine learning model to estimate performance scores.
- Classify synthetic attrition-risk labels.
- Group employees using K-Means clustering.
- Visualize results through an interactive dashboard.

## 3. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Plotly
- CSV datasets

## 4. Key Features

- Employee dataset analysis and validation.
- Weighted KPI-based performance scoring.
- Random Forest regression for performance-score estimation.
- Random Forest classification for synthetic attrition-risk labels.
- K-Means clustering for employee grouping.
- Department performance summaries.
- Interactive charts, employee filters, and CSV downloads.

## 5. Dataset

The project uses a **synthetic dataset containing 150 employee records**. It includes department, experience, task completion, quality, attendance, teamwork, initiative, communication, training hours, completed projects, and overtime hours.

The dataset is intended for demonstration and educational purposes.

## 6. Models and Analysis

### Performance Prediction

A Random Forest Regressor estimates performance scores from employee KPI data. The current test results are:

- MAE: 2.247
- RMSE: 3.100
- R²: 0.805

The target score is calculated from the same KPI variables using predefined weights. Therefore, these results measure how well the model approximates that scoring formula; they do not establish real-world employee-performance prediction accuracy.

### Attrition-Risk Classification

A Random Forest Classifier predicts labels generated from predefined rules involving overtime, performance score, and training hours. These labels are synthetic risk indicators, not actual employee resignations. The model has not been validated against real employee departures.

### Employee Clustering

K-Means groups employees into three clusters based on standardized KPI values. The clusters help explore different KPI patterns; cluster numbers do not indicate performance rankings.

## 7. Dashboard

The Streamlit dashboard provides:

- Employee and department performance metrics.
- Performance-category visualizations.
- Synthetic attrition-risk distribution.
- Employee-level data exploration.
- Employee clustering chart and KPI summary.
- CSV download functionality.

## 8. Project Structure

- `main.py` — Dataset processing and KPI analysis.
- `performance_model.py` — Performance-score regression.
- `attrition_model.py` — Synthetic attrition-risk classification.
- `clustering_model.py` — Employee clustering.
- `dashboard.py` — Interactive dashboard.
- `validate_project.py` — Project validation checks.
- `employee_data.csv` — Employee dataset.
- `performance_predictions.csv` — Performance-model output.
- `attrition_predictions.csv` — Attrition-classification output.
- `employee_clusters.csv` — Employee cluster assignments.
- `cluster_summary.csv` — KPI summary by cluster.
- `department_summary.csv` — Department performance summary.
- `department_risk_summary.csv` — Department risk summary.

## 9. How to Run

Run these commands in the VS Code terminal from the project folder, with the project virtual environment activated.

```powershell
python main.py
python performance_model.py
python attrition_model.py
python clustering_model.py
python validate_project.py
python -m streamlit run dashboard.py
```

Run the scripts in the listed order. The last command launches the interactive dashboard.

## 10. Validation Results

The current validation run reported:

- Employee records: 150
- Duplicate rows: 0
- Missing values: 0
- Performance score range: 58.18–86.21
- Employees in Cluster 0: 37
- Employees in Cluster 1: 56
- Employees in Cluster 2: 57

## 11. Limitations

- The dataset is synthetic and may not represent real employees.
- KPI weights are illustrative assumptions.
- Performance scores are calculated using a predefined formula.
- Attrition-risk labels are generated using rules rather than actual resignation records.
- Results should not be used to make real employee hiring, promotion, or termination decisions.

## 12. Conclusion

The project demonstrates how Python, machine learning, clustering, and interactive visualization can be combined for exploratory HR analytics. It provides an end-to-end workflow for data processing, model development, validation, and dashboard-based analysis.