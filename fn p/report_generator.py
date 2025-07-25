
import pandas as pd
import os
import json

def generate_report(df, eda_results, metrics, task_type, target_col):
    report = f"""
# AutoML Report

## Dataset Overview
- Rows: {df.shape[0]}
- Columns: {df.shape[1]}
- Target Variable: {target_col}

## Data Info
{eda_results['info']}

## Descriptive Statistics
### Numeric Columns
{json.dumps(eda_results['describe']['numeric'], indent=2)}

### Categorical Columns
{json.dumps(eda_results['describe']['categorical'], indent=2)}

## Groupby Analysis ({target_col})
{json.dumps(eda_results['groupby'], indent=2)}

## Feature Importance
{json.dumps(eda_results['feature_importance'], indent=2)}

## Data Cleaning
- Handled missing values, duplicates, and unnecessary columns
- Final shape: {df.shape}

## Exploratory Data Analysis
- Generated {len(eda_results['figures'])} figures (see UI)
- Visualizations: Countplot, scatter plot, boxplot, correlation heatmap, pairplot, feature importance

## Model Performance
- Task Type: {task_type.capitalize()}
- Metrics:
"""
    for key, value in metrics.items():
        report += f"  - {key}: {value}\n"
    
    report_path = "reports/output.md"
    with open(report_path, "w") as f:
        f.write(report)
    
    return report_path