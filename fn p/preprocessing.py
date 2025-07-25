
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import logging
import re

logging.basicConfig(level=logging.INFO)

def preprocess_data(df, task_type, clarified_goal=None):
    # Identify target column
    target_col = None
    if clarified_goal and "Predict '" in clarified_goal:
        match = re.search(r"Predict '([^']+)'", clarified_goal)
        if match:
            target_col = match.group(1)
            if target_col not in df.columns:
                target_col = None
                logging.warning(f"Target column '{target_col}' from clarified goal not found in DataFrame")

    if target_col is None:
        for col in df.columns:
            if task_type == "regression" and df[col].dtype in ['float64', 'int64']:
                if not any(keyword in col.lower() for keyword in ['id', 'number', 'cc']):
                    target_col = col
                    break
            elif task_type == "classification" and df[col].dtype.name in ['object', 'category']:
                target_col = col
                break

    if target_col is None:
        logging.error("No suitable target column found")
        raise ValueError("No suitable target column found")

    # Encode categorical variables
    for col in df.columns:
        if col != target_col and df[col].dtype.name in ['object', 'category']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Scale numeric features
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col != target_col and not any(keyword in col.lower() for keyword in ['id', 'number', 'cc'])]
    if numeric_cols:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    logging.info(f"Preprocessed data with target: {target_col}")
    return df, target_col
