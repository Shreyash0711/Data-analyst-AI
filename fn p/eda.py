
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from io import StringIO

logging.basicConfig(level=logging.INFO)

def perform_eda(df, target_col, task_type):
    results = {'figures': [], 'info': '', 'describe': {}, 'groupby': {}, 'feature_importance': {}}
    
    # 1. DataFrame Info
    info_buffer = StringIO()
    df.info(buf=info_buffer)
    results['info'] = info_buffer.getvalue()
    logging.info("DataFrame info captured")
    
    # 2. Describe (numeric and categorical)
    results['describe']['numeric'] = df.select_dtypes(include=['float64', 'int64']).describe().to_dict()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    results['describe']['categorical'] = df[categorical_cols].describe().to_dict() if len(categorical_cols) > 0 else {}
    logging.info("Descriptive statistics captured")
    
    # 3. Groupby target (for classification)
    if task_type == "classification" and target_col in df.columns and df[target_col].dtype.name in ['object', 'category']:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        results['groupby'] = df.groupby(target_col)[numeric_cols].mean().to_dict() if len(numeric_cols) > 0 else {}
        logging.info(f"Groupby {target_col} completed")
    
    # 4. Visualizations
    # Histogram for regression target
    if task_type == "regression" and target_col in df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[target_col], bins=30)
        plt.title(f"Distribution of {target_col}")
        plt.xlabel(target_col)
        plt.ylabel("Count")
        results['figures'].append(plt.gcf())
    
    # Scatter plot for numeric features vs target
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col != target_col]
    if len(numeric_cols) >= 1 and target_col in df.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=numeric_cols[0], y=target_col, data=df)
        plt.title(f"{numeric_cols[0]} vs {target_col}")
        results['figures'].append(plt.gcf())
    
    # Boxplot for categorical feature vs target
    categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col != target_col]
    if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=categorical_cols[0], y=target_col, data=df)
        plt.title(f"{target_col} by {categorical_cols[0]}")
        plt.xticks(rotation=45)
        results['figures'].append(plt.gcf())
    
    # Feature importance (prominent plot)
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        if len(X.columns) > 0:
            # Encode categorical variables for feature importance
            X_encoded = X.copy()
            for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
                X_encoded[col] = X_encoded[col].astype('category').cat.codes
            
            model = RandomForestRegressor(n_estimators=50, random_state=42) if task_type == "regression" else RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_encoded, y)
            importance = pd.Series(model.feature_importances_, index=X_encoded.columns).sort_values(ascending=False)
            results['feature_importance'] = importance.to_dict()
            
            # Enhanced feature importance plot
            plt.figure(figsize=(10, 6))
            sns.barplot(x=importance.values, y=importance.index, palette='viridis')
            plt.title(f"Feature Importance for Predicting {target_col}")
            plt.xlabel("Importance Score")
            plt.ylabel("Features")
            plt.tight_layout()
            results['figures'].append(plt.gcf())
    
    logging.info("EDA completed with %d figures", len(results['figures']))
    return results
