
import pandas as pd
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

def clean_data(df, target_col=None):
    # Log initial shape and dtypes
    logging.info(f"Initial data shape: {df.shape}")
    logging.info(f"Initial dtypes:\n{df.dtypes}")
    
    # Drop unnecessary columns (e.g., 'id', 'number')
    drop_keywords = ['id', 'number', 'cc']
    drop_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in drop_keywords)]
    for col in drop_cols:
        logging.info(f"Dropping unnecessary column: {col}")
        df = df.drop(columns=col)
    
    # Correct dtypes
    for col in df.columns:
        if col == target_col:
            if df[col].dtype == 'object' and df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
                logging.info(f"Converted '{col}' to category")
        else:
            if df[col].dtype == 'object':
                if df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')
                    logging.info(f"Converted '{col}' to category")
    
    # Handle missing values
    for col in df.columns:
        missing_ratio = df[col].isnull().sum() / len(df)
        if missing_ratio > 0.8 and col != target_col:
            logging.warning(f"Dropped column '{col}' due to {missing_ratio*100:.1f}% missing values")
            df = df.drop(columns=col)
        else:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].mean())
            elif df[col].dtype.name == 'category':
                df[col] = df[col].cat.add_categories(['Unknown']).fillna('Unknown')
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Log final dtypes and shape
    logging.info(f"Final dtypes:\n{df.dtypes}")
    logging.info(f"Data shape after cleaning: {df.shape}")
    return df
