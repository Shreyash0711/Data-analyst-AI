
import streamlit as st
import pandas as pd
import os
from data_handler import load_data
from cleaning import clean_data
from preprocessing import preprocess_data
from eda import perform_eda
from modeling import train_model
from evaluation import evaluate_model
from report_generator import generate_report
import json

def main():
    st.title("LLM-Powered AutoML Assistant")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Save uploaded file
        with open("data/input.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load data
        df = load_data("data/input.csv")
        if df is None:
            st.error("Failed to load CSV. Please check the file format.")
            return
        
        # Select target variable
        st.subheader("Select Target Variable")
        target_col = st.selectbox("Choose the target variable for prediction:", df.columns.tolist(), index=df.columns.tolist().index('total_bill') if 'total_bill' in df.columns else 0)
        
        # Determine task type based on target dtype
        task_type = 'regression' if df[target_col].dtype in ['float64', 'int64'] else 'classification'
        st.write(f"Goal: Predict '{target_col}' ({task_type})")
        
        # Data Cleaning
        cleaned_df = clean_data(df, target_col=target_col)
        
        # EDA
        st.subheader("Exploratory Data Analysis")
        eda_results = perform_eda(cleaned_df, target_col, task_type)
        st.text("Dataset Info:")
        st.text(eda_results['info'])
        st.text("Numeric Columns Description:")
        st.json(eda_results['describe']['numeric'])
        st.text("Categorical Columns Description:")
        st.json(eda_results['describe']['categorical'])
        st.text(f"Groupby {target_col}:")
        st.json(eda_results['groupby'])
        st.text("Feature Importance:")
        st.json(eda_results['feature_importance'])
        for fig in eda_results['figures']:
            st.pyplot(fig)
        
        # Preprocessing
        preprocessed_df, target_col = preprocess_data(cleaned_df, task_type, f"Predict '{target_col}'")
        
        # Modeling and Training
        model, X_test, y_test = train_model(preprocessed_df, target_col, task_type)
        
        # Evaluation
        metrics = evaluate_model(model, X_test, y_test, task_type)
        st.subheader("Model Evaluation")
        st.write(metrics)
        
        # Generate Report
        report_path = generate_report(cleaned_df, eda_results, metrics, task_type, target_col)
        with open(report_path, "r") as f:
            st.markdown(f.read())

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("reports"):
        os.makedirs("reports")
    main()
