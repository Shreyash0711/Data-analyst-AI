
from ollama import Client
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def clarify_prompt(goal, df):
    if "predict" not in goal.lower():
        return goal
    
    try:
        client = Client(host='http://localhost:11434', timeout=120)
        columns = df.columns.tolist()
        dtypes = df.dtypes.to_dict()
        
        response = client.chat(
            model='llama3',
            messages=[
                {
                    'role': 'system',
                    'content': f"""
                    You are an AutoML assistant. The user provided goal: '{goal}'.
                    Given dataset columns: {columns} and types: {dtypes},
                    suggest a specific target variable and clarify the goal.
                    Return a string with the clarified goal.
                    Example: "Predict 'diagnosis' using other features"
                    If 'diagnosis' is in columns, prioritize it for classification.
                    If unsure, select a categorical column for classification or numeric for regression.
                    Keep the response concise.
                    """
                }
            ]
        )
        clarified_goal = response['message']['content']
        logging.info(f"Clarified goal: {clarified_goal}")
        return clarified_goal
    except Exception as e:
        logging.error(f"Error in prompt clarification: {e}")
        return "Predict 'diagnosis' using other features" if 'diagnosis' in df.columns else f"Predict '{df.columns[-1]}' using other features"
