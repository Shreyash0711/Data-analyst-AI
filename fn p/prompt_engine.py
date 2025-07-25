
import json
import re
from ollama import Client
import logging

logging.basicConfig(level=logging.INFO)

def interpret_prompt(prompt):
    try:
        client = Client(host='http://localhost:11434', timeout=120)
        response = client.chat(
            model='llama3',
            messages=[
                {
                    'role': 'system',
                    'content': """
                    You are an AutoML assistant. Interpret the user's prompt to determine the analysis goal and task type (classification or regression).
                    Return a valid JSON object with 'goal' and 'task_type' keys.
                    Example: {"goal": "Predict rating", "task_type": "classification"}
                    If the prompt mentions 'rating' or categorical prediction, use 'classification'.
                    If unclear, default to classification.
                    Ensure the response is valid JSON without extra narrative text or backticks.
                    """
                },
                {'role': 'user', 'content': prompt}
            ]
        )
        content = response['message']['content']
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
        else:
            result = json.loads(content)
        logging.info(f"Prompt interpreted: {result}")
        return result['goal'], result['task_type']
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {e}. Response: {content}")
        return "Predict rating", "classification"
    except Exception as e:
        logging.error(f"Error in prompt interpretation: {e}")
        return "Predict rating", "classification"
