from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import logging

logging.basicConfig(level=logging.INFO)

def evaluate_model(model, X_test, y_test, task_type):
    predictions = model.predict(X_test)
    metrics = {}
    
    if task_type == "regression":
        metrics['MSE'] = mean_squared_error(y_test, predictions)
        metrics['R2'] = r2_score(y_test, predictions)
    else:
        metrics['Accuracy'] = accuracy_score(y_test, predictions)
        metrics['Classification Report'] = classification_report(y_test, predictions, output_dict=True)
    
    logging.info(f"Evaluation metrics: {metrics}")
    return metrics