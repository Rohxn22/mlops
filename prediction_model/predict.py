import pandas as pd
import numpy as np
import joblib
import os
from prediction_model.config import config
import mlflow


def _load_best_model():
    """Fetches the best model from MLflow based on highest F1 score."""
    mlflow.set_tracking_uri(config.TRACKING_URI)
    
    # Try to get the specific experiment first
    experiment = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME)
    
    if experiment is None:
        print(f"Warning: Experiment '{config.EXPERIMENT_NAME}' not found. Checking default experiment...")
        # Fallback: search in default experiment (ID=0) where "Training runs" are
        experiment_ids = ["0"]
    else:
        experiment_ids = [experiment.experiment_id]
    
    # Search for runs with the model
    runs_df = mlflow.search_runs(
        experiment_ids=experiment_ids,
        order_by=['metrics.f1_score DESC'],
        max_results=50
    )
    
    if runs_df.empty:
        # Final fallback: try to load local model file
        local_model_path = os.path.join(config.PACKAGE_ROOT, 'trained_models', 'loan_model_v2.pkl')
        if os.path.exists(local_model_path):
            print(f"Warning: No MLflow runs found. Loading local model: {local_model_path}")
            return joblib.load(local_model_path)
        else:
            raise ValueError(f"No runs found in any experiment and no local model at {local_model_path}")
    
    # Prioritize models tagged as "BEST_MODEL"
    best_tagged_runs = runs_df[runs_df['tags.model_status'] == 'BEST_MODEL']
    if not best_tagged_runs.empty:
        print(f"🏆 Found {len(best_tagged_runs)} model(s) tagged as BEST_MODEL")
        runs_to_try = best_tagged_runs
    else:
        print("ℹ️ No models tagged as BEST_MODEL, using highest F1 score")
        runs_to_try = runs_df
    
    # Find the best run with the model
    for _, run in runs_to_try.iterrows():
        try:
            run_id = run['run_id']
            model_uri = f'runs:/{run_id}/{config.MODEL_NAME}'
            model = mlflow.sklearn.load_model(model_uri)
            print(f"Successfully loaded model from run: {run_id}")
            return model
        except Exception as e:
            print(f"Failed to load model from run {run_id}: {e}")
            continue
    
    # If all MLflow attempts fail, try local model
    local_model_path = os.path.join(config.PACKAGE_ROOT, 'trained_models', 'loan_model_v2.pkl')
    if os.path.exists(local_model_path):
        print(f"Warning: All MLflow models failed. Loading local model: {local_model_path}")
        return joblib.load(local_model_path)
    
    raise ValueError("Could not load any model from MLflow or local storage")


def generate_predictions(data_input):
    data = pd.DataFrame(data_input)
    model = _load_best_model()
    prediction = model.predict(data)
    output = np.where(prediction == 1, 'Y', 'N')
    return {"prediction": output}


def generate_predictions_batch(data_input):
    model = _load_best_model()
    prediction = model.predict(data_input)
    output = np.where(prediction == 1, 'Y', 'N')
    return {"prediction": output}