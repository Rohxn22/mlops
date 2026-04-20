import pandas as pd
import numpy as np
import joblib
import os
from prediction_model.config import config
import mlflow


def _load_best_model():
    """Fetches the best model - tries local first, then MLflow as fallback."""
    
    # QUICK FIX: Try local model first (most reliable)
    local_model_paths = [
        os.path.join(config.PACKAGE_ROOT, 'trained_models', 'loan_model_v2.pkl'),
        os.path.join(config.PACKAGE_ROOT, 'trained_models', 'production_model.pkl')
    ]
    
    for local_path in local_model_paths:
        if os.path.exists(local_path):
            try:
                print(f"✅ Loading local model: {local_path}")
                return joblib.load(local_path)
            except Exception as e:
                print(f"❌ Failed to load local model {local_path}: {e}")
                continue
    
    print("🔍 No local model found, trying MLflow...")
    
    # FALLBACK: Try MLflow (can be unreliable)
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
    
    print(f"🔍 Debug: Found {len(runs_df)} runs in experiment")
    
    if runs_df.empty:
        raise ValueError(f"No runs found in any experiment and no local model available")
    
    # Debug: Print run info
    for i, (_, run) in enumerate(runs_df.head(3).iterrows()):
        print(f"🔍 Run {i+1}: {run['run_id'][:8]} - F1: {run.get('metrics.f1_score', 'N/A')} - Status: {run.get('status', 'N/A')}")
    
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
            print(f"🔍 Trying to load from run: {run_id[:8]}")
            
            # Try the artifact path we actually use in training
            artifact_paths = ["trained_model"]  # Only try the path that exists
            
            for artifact_path in artifact_paths:
                try:
                    model_uri = f'runs:/{run_id}/{artifact_path}'
                    print(f"   Trying artifact path: {artifact_path}")
                    model = mlflow.sklearn.load_model(model_uri)
                    print(f"✅ Successfully loaded model from run: {run_id[:8]} (artifact: {artifact_path})")
                    return model
                except Exception as e:
                    print(f"   ❌ Failed to load from {artifact_path}: {str(e)[:100]}")
                    continue
                    
        except Exception as e:
            print(f"❌ Failed to load model from run {run_id[:8]}: {e}")
            continue
    
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