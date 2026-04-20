import pandas as pd
import numpy as np
from prediction_model.config import config
import mlflow


def _load_best_model():
    """Fetches the best model from MLflow based on highest F1 score."""
    experiment = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME)
    runs_df = mlflow.search_runs(
        experiment_ids=experiment.experiment_id,
        order_by=['metrics.f1_score DESC']
    )
    best_run_id = runs_df.iloc[0]['run_id']
    return mlflow.sklearn.load_model('runs:/' + best_run_id + config.MODEL_NAME)


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