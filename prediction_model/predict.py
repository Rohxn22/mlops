import pandas as pd
import numpy as np
import joblib
import os
from prediction_model.config import config

MODEL_PATH = os.path.join(config.PACKAGE_ROOT, 'trained_models', 'model.pkl')


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training_pipeline.py first.")
    return joblib.load(MODEL_PATH)


def generate_predictions(data_input):
    data = pd.DataFrame(data_input)
    model = load_model()
    prediction = model.predict(data)
    return {"prediction": np.where(prediction == 1, 'Y', 'N')}


def generate_predictions_batch(data_input):
    model = load_model()
    prediction = model.predict(data_input)
    return {"prediction": np.where(prediction == 1, 'Y', 'N')}
