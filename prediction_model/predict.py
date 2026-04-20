"""
Prediction functions for the new loan approval model
"""

import pandas as pd
import numpy as np
import joblib
import os
from prediction_model.config import config
from prediction_model.pipeline import preprocessing_pipeline


def load_model():
    """Load the trained model from disk"""
    model_path = os.path.join(config.PACKAGE_ROOT, 'trained_models', 'loan_model_v2.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")


def generate_predictions(data_input):
    """
    Generate predictions for single or multiple samples
    
    Args:
        data_input: List of dictionaries or DataFrame with loan features
        
    Returns:
        Dictionary with predictions ('Y' for approved, 'N' for rejected)
    """
    # Convert to DataFrame if needed
    if isinstance(data_input, list):
        data = pd.DataFrame(data_input)
    else:
        data = data_input.copy()
    
    # Ensure all required features are present
    missing_features = set(config.FEATURES) - set(data.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Select only the features we need
    data = data[config.FEATURES]
    
    # Load model and make predictions
    model = load_model()
    predictions = model.predict(data)
    
    # Convert to Y/N format
    output = np.where(predictions == 1, 'Y', 'N')
    
    return {"prediction": output.tolist()}


def generate_predictions_batch(data_input):
    """Batch prediction function (same as generate_predictions)"""
    return generate_predictions(data_input)