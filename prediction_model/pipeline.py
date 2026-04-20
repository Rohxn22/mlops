"""
Inference pipeline for the new loan approval model
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from prediction_model.config import config
import prediction_model.processing.preprocessing as pp

# Preprocessing pipeline used for inference
preprocessing_pipeline = Pipeline([
    ('FeatureEngineer',     pp.FeatureEngineer()),
    ('CategoricalEncoder',  pp.CategoricalEncoder(variables=config.FEATURES_TO_ENCODE)),
    ('Scaler',              MinMaxScaler())
])