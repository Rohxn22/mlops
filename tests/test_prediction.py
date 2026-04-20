import pytest
import mlflow
import pandas as pd
from prediction_model.config import config
from prediction_model.predict import generate_predictions

mlflow.set_tracking_uri(config.TRACKING_URI)


@pytest.fixture
def single_prediction():
    # Create sample data with new features
    sample_data = pd.DataFrame([{
        'age': 35,
        'occupation_status': 'Employed',
        'years_employed': 8.5,
        'annual_income': 75000,
        'credit_score': 720,
        'credit_history_years': 12.0,
        'savings_assets': 25000,
        'current_debt': 15000,
        'defaults_on_file': 0,
        'delinquencies_last_2yrs': 1,
        'derogatory_marks': 0,
        'product_type': 'Personal',
        'loan_intent': 'debt_consolidation',
        'loan_amount': 50000,
        'debt_to_income_ratio': 0.35
    }])
    return generate_predictions(sample_data)


def test_single_pred_not_none(single_prediction):
    assert single_prediction is not None


def test_single_pred_str_type(single_prediction):
    assert isinstance(single_prediction.get('prediction')[0], str)


def test_single_pred_validate(single_prediction):
    assert single_prediction.get('prediction')[0] in ['Y', 'N']