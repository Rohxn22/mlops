import pytest
import pandas as pd
from prediction_model.config import config


def test_config_features():
    """Test that all required features are defined in config"""
    expected_features = [
        'age', 'occupation_status', 'years_employed', 'annual_income',
        'credit_score', 'credit_history_years', 'savings_assets', 'current_debt',
        'defaults_on_file', 'delinquencies_last_2yrs', 'derogatory_marks',
        'product_type', 'loan_intent', 'loan_amount', 'debt_to_income_ratio'
    ]
    assert set(config.FEATURES) == set(expected_features)


def test_config_target():
    """Test that target is correctly defined"""
    assert config.TARGET == 'loan_status'


def test_config_categorical_features():
    """Test that categorical features are correctly defined"""
    expected_cat_features = ['occupation_status', 'product_type', 'loan_intent']
    assert set(config.FEATURES_TO_ENCODE) == set(expected_cat_features)


def test_sample_data_structure():
    """Test that sample data has correct structure"""
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
    
    # Check all required features are present
    missing_features = set(config.FEATURES) - set(sample_data.columns)
    assert len(missing_features) == 0, f"Missing features: {missing_features}"
    
    # Check data types
    assert sample_data['age'].dtype in ['int64', 'int32']
    assert sample_data['occupation_status'].dtype == 'object'
    assert sample_data['credit_score'].dtype in ['int64', 'int32']