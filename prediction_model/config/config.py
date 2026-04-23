import os

current_directory = os.path.dirname(os.path.realpath(__file__))
PACKAGE_ROOT = os.path.dirname(current_directory)

DATAPATH = os.path.join(PACKAGE_ROOT, "datasets")

# Single source dataset — will be split 75/25 in data_handling
# NOTE: Changing this will automatically update CI/CD and Docker build
DATASET_FILE = 'loan_data_part_2.csv'

TARGET = 'loan_status'

FEATURES = [
    'age', 'occupation_status', 'years_employed', 'annual_income',
    'credit_score', 'credit_history_years', 'savings_assets', 'current_debt',
    'defaults_on_file', 'delinquencies_last_2yrs', 'derogatory_marks',
    'product_type', 'loan_intent', 'loan_amount', 'debt_to_income_ratio'
]

NUM_FEATURES = [
    'years_employed', 'annual_income', 'credit_score', 'credit_history_years',
    'savings_assets', 'current_debt', 'loan_amount', 'debt_to_income_ratio',
    'age', 'defaults_on_file', 'delinquencies_last_2yrs', 'derogatory_marks'
]

CAT_FEATURES = ['occupation_status', 'product_type', 'loan_intent']

FEATURES_TO_ENCODE = ['occupation_status', 'product_type', 'loan_intent']

# MLflow — reuse same server, separate experiment
TRACKING_URI    = "http://ec2-13-63-102-32.eu-north-1.compute.amazonaws.com:5000/"
EXPERIMENT_NAME = "loan_prediction_v2"
MODEL_NAME      = "Loanprediction-model"

S3_BUCKET = "mlops-dataset-792633646256-eu-north-1-an"
FOLDER    = "datadrift_v2"
