import os

current_directory = os.path.dirname(os.path.realpath(__file__))
PACKAGE_ROOT = os.path.dirname(current_directory)

DATAPATH = os.path.join(PACKAGE_ROOT, "datasets")

TRAIN_FILE = 'train.csv'
TEST_FILE  = 'test.csv'

TARGET = 'Loan_Status'

FEATURES = ['Gender', 'Married', 'Dependents', 'Education',
            'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
            'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']

NUM_FEATURES = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

CAT_FEATURES = ['Gender', 'Married', 'Dependents', 'Education',
                'Self_Employed', 'Credit_History', 'Property_Area']

FEATURES_TO_ENCODE = ['Gender', 'Married', 'Dependents', 'Education',
                      'Self_Employed', 'Credit_History', 'Property_Area']

FEATURE_TO_MODIFY = ['ApplicantIncome']
FEATURE_TO_ADD    = 'CoapplicantIncome'
DROP_FEATURES     = ['CoapplicantIncome']
LOG_FEATURES      = ['ApplicantIncome', 'LoanAmount']

S3_BUCKET    = "mlops-dataset-792633646256-eu-north-1-an"
FOLDER       = "datadrift"

TRACKING_URI    = "http://ec2-13-63-102-32.eu-north-1.compute.amazonaws.com:5000/"
EXPERIMENT_NAME = "loan_prediction_model"
MODEL_NAME      = "/Loanprediction-model"
