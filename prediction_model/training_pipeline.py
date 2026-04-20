import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import xgboost as xgb
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset
import prediction_model.processing.preprocessing as pp
import prediction_model.pipeline as pipe
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


mlflow.set_tracking_uri(config.TRACKING_URI)

def get_data(input):
    data = load_dataset(input)
    x = data[config.FEATURES]
    y = data[config.TARGET].map({'N': 0, 'Y': 1})
    return x, y


X, Y = get_data(config.TRAIN_FILE)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# BASELINE MODEL — Logistic Regression
def train_baseline():
    baseline_pipeline = Pipeline([
        ('DomainProcessing', pp.DomainProcessing(variable_to_modify=config.FEATURE_TO_MODIFY, variable_to_add=config.FEATURE_TO_ADD)),
        ('MeanImputation', pp.MeanImputer(variables=config.NUM_FEATURES)),
        ('ModeImputation', pp.ModeImputer(variables=config.CAT_FEATURES)),
        ('DropFeatures', pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
        ('LabelEncoder', pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
        ('LogTransform', pp.LogTransforms(variables=config.LOG_FEATURES)),
        ('MinMaxScale', MinMaxScaler()),
        ('LogisticRegression', LogisticRegression(max_iter=1000))
    ])

    mlflow.set_experiment("loan_prediction_model")
    run_tag = datetime.now().strftime("%m%d-%H%M")
    with mlflow.start_run(run_name=f"baseline-lr-{run_tag}"):
        baseline_pipeline.fit(X_train, y_train)
        y_pred = baseline_pipeline.predict(X_test)

        f1       = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        recall   = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metrics({
            'f1_score': f1,
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision
        })
        mlflow.sklearn.log_model(baseline_pipeline, "Loanprediction-model")

        print(f"Baseline (Logistic Regression) → F1: {f1:.4f} | Accuracy: {accuracy:.4f}")



# OPTIMIZED MODEL — XGBoost with Hyperopt

# Define the search space
search_space = {
    'max_depth': hp.choice('max_depth', np.arange(3, 10, dtype=int)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'n_estimators': hp.choice('n_estimators', np.arange(50, 300, 50, dtype=int)),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
    'gamma': hp.uniform('gamma', 0, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1)
}


def objective(params):
    clf = xgb.XGBClassifier(
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        gamma=params['gamma'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        use_label_encoder=False,
        eval_metric='mlogloss'
    )

    classification_pipeline = Pipeline([
        ('DomainProcessing', pp.DomainProcessing(variable_to_modify=config.FEATURE_TO_MODIFY, variable_to_add=config.FEATURE_TO_ADD)),
        ('MeanImputation', pp.MeanImputer(variables=config.NUM_FEATURES)),
        ('ModeImputation', pp.ModeImputer(variables=config.CAT_FEATURES)),
        ('DropFeatures', pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
        ('LabelEncoder', pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
        ('LogTransform', pp.LogTransforms(variables=config.LOG_FEATURES)),
        ('MinMaxScale', MinMaxScaler()),
        ('XGBoostClassifier', clf)
    ])

    mlflow.xgboost.autolog()
    mlflow.set_experiment("loan_prediction_model")
    run_tag = datetime.now().strftime("%m%d-%H%M")
    run_name = f"xgboost-trial-{len(trials.trials) + 1}-{run_tag}"

    with mlflow.start_run(nested=True, run_name=run_name):
        classification_pipeline.fit(X_train, y_train)
        y_pred = classification_pipeline.predict(X_test)

        f1        = f1_score(y_test, y_pred)
        accuracy  = accuracy_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        mlflow.log_metrics({
            'f1_score': f1,
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision
        })
        mlflow.sklearn.log_model(classification_pipeline, "Loanprediction-model")

    return {'loss': 1 - f1, 'status': STATUS_OK}


# ─────────────────────────────────────────────
# RUN TRAINING
# ─────────────────────────────────────────────

# 1. Train baseline first
print("Training baseline model (Logistic Regression)...")
train_baseline()

# 2. Run XGBoost hyperparameter tuning
print("\nRunning XGBoost hyperparameter tuning (5 trials)...")
trials = Trials()
best_params = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=5, trials=trials)

print("\nBest XGBoost hyperparameters:", best_params)
