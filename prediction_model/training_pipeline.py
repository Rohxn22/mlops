import numpy as np
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RandomizedSearchCV
import mlflow
import mlflow.sklearn
import xgboost as xgb

from prediction_model.config import config
from prediction_model.processing.data_handling import load_and_split_dataset


mlflow.set_tracking_uri(config.TRACKING_URI)

# ── Custom transformer: label-encode categoricals ──────────────────────────
class CatEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        self.encoders_ = {}
        for col in self.variables:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders_[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.variables:
            X[col] = self.encoders_[col].transform(X[col].astype(str))
        return X


# ── Feature Engineering ────────────────────────────────────────────────────
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['loan_to_income']    = X['loan_amount']    / (X['annual_income'] + 1)
        X['debt_burden']       = X['current_debt']   / (X['annual_income'] + 1)
        X['net_worth']         = X['savings_assets'] - X['current_debt']
        X['credit_risk_index'] = (X['defaults_on_file'] +
                                  X['delinquencies_last_2yrs'] +
                                  X['derogatory_marks'])
        X['savings_to_debt']   = X['savings_assets'] / (X['current_debt'] + 1)
        return X


# ── Load & split data ──────────────────────────────────────────────────────
print("Loading and splitting dataset (75/25)...")
X_train, X_test, y_train = load_and_split_dataset()
print(f"  Train: {X_train.shape} | Test (no target): {X_test.shape}")


# ── Shared preprocessing steps ─────────────────────────────────────────────
def build_preprocessing():
    return [
        ('FeatureEngineer', FeatureEngineer()),
        ('CatEncoder',      CatEncoder(variables=config.FEATURES_TO_ENCODE)),
        ('Scaler',          MinMaxScaler()),
    ]


# ── Baseline: Logistic Regression ─────────────────────────────────────────
def train_baseline():
    pipeline = Pipeline(build_preprocessing() + [
        ('LogisticRegression', LogisticRegression(max_iter=1000))
    ])

    mlflow.set_experiment(config.EXPERIMENT_NAME)
    run_tag = datetime.now().strftime("%m%d-%H%M")

    with mlflow.start_run(run_name=f"baseline-lr-{run_tag}"):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_train)   # train-set sanity check

        metrics = {
            'f1_score':  f1_score(y_train, y_pred),
            'accuracy':  accuracy_score(y_train, y_pred),
            'recall':    recall_score(y_train, y_pred),
            'precision': precision_score(y_train, y_pred),
        }
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, config.MODEL_NAME)

        print(f"Baseline LR → F1: {metrics['f1_score']:.4f} | Acc: {metrics['accuracy']:.4f}")


# ── XGBoost with RandomizedSearchCV ───────────────────────────────────────
param_dist = {
    'XGBoost__max_depth':        np.arange(3, 10, dtype=int),
    'XGBoost__learning_rate':    [0.01, 0.05, 0.1, 0.2, 0.3],
    'XGBoost__n_estimators':     np.arange(50, 300, 50, dtype=int),
    'XGBoost__subsample':        [0.5, 0.7, 0.8, 1.0],
    'XGBoost__colsample_bytree': [0.5, 0.7, 0.8, 1.0],
    'XGBoost__gamma':            [0, 0.5, 1, 2, 5],
    'XGBoost__reg_alpha':        [0, 0.1, 0.5, 1],
    'XGBoost__reg_lambda':       [0, 0.1, 0.5, 1],
}


def train_xgboost():
    clf = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    pipeline = Pipeline(build_preprocessing() + [('XGBoost', clf)])

    search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist,
        n_iter=5, scoring='f1', cv=3, random_state=42, n_jobs=-1
    )

    mlflow.set_experiment(config.EXPERIMENT_NAME)
    run_tag = datetime.now().strftime("%m%d-%H%M")

    with mlflow.start_run(run_name=f"xgb-randomsearch-{run_tag}"):
        search.fit(X_train, y_train)
        best = search.best_estimator_
        y_pred = best.predict(X_train)

        metrics = {
            'f1_score':  f1_score(y_train, y_pred),
            'accuracy':  accuracy_score(y_train, y_pred),
            'recall':    recall_score(y_train, y_pred),
            'precision': precision_score(y_train, y_pred),
        }
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best, config.MODEL_NAME)

        print(f"XGBoost best → F1: {metrics['f1_score']:.4f} | Acc: {metrics['accuracy']:.4f}")
        print("Best params:", search.best_params_)


# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nTraining baseline (Logistic Regression)...")
    train_baseline()

    print("\nRunning XGBoost RandomizedSearchCV (5 iterations)...")
    train_xgboost()