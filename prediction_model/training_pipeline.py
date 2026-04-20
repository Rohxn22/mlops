import numpy as np
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
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


# ── XGBoost with Hyperopt ──────────────────────────────────────────────────

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

trials = Trials()


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
        eval_metric='logloss'
    )

    classification_pipeline = Pipeline(build_preprocessing() + [('XGBoostClassifier', clf)])

    mlflow.xgboost.autolog()
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    run_tag = datetime.now().strftime("%m%d-%H%M")
    run_name = f"xgboost-trial-{len(trials.trials) + 1}-{run_tag}"

    with mlflow.start_run(nested=True, run_name=run_name):
        classification_pipeline.fit(X_train, y_train)
        y_pred = classification_pipeline.predict(X_train)  # train set for optimization

        f1        = f1_score(y_train, y_pred)
        accuracy  = accuracy_score(y_train, y_pred)
        recall    = recall_score(y_train, y_pred)
        precision = precision_score(y_train, y_pred)

        mlflow.log_metrics({
            'f1_score': f1,
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision
        })
        mlflow.sklearn.log_model(classification_pipeline, config.MODEL_NAME)

    return {'loss': 1 - f1, 'status': STATUS_OK}


# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Training baseline model (Logistic Regression)...")
    train_baseline()

    print("\nRunning XGBoost hyperparameter tuning (5 trials)...")
    best_params = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=5, trials=trials)
    print("\nBest XGBoost hyperparameters:", best_params)