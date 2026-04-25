import numpy as np
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import xgboost as xgb
import mlflow
import mlflow.sklearn
import joblib
import os

from prediction_model.config import config
from prediction_model.processing.data_handling import load_and_split_dataset
from prediction_model.processing.preprocessing import FeatureEngineer, CategoricalEncoder

MODEL_PATH = os.path.join(config.PACKAGE_ROOT, 'trained_models', 'model.pkl')

mlflow.set_tracking_uri(config.TRACKING_URI)
mlflow.set_experiment(config.EXPERIMENT_NAME)

# Load data once
X_train, X_test, y_train, y_test = load_and_split_dataset()

# Shared preprocessing steps used by both models
def build_pipeline(model):
    return Pipeline([
        ('engineer', FeatureEngineer()),
        ('encoder',  CategoricalEncoder(variables=config.FEATURES_TO_ENCODE)),
        ('scaler',   MinMaxScaler()),
        ('model',    model),
    ])

def log_metrics(pipeline, run_name, model_type, params=None):
    with mlflow.start_run(run_name=run_name, nested=True):
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        metrics = {
            'accuracy':  accuracy_score(y_test, y_pred),
            'f1_score':  f1_score(y_test, y_pred),
            'recall':    recall_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
        }

        mlflow.log_param('model_type', model_type)
        mlflow.log_param('dataset_file', config.DATASET_FILE)
        if params:
            mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        # Log model artifact to MLflow for tracking/registry
        mlflow.sklearn.log_model(pipeline, 'model', registered_model_name=config.MODEL_NAME)

        print(f"{model_type} -> F1: {metrics['f1_score']:.4f} | Acc: {metrics['accuracy']:.4f}")
    return metrics['f1_score'], pipeline


# Baseline: Logistic Regression
def train_baseline():
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    mlflow.sklearn.autolog(disable=True)
    tag = datetime.now().strftime("%m%d-%H%M")
    f1, pipeline = log_metrics(pipeline, f"baseline-lr-{tag}", "LogisticRegression")
    return f1, pipeline


# XGBoost with Hyperopt tuning
search_space = {
    'max_depth':        hp.choice('max_depth', np.arange(3, 10, dtype=int)),
    'learning_rate':    hp.uniform('learning_rate', 0.01, 0.3),
    'n_estimators':     hp.choice('n_estimators', np.arange(50, 300, 50, dtype=int)),
    'subsample':        hp.uniform('subsample', 0.5, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
}

trials = Trials()

def objective(params):
    mlflow.xgboost.autolog(disable=True)
    clf = xgb.XGBClassifier(**params, eval_metric='logloss')
    pipeline = build_pipeline(clf)
    tag = datetime.now().strftime("%m%d-%H%M")
    f1, pipeline = log_metrics(pipeline, f"xgb-trial-{len(trials.trials)+1}-{tag}", "XGBoost", params)
    trials.trials[-1]['result']['pipeline'] = pipeline  # store for later
    return {'loss': 1 - f1, 'status': STATUS_OK}


def tag_best_model():
    experiment = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME)
    if not experiment:
        return
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=['metrics.f1_score DESC'],
        max_results=1
    )
    if runs.empty:
        return
    best_run_id = runs.iloc[0]['run_id']
    with mlflow.start_run(run_id=best_run_id):
        mlflow.set_tag('model_status', 'BEST_MODEL')
    print(f"Best model tagged: {best_run_id[:8]} (F1: {runs.iloc[0]['metrics.f1_score']:.4f})")


if __name__ == "__main__":
    print("Training baseline (Logistic Regression)...")
    baseline_f1, baseline_pipeline = train_baseline()

    print("\nRunning XGBoost hyperparameter tuning (3 trials)...")
    with mlflow.start_run(run_name="xgboost-hyperopt"):
        fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=3, trials=trials)

    # Pick best pipeline across baseline + xgboost trials
    best_xgb_f1    = min(t['result']['loss'] for t in trials.trials)
    best_xgb_pipe  = next(t['result']['pipeline'] for t in trials.trials
                          if t['result']['loss'] == best_xgb_f1)

    best_pipeline = baseline_pipeline if baseline_f1 >= (1 - best_xgb_f1) else best_xgb_pipe

    # Save best model as .pkl — used by runtime image (no MLflow needed at serving time)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\nBest model saved to {MODEL_PATH}")

    print("\nTagging best model in MLflow...")
    tag_best_model()

    print("\nTraining pipeline complete.")