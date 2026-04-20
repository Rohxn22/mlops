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
print(f"  Train: {X_train.shape} | Test: {X_test.shape}")

# For proper evaluation, we need y_test too
# Let's modify the data loading to get y_test
import os
import pandas as pd
from sklearn.model_selection import train_test_split

filepath = os.path.join(config.DATAPATH, config.DATASET_FILE)
df = pd.read_csv(filepath)
X = df[config.FEATURES]
y = df[config.TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
print(f"  y_train: {y_train.shape} | y_test: {y_test.shape}")


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

    # Ensure experiment exists
    try:
        mlflow.create_experiment(config.EXPERIMENT_NAME)
        print(f"Created experiment: {config.EXPERIMENT_NAME}")
    except Exception:
        print(f"Experiment {config.EXPERIMENT_NAME} already exists or couldn't be created")
    
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    run_tag = datetime.now().strftime("%m%d-%H%M")

    with mlflow.start_run(run_name=f"baseline-lr-{run_tag}"):
        pipeline.fit(X_train, y_train)
        
        # Evaluate on both train and test sets
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        # Train metrics
        train_metrics = {
            'train_f1_score':  f1_score(y_train, y_train_pred),
            'train_accuracy':  accuracy_score(y_train, y_train_pred),
            'train_recall':    recall_score(y_train, y_train_pred),
            'train_precision': precision_score(y_train, y_train_pred),
        }
        
        # Test metrics (the important ones!)
        test_metrics = {
            'f1_score':  f1_score(y_test, y_test_pred),
            'accuracy':  accuracy_score(y_test, y_test_pred),
            'recall':    recall_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
        }
        
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metrics({**train_metrics, **test_metrics})
        mlflow.sklearn.log_model(pipeline, config.MODEL_NAME, registered_model_name=config.MODEL_NAME)

        print(f"Baseline LR → Train F1: {train_metrics['train_f1_score']:.4f} | Test F1: {test_metrics['f1_score']:.4f}")
        print(f"             Train Acc: {train_metrics['train_accuracy']:.4f} | Test Acc: {test_metrics['accuracy']:.4f}")


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
        eval_metric='logloss'
    )

    classification_pipeline = Pipeline(build_preprocessing() + [('XGBoostClassifier', clf)])

    mlflow.xgboost.autolog()
    
    # Ensure experiment exists
    try:
        mlflow.create_experiment(config.EXPERIMENT_NAME)
    except Exception:
        pass  # Experiment already exists
    
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    run_tag = datetime.now().strftime("%m%d-%H%M")
    run_name = f"xgboost-trial-{len(trials.trials) + 1}-{run_tag}"

    with mlflow.start_run(nested=True, run_name=run_name):
        classification_pipeline.fit(X_train, y_train)
        
        # Evaluate on both train and test sets
        y_train_pred = classification_pipeline.predict(X_train)
        y_test_pred = classification_pipeline.predict(X_test)

        # Train metrics
        train_f1 = f1_score(y_train, y_train_pred)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred)
        
        # Test metrics (the important ones!)
        test_f1 = f1_score(y_test, y_test_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)

        mlflow.log_metrics({
            'train_f1_score': train_f1,
            'train_accuracy': train_accuracy,
            'train_recall': train_recall,
            'train_precision': train_precision,
            'f1_score': test_f1,
            'accuracy': test_accuracy,
            'recall': test_recall,
            'precision': test_precision
        })
        mlflow.sklearn.log_model(classification_pipeline, config.MODEL_NAME, registered_model_name=config.MODEL_NAME)

    # Use TEST F1 for optimization (not train F1!)
    return {'loss': 1 - test_f1, 'status': STATUS_OK}


# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Training baseline model (Logistic Regression)...")
    train_baseline()

    print("\nRunning XGBoost hyperparameter tuning (5 trials)...")
    best_params = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=5, trials=trials)
    print("\nBest XGBoost hyperparameters:", best_params)
    
    # ── Tag Best Model ─────────────────────────────────────────────────────────
    print("\n🏆 Identifying and tagging best model from ALL runs...")
    
    try:
        # Get all runs from this experiment (not just current training session)
        experiment = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME)
        if experiment:
            # Search ALL runs in the experiment, not just recent ones
            runs_df = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=['metrics.f1_score DESC'],
                max_results=1000  # Get all runs to find the absolute best
            )
            
            if not runs_df.empty:
                # Get the absolute best run across all time
                best_run = runs_df.iloc[0]
                best_run_id = best_run['run_id']
                best_f1 = best_run.get('metrics.f1_score', 0)
                best_model_type = best_run.get('params.model_type', 'Unknown')
                best_timestamp = best_run['start_time']
                
                # Clear all existing "BEST_MODEL" tags first
                print(f"🧹 Clearing existing BEST_MODEL tags from all runs...")
                for _, run in runs_df.iterrows():
                    try:
                        with mlflow.start_run(run_id=run['run_id']):
                            mlflow.set_tag("model_status", "")  # Clear the tag
                    except:
                        pass  # Ignore errors for old runs
                
                # Tag the absolute best model
                with mlflow.start_run(run_id=best_run_id):
                    mlflow.set_tag("model_status", "BEST_MODEL")
                    mlflow.set_tag("best_model_timestamp", datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
                    mlflow.set_tag("selection_criteria", "highest_f1_score_all_time")
                    mlflow.set_tag("selected_from_runs", len(runs_df))
                
                print(f"✅ Tagged ABSOLUTE best model from {len(runs_df)} total runs:")
                print(f"   Model: {best_model_type}")
                print(f"   F1 Score: {best_f1:.4f}")
                print(f"   Run ID: {best_run_id[:8]}")
                print(f"   Trained: {best_timestamp}")
                        
            else:
                print("❌ No runs found to tag")
        else:
            print(f"❌ Experiment '{config.EXPERIMENT_NAME}' not found")
            
    except Exception as e:
        print(f"⚠️ Error tagging best model: {e}")
        
    print("\n🎉 Training pipeline completed!")