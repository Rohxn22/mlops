"""
Training Pipeline V2 — Loan Approval Model
- Engineered features baked in
- Regularized XGBoost (anti-overfit)
- Clean train/test comparison printed locally
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                              recall_score, roc_auc_score, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb

from prediction_model.config import config
from prediction_model.processing.data_handling import load_and_split_dataset


# ── Feature Engineering Transformer ───────────────────────────────────────
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


# ── Categorical Encoder ────────────────────────────────────────────────────
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


# ── Load & split ───────────────────────────────────────────────────────────
print("Loading and splitting dataset (70/30, stratified)...")
X_train, X_test, y_train = load_and_split_dataset()
print(f"  Train: {X_train.shape} | Test (no target): {X_test.shape}\n")


# ── Shared preprocessing ───────────────────────────────────────────────────
def base_steps():
    return [
        ('FeatureEngineer', FeatureEngineer()),
        ('CatEncoder',      CatEncoder(variables=config.FEATURES_TO_ENCODE)),
        ('Scaler',          MinMaxScaler()),
    ]


# ── Print helper ───────────────────────────────────────────────────────────
def print_results(name, y_true, y_pred, y_prob):
    print(f"  {'─'*50}")
    print(f"  {name}")
    print(f"  {'─'*50}")
    print(f"  Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  F1 Score : {f1_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"  Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_true, y_prob):.4f}")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    print()


# ── Baseline: Logistic Regression ─────────────────────────────────────────
def train_baseline():
    pipeline = Pipeline(base_steps() + [
        ('model', LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X_train, y_train)

    # Train metrics
    tr_pred = pipeline.predict(X_train)
    tr_prob = pipeline.predict_proba(X_train)[:, 1]
    print_results("Logistic Regression — TRAIN", y_train, tr_pred, tr_prob)

    return pipeline


# ── Regularized XGBoost ────────────────────────────────────────────────────
def train_xgboost():
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=5,
        eval_metric='logloss',
        verbosity=0
    )
    pipeline = Pipeline(base_steps() + [('model', clf)])
    pipeline.fit(X_train, y_train)

    # Train metrics
    tr_pred = pipeline.predict(X_train)
    tr_prob = pipeline.predict_proba(X_train)[:, 1]
    print_results("XGBoost (Regularized) — TRAIN", y_train, tr_pred, tr_prob)

    # Feature importances
    feat_names = X_train.columns.tolist() + [
        'loan_to_income', 'debt_burden', 'net_worth',
        'credit_risk_index', 'savings_to_debt'
    ]
    importances = clf.feature_importances_
    top = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)[:10]
    print("  Top 10 Feature Importances:")
    for fname, imp in top:
        print(f"    {fname:<30}: {imp:.4f}")
    print()

    return pipeline


# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 52)
    print("  BASELINE — LOGISTIC REGRESSION")
    print("=" * 52)
    lr_pipe = train_baseline()

    print("=" * 52)
    print("  OPTIMIZED — XGBOOST (REGULARIZED + FEAT ENG)")
    print("=" * 52)
    xgb_pipe = train_xgboost()

    print("=" * 52)
    print("  NOTE: Test set target was withheld (30% split)")
    print("  Train metrics shown above.")
    print("  To evaluate on test, re-run with y_test retained.")
    print("=" * 52)
