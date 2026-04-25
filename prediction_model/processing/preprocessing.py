from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates derived financial features from raw loan data."""

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


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Label-encodes categorical columns."""

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