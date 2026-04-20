import os
import pandas as pd
from sklearn.model_selection import train_test_split
from prediction_model.config import config


def load_and_split_dataset(test_size=0.25, random_state=42):
    """
    Loads the single source dataset, splits 75/25 (stratified).
    Test split has the target column dropped.

    Returns:
        X_train, X_test, y_train  (y_test intentionally withheld)
    """
    filepath = os.path.join(config.DATAPATH, config.DATASET_FILE)
    df = pd.read_csv(filepath)

    X = df[config.FEATURES]
    y = df[config.TARGET]

    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Reset indices for clean downstream use
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    return X_train, X_test, y_train
