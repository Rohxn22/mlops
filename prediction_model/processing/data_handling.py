import os
import pandas as pd
from sklearn.model_selection import train_test_split
from prediction_model.config import config


def load_and_split_dataset(test_size=0.25, random_state=42):
    filepath = os.path.join(config.DATAPATH, config.DATASET_FILE)
    df = pd.read_csv(filepath)

    X = df[config.FEATURES]
    y = df[config.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )