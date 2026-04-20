import os
import pandas as pd
from prediction_model.config import config


def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH, file_name)
    return pd.read_csv(filepath)
