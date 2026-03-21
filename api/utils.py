import pandas as pd
from datetime import datetime
import os

LOG_PATH = "logs/predictions_append.csv"

def log_prediction(data: dict, prediction: float):

    df = pd.DataFrame([data])
    df["prediction"] = prediction
    df["timestamp"] = datetime.now()

    file_exists = os.path.isfile(LOG_PATH)

    df.to_csv(
        LOG_PATH,
        mode="a",
        header=not file_exists,
        index=False
    )
