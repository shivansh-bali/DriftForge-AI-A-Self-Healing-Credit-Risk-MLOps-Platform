import pandas as pd
from datetime import datetime
import os

LOG_PATH = "retraining/retraining_log.csv"


def log_retraining_event(old_auc, new_auc):
   

    log = pd.DataFrame([{
        "timestamp": datetime.now(),
        "old_auc": old_auc,
        "new_auc": new_auc,
        "retrained": new_auc > old_auc
    }])

    file_exists = os.path.isfile(LOG_PATH)

    log.to_csv(
        LOG_PATH,
        mode="a",
        header=not file_exists,
        index=False
    )
