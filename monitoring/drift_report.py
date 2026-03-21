import pandas as pd

reference_df = pd.read_csv("data/processed/X_train.csv")

y_train = pd.read_csv("data/processed/y_train.csv")
reference_df["target"] = y_train

current_df = pd.read_csv("logs/predictions.csv")
current_df["target"]=current_df["actual_default"]
current_df = current_df.drop(
    columns=["prediction", "timestamp","actual_default"],
    errors="ignore"
)


from evidently import Report
from evidently.presets import DataDriftPreset


report = Report(metrics=[DataDriftPreset()])

myeval=report.run(
    reference_data=reference_df,
    current_data=current_df
)
myeval.save_html("monitoring/drift_report.html")

drift_dict = myeval.dict()

drift_share = drift_dict["metrics"][0]["value"]["share"]

dataset_drift = drift_share > 0.5


drifted_features = drift_dict["metrics"][0]["value"]["count"]

total_features = len(drift_dict["metrics"]) - 1

drift_pct = (drifted_features / total_features) * 100

import json
from datetime import datetime

metrics_path = "monitoring/monitoring_metrics.json"

with open(metrics_path) as f:
    metrics = json.load(f)

metrics["data_drift"] = dataset_drift
metrics["drifted_features_pct"] = drift_pct
metrics["last_updated"] = str(datetime.now())

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)
