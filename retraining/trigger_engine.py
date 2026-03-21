import json
from retrain_pipeline import retrain
with open("monitoring/monitoring_metrics.json") as f:
    metrics = json.load(f)


DRIFT_THRESHOLD = 30      # % features drifted
ROC_THRESHOLD   = 0.75
RECALL_THRESHOLD = 0.65

def should_retrain(metrics):

    if metrics["data_drift"]:
        return True

    if metrics["drifted_features_pct"] > DRIFT_THRESHOLD:
        return True

    if metrics["roc_auc"] < ROC_THRESHOLD:
        return True

    if metrics["recall"] < RECALL_THRESHOLD:
        return True

    return False

if should_retrain(metrics):
    print("⚠️ Retraining triggered")
else:
    print("✅ Model stable — no retraining")



if should_retrain(metrics):
    retrain()