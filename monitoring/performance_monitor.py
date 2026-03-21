from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score
)
import pandas as pd
df = pd.read_csv("logs/predictions.csv")
y_true = df["actual_default"]
y_pred_prob = df["prediction"]
y_pred = (y_pred_prob > 0.5).astype(int)


roc = roc_auc_score(y_true, y_pred_prob)
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)


print("=== Production Performance ===")
print(f"ROC-AUC: {roc:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")


if roc < 0.75:
    print("⚠️ ROC dropped — Concept drift detected")

if rec < 0.65:
    print("⚠️ Recall dropped — Risk detection failing")

df["prediction"].hist()
print("Avg default probability:",
      df["prediction"].mean())

import json
from datetime import datetime

metrics_path = "monitoring/monitoring_metrics.json"

with open(metrics_path) as f:
    metrics = json.load(f)

metrics["roc_auc"] = float(roc)
metrics["recall"] = float(rec)
metrics["last_updated"] = str(datetime.now())

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)
