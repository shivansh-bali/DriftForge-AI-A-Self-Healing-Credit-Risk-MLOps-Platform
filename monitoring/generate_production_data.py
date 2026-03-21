import pandas as pd
import joblib
import numpy as np
# Load model
model = joblib.load("models/model_v1.pkl")

# Load test set
X_test_1 = pd.read_csv("data/processed/X_test.csv")


probs = model.predict_proba(X_test_1)[:, 1]

X_test_1["prediction"] = probs

from datetime import datetime

X_test_1["timestamp"] = datetime.now()
X_test_1["loan_amnt"] *= 1.6
X_test_1["int_rate"] *= 1.3
X_test_1["annual_inc"] *= 0.75
X_test_1["dti"] *= 1.4
X_test_1["total_acc"] += 5
X_test_1["revol_util"] += np.random.normal(10, 5, len(X_test_1))
grade_map = {"A":"C","B":"D","C":"E","D":"F","E":"G"}
X_test_1["grade"] = X_test_1["grade"].replace(grade_map)

X_test_1.to_csv(
    "logs/predictions.csv",
    index=False
)
