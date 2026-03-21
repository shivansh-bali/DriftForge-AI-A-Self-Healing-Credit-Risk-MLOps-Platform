import pandas as pd
import joblib

model = joblib.load("models/model_v1.pkl")

data = {
    "loan_amnt": 15000,
    "int_rate": 14.2,
    "annual_inc": 75000,
    "dti": 18,
    "installment": 400,
    "revol_util": 52,
    "total_acc": 22,
    "grade": "C",
    "home_ownership": "RENT",
    "verification_status": "Verified",
    "purpose": "debt_consolidation",
    "term": "36 months"
}

df = pd.DataFrame([data])

probs = model.predict_proba(df)

print("Full probs:", probs)
print("Default probability:", probs[0][1])
