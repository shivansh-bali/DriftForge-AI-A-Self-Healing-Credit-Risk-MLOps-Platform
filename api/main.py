from fastapi import FastAPI
import joblib
import pandas as pd
from retraining.versioning import load_production_model
from api.schema import LoanApplication
from api.utils import log_prediction

import os

app = FastAPI(title="Credit Risk Model API")



model = load_production_model()



@app.get("/")
def home():
    return {
        "status": "Credit Risk API Running"}
@app.post("/predict")
def predict(application: LoanApplication):

    data = application.model_dump()

    df = pd.DataFrame([data])

    probability = model.predict_proba(df)[0][1]

    prediction = int(probability > 0.5)

    log_prediction(data,probability)

    return {
        "default_probability": float(probability),
        "default_prediction": prediction
    }



@app.post("/run-monitoring")
def run_monitoring():

    os.system("python run_system.py")

    return {"status": "Monitoring + Retraining Triggered"}
