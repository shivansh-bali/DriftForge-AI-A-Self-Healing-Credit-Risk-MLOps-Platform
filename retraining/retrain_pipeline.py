import shutil
from versioning import get_next_version, update_registry,load_production_model
import joblib
from logger import log_retraining_event

import mlflow
import mlflow.sklearn



def retrain(): 
    print("Starting retraining pipeline...")
    import pandas as pd
    import joblib

    df = pd.read_csv("logs/predictions.csv")

    X_new = df.drop(
        columns=["prediction", "actual_default", "timestamp"],
        errors="ignore"
    )

    y_new = df["actual_default"]
    X_old = pd.read_csv("data/processed/X_train.csv")
    y_old = pd.read_csv("data/processed/y_train.csv").squeeze()

    X_combined = pd.concat([X_old, X_new])
    y_combined = pd.concat([y_old, y_new])

    

    old_model = load_production_model()

    from sklearn.base import clone

    new_model = clone(old_model)

    mlflow.set_experiment("Credit_Risk_Retraining")

    with mlflow.start_run():

        new_model.fit(X_combined, y_combined)

        from sklearn.metrics import roc_auc_score

        X_test = pd.read_csv("data/processed/X_test.csv")
        y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

        old_probs = old_model.predict_proba(X_test)[:, 1]
        old_auc = roc_auc_score(y_test, old_probs)

        new_probs = new_model.predict_proba(X_test)[:, 1]
        new_auc = roc_auc_score(y_test, new_probs)

        print("Old ROC:", old_auc)
        print("New ROC:", new_auc)

        mlflow.log_param("training_rows", len(X_combined))

        mlflow.log_metric("old_auc", old_auc)
        mlflow.log_metric("new_auc", new_auc)
        mlflow.log_metric("auc_improvement", new_auc - old_auc)



        if new_auc > old_auc:
            print("New model better — deploying!")
            next_version = get_next_version()
            new_model_path = f"models/model_v{next_version}.pkl"
            joblib.dump(
                new_model,
                new_model_path
            )
            update_registry(next_version)
            

            mlflow.sklearn.log_model(
                new_model,
                artifact_path="model",
                registered_model_name="CreditRiskModel"
            )
            mlflow.set_tag("deployment_status", "promoted")

        else:
            print("Old model retained!")
            mlflow.set_tag("deployment_status", "rejected")

        log_retraining_event(old_auc,new_auc)    


    




