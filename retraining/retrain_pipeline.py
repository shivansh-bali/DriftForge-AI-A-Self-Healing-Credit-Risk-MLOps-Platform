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

   

    mlflow.set_experiment("Credit_Risk_Retraining")
    
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    num_cols = [
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "dti",
    "installment",
    "revol_util",
    "total_acc"
    ]

    cat_cols = [
        "grade",
        "home_ownership",
        "verification_status",
        "purpose",
        "term"
    ]
    num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])
    models = {

        "log_reg": Pipeline([
            ("preprocessing", preprocessor),
            ("model", LogisticRegression(max_iter=1000))
        ]),

        "random_forest": Pipeline([
            ("preprocessing", preprocessor),
            ("model", RandomForestClassifier(n_estimators=200))
        ]),

        "xgboost": Pipeline([
            ("preprocessing", preprocessor),
            ("model", XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6
            ))
        ])
    }
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    X_train, X_dev, y_train, y_dev = train_test_split(
    X_combined,
    y_combined,
    test_size=0.2,
    stratify=y_combined,
    random_state=42
    )
    results = {}

    for name, pipeline in models.items():

        pipeline.fit(X_train, y_train)

        probs = pipeline.predict_proba(X_dev)[:, 1]

        auc = roc_auc_score(y_dev, probs)

        results[name] = auc

        print(f"{name} ROC-AUC: {auc:.4f}")
    best_model_name = max(results, key=results.get)
    new_model = models[best_model_name]
    new_auc = results[best_model_name]
    with mlflow.start_run():

        

        from sklearn.metrics import roc_auc_score

        X_test = pd.read_csv("data/processed/X_test.csv")
        y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

        old_probs = old_model.predict_proba(X_dev)[:, 1]
        old_auc = roc_auc_score(y_dev, old_probs)
        new_probs = new_model.predict_proba(X_test)[:, 1]
        new_test_auc = roc_auc_score(y_test, new_probs)
        

        print("Old ROC:", old_auc)
        print("New ROC:", new_auc)

        mlflow.log_param("training_rows", len(X_combined))

        mlflow.log_metric("old_auc", old_auc)
        mlflow.log_metric("new_auc", new_auc)
        mlflow.log_metric("auc_improvement", new_auc - old_auc)



        if new_auc > old_auc:
            print("New model better — deploying!")
            print("New model auc:",new_test_auc)
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


    




