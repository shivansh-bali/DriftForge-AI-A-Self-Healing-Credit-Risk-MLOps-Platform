import pandas as pd

X_train = pd.read_csv("data/processed/X_train.csv")
X_test  = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()
from sklearn.model_selection import train_test_split

X_train, X_dev, y_train, y_dev = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)



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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
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

results = {}

for name, pipeline in models.items():

    pipeline.fit(X_train, y_train)

    probs = pipeline.predict_proba(X_dev)[:, 1]

    auc = roc_auc_score(y_dev, probs)

    results[name] = auc

    print(f"{name} ROC-AUC: {auc:.4f}")
best_model_name = max(results, key=results.get)
best_pipeline = models[best_model_name]

print("Best model:", best_model_name)
probs_test = best_pipeline.predict_proba(X_test)[:, 1]

test_auc = roc_auc_score(y_test, probs_test)

print("Final Test ROC-AUC:", test_auc)
import joblib

joblib.dump(
    best_pipeline,
    "models/model_v1.pkl"
)
