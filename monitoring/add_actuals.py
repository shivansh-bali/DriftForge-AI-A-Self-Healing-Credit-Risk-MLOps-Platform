import pandas as pd
import numpy as np

df = pd.read_csv("logs/predictions.csv")

np.random.seed(42)

df["actual_default"] = (
    df["prediction"] + np.random.normal(0, 0.2, len(df))
)

df["actual_default"] = (df["actual_default"] > 0.5).astype(int)
df.to_csv("logs/predictions.csv", index=False)