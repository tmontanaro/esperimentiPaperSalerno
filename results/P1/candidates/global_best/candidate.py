import joblib
from pathlib import Path
import pandas as pd
_M = joblib.load(Path(__file__).parent / "model.joblib")
def predict(df, lib):
    X = df[lib.ECG_FEATURES].astype(float)
    return pd.DataFrame({"score": _M.predict_proba(X)[:, 1], "in_scope": True}, index=df.index)
