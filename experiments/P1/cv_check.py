"""Indagine CV>>validation: CV raggruppata sul train per tutti i modelli/feature set, confronto con validation (n piccolo)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import pandas as pd
from lib import echonext_lib as lib
d = lib.load_data(); T = "shd_moderate_or_greater_flag"
res = pd.read_csv(ROOT/"results/P1/val_results.csv")
rows = []
for s, f in {"eta_sesso": ["age_at_ecg", "sex"], "tutte_ecg": lib.ECG_FEATURES}.items():
    X, y, dd = lib.get_xy(d.train, T, f)
    for n, m in lib.model_zoo().items():
        o = lib.oof_proba(m, X, y, dd.patient_key.values); c = lib.metrics_binary(y, o)
        v = res[(res.features == s) & (res.model == n)].iloc[0]
        rows.append({"features": s, "model": n, "cv_auroc": c["auroc"], "cv_lo": c["auroc_lo"], "cv_hi": c["auroc_hi"],
                     "val_auroc": v.auroc, "val_lo": v.auroc_lo, "val_hi": v.auroc_hi})
r = pd.DataFrame(rows); r["diff"] = r.cv_auroc - r.val_auroc
r.to_csv(ROOT/"results/P1/cv_vs_val.csv", index=False); lib.show(r)
print("pazienti unici train / righe:", d.train.patient_key.nunique(), len(d.train))
