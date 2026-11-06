"""P1 baseline: età+sesso vs tutte le feature ECG, 5 modelli, CV per paziente, permutation importance, candidato globale.
Uso: python3 experiments/P1/baseline.py [--debug]   (debug: debug_n=2000, output in results/P1/debug/)"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, joblib
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, roc_auc_score
from lib import echonext_lib as lib

DEBUG = "--debug" in sys.argv
out = ROOT / "results/P1" / ("debug" if DEBUG else "")
out.mkdir(parents=True, exist_ok=True)
cfg = json.load(open(ROOT / "agentic/pipeline_state.json"))["config"]
T = cfg["primary_target"]
th = cfg["thresholds"]

data = lib.load_data(debug_n=2000 if DEBUG else None)
sets = {"eta_sesso": ["age_at_ecg", "sex"], "tutte_ecg": lib.ECG_FEATURES}
rows, preds = [], {}
for sname, feats in sets.items():
    Xtr, ytr, dtr = lib.get_xy(data.train, T, feats)
    Xva, yva, dva = lib.get_xy(data.val, T, feats)
    tab, p = lib.eval_models(Xtr, ytr, Xva, yva)
    tab.insert(0, "features", sname)
    rows.append(tab); preds[sname] = p
res = pd.concat(rows, ignore_index=True)
res.to_csv(out / "val_results.csv", index=False)
lib.show(res[["features", "model", "n", "n_pos", "auroc", "auroc_lo", "auroc_hi", "auprc"]])

best = {s: res[res.features == s].sort_values("auroc").iloc[-1] for s in sets}
A, B = best["eta_sesso"], best["tutte_ecg"]
gain = float(B.auroc - A.auroc)
signal = bool(gain >= th["tabular_signal_min_gain"] and B.auroc_lo > 0.55)
print("migliore età+sesso:", A.model, round(A.auroc, 4), "| tutte:", B.model, round(B.auroc, 4), "gain", round(gain, 4), "SIGNAL", signal)

# coerenza CV raggruppata sul train (hist_gb, tutte le feature)
Xtr, ytr, dtr = lib.get_xy(data.train, T)
oof = lib.oof_proba(lib.model_zoo()["hist_gb"], Xtr, ytr, dtr["patient_key"].values)
cv = lib.metrics_binary(ytr, oof)
val_hgb = res[(res.features == "tutte_ecg") & (res.model == "hist_gb")].iloc[0]
cv_diff = cv["auroc"] - float(val_hgb.auroc)
print("CV train hist_gb:", cv, "| val hist_gb:", round(val_hgb.auroc, 4), "diff", round(cv_diff, 4))

# permutation importance su validation, modello migliore (rifit su train)
Xva, yva, dva = lib.get_xy(data.val, T)
model = lib.model_zoo()[B.model]
model.fit(Xtr, ytr)
pi = permutation_importance(model, Xva, yva, scoring="roc_auc", n_repeats=20, random_state=lib.SEED)
pit = pd.DataFrame({"feature": lib.ECG_FEATURES, "delta_auroc": pi.importances_mean, "sd": pi.importances_std}).sort_values("delta_auroc", ascending=False)
pit.to_csv(out / "permutation_importance.csv", index=False)
lib.show(pit)

pv = model.predict_proba(Xva)[:, 1]
vm = lib.metrics_binary(yva, pv)
summary = {"best_age_sex": {"model": A.model, **{k: float(A[k]) for k in ["auroc", "auroc_lo", "auroc_hi", "auprc"]}},
           "best_all": {"model": B.model, **{k: float(B[k]) for k in ["auroc", "auroc_lo", "auroc_hi", "auprc"]}},
           "gain": gain, "tabular_signal": signal, "cv_train_hist_gb": cv, "cv_minus_val": cv_diff, "val_best": vm}
json.dump(summary, open(out / "summary.json", "w"), indent=2)

if not DEBUG:
    src = '''import joblib
from pathlib import Path
import pandas as pd
_M = joblib.load(Path(__file__).parent / "model.joblib")
def predict(df, lib):
    X = df[lib.ECG_FEATURES].astype(float)
    return pd.DataFrame({"score": _M.predict_proba(X)[:, 1], "in_scope": True}, index=df.index)
'''
    lib.save_candidate(out / "candidates/global_best", "global_best", T, "global",
                       f"{B.model} addestrato sul train con le 7 feature ECG (target composito)", vm, src,
                       {"model.joblib": model})
print("fatto")
