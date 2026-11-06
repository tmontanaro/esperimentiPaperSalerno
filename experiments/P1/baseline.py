"""P1 baseline: età+sesso vs 7 feature ECG, 5 modelli. Uso: python3 baseline.py [debug]"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from lib import echonext_lib as lib
from sklearn.inspection import permutation_importance

debug = len(sys.argv) > 1 and sys.argv[1] == "debug"
out = ROOT / "results" / "P1" / ("debug" if debug else "")
out.mkdir(parents=True, exist_ok=True)
cfg = json.loads((ROOT / "agentic/pipeline_state.json").read_text())["config"]
T = cfg["primary_target"]; thr = cfg["thresholds"]
data = lib.load_data(debug_n=2000 if debug else None)
sets = {"eta_sesso": ["age_at_ecg", "sex"], "tutte_ecg": list(lib.ECG_FEATURES)}
tabs, fitted = [], {}
for sname, feats in sets.items():
    Xtr, ytr, dtr = lib.get_xy(data.train, T, feats)
    Xva, yva, dva = lib.get_xy(data.val, T, feats)
    zoo = lib.model_zoo()
    tab, p = lib.eval_models(Xtr, ytr, Xva, yva, zoo)
    tab.insert(0, "features", sname); tabs.append(tab)
    fitted[sname] = zoo
res = pd.concat(tabs, ignore_index=True)
res.to_csv(out / "val_models.csv", index=False)
lib.show(res)
best = {s: res[res.features == s].iloc[0] for s in sets}
gain = best["tutte_ecg"].auroc - best["eta_sesso"].auroc
signal = bool(gain >= thr["tabular_signal_min_gain"] and best["tutte_ecg"].auroc_lo > 0.55)
bname = best["tutte_ecg"].model

# coerenza CV raggruppata (hist_gb, train, tutte le feature)
Xtr, ytr, dtr = lib.get_xy(data.train, T)
oof = lib.oof_proba(lib.model_zoo()["hist_gb"], Xtr, ytr, dtr.patient_key.values)
cv = lib.metrics_binary(ytr, oof)
hg_val = res[(res.features == "tutte_ecg") & (res.model == "hist_gb")].iloc[0].auroc
cv_diff = cv["auroc"] - hg_val

# permutation importance su validation, miglior modello (tutte le feature)
Xva, yva, dva = lib.get_xy(data.val, T)
m = fitted["tutte_ecg"][bname]
pi = permutation_importance(m, Xva, yva, scoring="roc_auc", n_repeats=10, random_state=42)
pid = pd.DataFrame({"feature": Xva.columns, "mean_drop_auroc": pi.importances_mean, "std": pi.importances_std}).sort_values("mean_drop_auroc", ascending=False)
pid.to_csv(out / "perm_importance.csv", index=False)
lib.show(pid)

summ = {"target": T, "debug": debug, "best_all": best["tutte_ecg"].to_dict(), "best_age_sex": best["eta_sesso"].to_dict(),
        "gain": float(gain), "tabular_signal": signal, "cv_hist_gb": cv, "val_hist_gb_auroc": float(hg_val), "cv_minus_val": float(cv_diff),
        "best_model": bname}
(out / "summary.json").write_text(json.dumps(summ, indent=2, default=float))
print(json.dumps(summ, indent=1, default=float))

if not debug:
    cand_src = '''import joblib
from pathlib import Path
import pandas as pd
_M = joblib.load(Path(__file__).parent / "model.joblib")
def predict(df, lib):
    X = df[lib.ECG_FEATURES].astype(float)
    return pd.DataFrame({"score": _M.predict_proba(X)[:, 1], "in_scope": True}, index=df.index)
'''
    vm = {k: float(best["tutte_ecg"][k]) for k in ["n", "n_pos", "auroc", "auroc_lo", "auroc_hi", "auprc"]}
    lib.save_candidate(ROOT / "results/P1/candidates/global_best", "global_best", T, "global",
                       f"Modello globale {bname} su 7 feature ECG, addestrato su train", vm, cand_src,
                       {"model.joblib": m})
