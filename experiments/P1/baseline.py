import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from sklearn.inspection import permutation_importance
from lib import echonext_lib as lib

debug = len(sys.argv) > 1 and sys.argv[1] == "debug"
out = ROOT / "results/P1" / ("debug" if debug else "")
out.mkdir(parents=True, exist_ok=True)
cfg = json.load(open(ROOT / "agentic/pipeline_state.json"))["config"]
T = cfg["primary_target"]; th = cfg["thresholds"]
d = lib.load_data(debug_n=2000 if debug else None)
res, models_fit = {}, {}
sets = {"eta_sesso": ["age_at_ecg", "sex"], "tutte": lib.ECG_FEATURES}
for k, f in sets.items():
    Xtr, ytr, _ = lib.get_xy(d.train, T, f); Xva, yva, _ = lib.get_xy(d.val, T, f)
    zoo = lib.model_zoo()
    tab, p = lib.eval_models(Xtr, ytr, Xva, yva, zoo)
    tab["features"] = k; res[k] = tab; models_fit[k] = zoo
    print(k); lib.show(tab.drop(columns="features"))
pd.concat(res.values()).to_csv(out / "val_models.csv", index=False)
best = {k: res[k].iloc[0] for k in res}
# controllo di coerenza: CV raggruppata per paziente (hist_gb, tutte le feature)
Xtr, ytr, dtr = lib.get_xy(d.train, T, lib.ECG_FEATURES)
oof = lib.oof_proba(lib.model_zoo()["hist_gb"], Xtr, ytr, dtr["patient_key"].values)
cv = lib.metrics_binary(ytr, oof)
Xva, yva, _ = lib.get_xy(d.val, T, lib.ECG_FEATURES)
hv = res["tutte"].set_index("model").loc["hist_gb"]
print("CV hist_gb", cv, "val hist_gb", hv.auroc)
bn = best["tutte"].model; m = models_fit["tutte"][bn]
pi = permutation_importance(m, Xva, yva, scoring="roc_auc", n_repeats=20, random_state=42)
pid = pd.DataFrame({"feature": lib.ECG_FEATURES, "mean": pi.importances_mean, "std": pi.importances_std}).sort_values("mean", ascending=False)
pid.to_csv(out / "perm_importance.csv", index=False); lib.show(pid)
gain = float(best["tutte"].auroc - best["eta_sesso"].auroc)
sig = bool(gain >= th["tabular_signal_min_gain"] and best["tutte"].auroc_lo > 0.55)
summary = {"target": T, "debug": debug, "best_eta_sesso": best["eta_sesso"].to_dict(), "best_tutte": best["tutte"].to_dict(),
           "gain": gain, "cv_hist_gb": cv, "val_hist_gb_auroc": float(hv.auroc), "TABULAR_SIGNAL": sig,
           "perm": pid.to_dict("records")}
json.dump(summary, open(out / "summary.json", "w"), indent=1, default=str)
print("gain", gain, "TABULAR_SIGNAL", sig, "best", bn)
if not debug:
    src = '''import joblib
from pathlib import Path
import pandas as pd
_M = joblib.load(Path(__file__).parent / "model.joblib")
def predict(df, lib):
    X = df[lib.ECG_FEATURES].astype(float)
    return pd.DataFrame({"score": _M.predict_proba(X)[:, 1], "in_scope": True}, index=df.index)
'''
    vm = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
          for k, v in best["tutte"].to_dict().items() if k != "features"}
    lib.save_candidate(ROOT / "results/P1/candidates/global_best", "global_best", T, "global",
                       f"{bn} su 7 feature ECG, addestrato su train", vm, src, {"model.joblib": m})
