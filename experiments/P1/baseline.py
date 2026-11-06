"""P1 baseline: età+sesso vs feature ECG, 5 modelli, CV per paziente, permutation importance, candidato globale.
Uso: python3 experiments/P1/baseline.py [debug]"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))
import pandas as pd
import echonext_lib as lib
from sklearn.inspection import permutation_importance

debug = len(sys.argv) > 1 and sys.argv[1] == "debug"
out = ROOT / "results" / "P1" / ("debug" if debug else "")
out.mkdir(parents=True, exist_ok=True)
cfg = json.load(open(ROOT / "agentic" / "pipeline_state.json"))["config"]
target = cfg["primary_target"]
gain_min = cfg["thresholds"]["tabular_signal_min_gain"]
data = lib.load_data(debug_n=2000 if debug else None)
res = {}
fitted = {}
for name, feats in {"eta_sesso": ["age_at_ecg", "sex"], "tutte": lib.ECG_FEATURES}.items():
    Xtr, ytr, _ = lib.get_xy(data.train, target, feats)
    Xva, yva, _ = lib.get_xy(data.val, target, feats)
    zoo = lib.model_zoo()
    tab, preds = lib.eval_models(Xtr, ytr, Xva, yva, zoo)
    tab.to_csv(out / f"models_{name}.csv", index=False)
    print(name)
    lib.show(tab)
    res[name] = tab.to_dict("records")
    fitted[name] = (zoo, Xva, yva)

best = {k: v[0] for k, v in res.items()}
bm = best["tutte"]["model"]
# CV di coerenza (hist_gb, tutte le feature, train, GroupKFold per patient_key)
Xtr, ytr, dtr = lib.get_xy(data.train, target)
oof = lib.oof_proba(lib.model_zoo()["hist_gb"], Xtr, ytr, dtr["patient_key"].values)
cv_auc = lib.safe_auc(ytr, oof)
val_hgb = next(r for r in res["tutte"] if r["model"] == "hist_gb")["auroc"]
print("CV hist_gb", cv_auc, "val hist_gb", val_hgb)
# permutation importance su validation (miglior modello, tutte le feature)
zoo, Xva, yva = fitted["tutte"]
pi = permutation_importance(zoo[bm], Xva, yva, scoring="roc_auc", n_repeats=10, random_state=42)
pidf = pd.DataFrame({"feature": Xva.columns, "delta_auroc": pi.importances_mean, "sd": pi.importances_std}).sort_values("delta_auroc", ascending=False)
pidf.to_csv(out / "permutation_importance.csv", index=False)
lib.show(pidf)
gain = best["tutte"]["auroc"] - best["eta_sesso"]["auroc"]
signal = bool(gain >= gain_min and best["tutte"]["auroc_lo"] > 0.55)
summary = {"target": target, "best_eta_sesso": best["eta_sesso"], "best_tutte": best["tutte"], "gain": gain,
           "cv_auc_hist_gb": cv_auc, "val_auc_hist_gb": val_hgb, "tabular_signal": signal}
json.dump(summary, open(out / "summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
if not debug:
    src = '''import joblib
from pathlib import Path
import pandas as pd
_M = joblib.load(Path(__file__).parent / "model.joblib")
def predict(df, lib):
    X = df[lib.ECG_FEATURES].astype(float)
    return pd.DataFrame({"score": _M.predict_proba(X)[:, 1], "in_scope": True}, index=df.index)
'''
    b = best["tutte"]
    lib.save_candidate(out / "candidates" / "global_best", "global_best", target, "global",
                       f"{bm} addestrato sul train con le 7 feature ECG",
                       {k: b[k] for k in ("n", "n_pos", "auroc", "auroc_lo", "auroc_hi", "auprc")},
                       src, {"model.joblib": zoo[bm]})
