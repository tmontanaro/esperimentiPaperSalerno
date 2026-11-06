import sys, json
sys.path.insert(0, "lib")
import numpy as np, pandas as pd
import pipe2_lib as p2
data, dev = p2.load_dev()
out = {}
out["N_DEV"] = int(len(dev)); out["N_PATIENTS"] = int(dev.patient_key.nunique())
out["N_CLEAN"] = int(dev.lvef_known.sum()); out["CLEAN_FRACTION"] = float(dev.lvef_known.mean())
out["clean_by_split"] = {s: [int((dev.split==s).sum()), int(dev.lvef_known[dev.split==s].sum())] for s in ("train","val")}
out["prev_shd_all"] = float(dev[p2.MAIN].mean()); out["prev_shd_clean"] = float(dev[p2.MAIN][dev.lvef_known].mean())
out["folds"] = {int(f): {"n_ecg": int((dev.fold==f).sum()), "n_pazienti": int(dev.patient_key[dev.fold==f].nunique())} for f in range(5)}
# derivazioni: ~2000 ECG di train
tr = dev[dev.split=="train"]
rng = np.random.default_rng(42)
sel = rng.choice(tr.index.values, 2000, replace=False)
X = p2.WaveReader(data, dev).get(sel).astype(np.float64)  # n,2500,12
X = X.reshape(-1, 12)
X = X[np.isfinite(X).all(1)]
def reg(t, a, b):
    A = np.c_[X[:, a], X[:, b]]; y = X[:, t]
    A1 = np.c_[A, np.ones(len(A))]
    c, *_ = np.linalg.lstsq(A1, y, rcond=None)
    r2 = 1 - ((y - A1@c)**2).sum() / ((y - y.mean())**2).sum()
    return {"coef": [float(c[0]), float(c[1])], "R2": float(r2)}
exp = {"c1~c0+c2": (1,0,2,(1,1)), "c3~c0+c1": (3,0,1,(-1,-1)), "c4~c0+c2": (4,0,2,(1,-1)), "c5~c1+c2": (5,1,2,(1,1))}
rel = {}; ok = 0
for k,(t,a,b,sg) in exp.items():
    r = reg(t,a,b); r["segni_attesi"] = bool(np.sign(r["coef"][0])==sg[0] and np.sign(r["coef"][1])==sg[1])
    r["ok"] = bool(r["R2"]>=0.95 and r["segni_attesi"]); ok += r["ok"]; rel[k] = r
out["relations"] = rel
out["LEAD_ORDER_OK"] = "yes" if ok==4 else ("partial" if ok>0 else "no")
out["n_samples_used"] = int(len(X))
json.dump(out, open("results/Q0/leads_pop.json","w"), indent=1)
print(json.dumps(out, indent=1))
