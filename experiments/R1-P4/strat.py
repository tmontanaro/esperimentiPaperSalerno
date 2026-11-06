import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/idalab/Desktop/Teo/esperimentiClaude/echonext-agentic")
import importlib.util
import lib.echonext_lib as lib
ROOT = lib.ROOT; OUT = ROOT / "results/R1-P4"; OUT.mkdir(parents=True, exist_ok=True)
data = lib.load_data(); val = lib.add_groups(data.val).reset_index(drop=True)
z = np.load(ROOT / "results/P4/val_pred_emb.npz", allow_pickle=True)
targets = list(z["targets"]); pred = z["pred"]; rows = z["row_in_split"]
assert (rows == val.row_in_split.values).all()
print("targets", targets[:3], "pred range", float(pred.min()), float(pred.max()), pred.shape)
T = lib.TARGET; ic = targets.index(T)
cnn = pred[:, ic]
if cnn.min() < 0 or cnn.max() > 1:
    cnn = 1 / (1 + np.exp(-cnn))
y = val[T].values.astype(int)
spec = importlib.util.spec_from_file_location("p1c", ROOT / "results/P1/candidates/global_best/candidate.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
tab = m.predict(val, lib)["score"].values
print("global AUROC cnn/tab", lib.safe_auc(y, cnn), lib.safe_auc(y, tab))
rng = np.random.default_rng(42)


def paired(yy, a, b, n=1000):
    d = []
    for _ in range(n):
        i = rng.integers(0, len(yy), len(yy))
        if len(np.unique(yy[i])) < 2:
            continue
        d.append(lib.safe_auc(yy[i], a[i]) - lib.safe_auc(yy[i], b[i]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


val["_cnn"] = cnn; val["_tab"] = tab
out = []


def run(col):
    for g, d in val.groupby(col, dropna=False):
        yy = d[T].values.astype(int)
        r = {"variabile": col, "livello": str(g), "n": len(d), "n_pos": int(yy.sum())}
        if yy.sum() < 10 or (len(yy) - yy.sum()) < 10:
            r["nota"] = "troppo pochi positivi/negativi"; out.append(r); continue
        mc = lib.metrics_binary(yy, d._cnn.values, 1000); mt = lib.metrics_binary(yy, d._tab.values, 1000)
        lo, hi = paired(yy, d._cnn.values, d._tab.values)
        r.update(cnn=mc["auroc"], cnn_lo=mc["auroc_lo"], cnn_hi=mc["auroc_hi"], tab=mt["auroc"], tab_lo=mt["auroc_lo"],
                 tab_hi=mt["auroc_hi"], diff=mc["auroc"] - mt["auroc"], diff_lo=lo, diff_hi=hi)
        out.append(r)


for col in ["race_ethnicity", "location_setting", "sex", "age_bin"]:
    if col in val.columns:
        run(col)
    else:
        print("manca", col)
df = pd.DataFrame(out); df.to_csv(OUT / "val_stratified.csv", index=False)
print(df.round(3).to_string())

spec = importlib.util.spec_from_file_location("cc", ROOT / "results/P4/candidates/cnn_multilabel/candidate.py")
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
sub = val.sample(500, random_state=42).sort_index()
inp = sub.drop(columns=["_cnn", "_tab"])
res = c.predict(inp, lib)
print("columns", list(res.columns), "index equal", res.index.equals(inp.index), "in_scope all", bool(res.in_scope.all()), "dtype", res.score.dtype)
diff = np.abs(res.score.values - sub["_cnn"].values)
print("max|diff|", diff.max(), "mean", diff.mean(), "score range", res.score.min(), res.score.max())
json.dump({"colonne": list(res.columns), "indice_uguale": bool(res.index.equals(inp.index)), "n": 500,
           "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
           "auroc_500": lib.safe_auc(sub[T].values.astype(int), res.score.values)}, open(OUT / "candidate_check.json", "w"))
