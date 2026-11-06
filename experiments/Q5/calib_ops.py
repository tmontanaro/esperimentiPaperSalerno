"""Q5 a)+b): calibrazione OOF (clean) e punti operativi. Solo aggregati. Uso: python3 experiments/Q5/calib_ops.py [debug_n]"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))
import pipe2_lib as P
from sklearn.isotonic import IsotonicRegression

OUT = ROOT / "results/Q5"; OUT.mkdir(exist_ok=True, parents=True)
if (OUT / "calib_ops.json").exists():
    print("già fatto"); sys.exit(0)
data, dev = P.load_dev()
ens = np.load(ROOT / "results/Q1/main/oof_ensemble.npy")
clean = dev.lvef_known.values & ~np.isnan(ens[:, 0])
print("n clean", clean.sum())
d = dev[clean].reset_index(drop=True); E = ens[clean]
g = d.patient_key.values; fold = d.fold.values
NB = 200
names = ["shd_moderate_or_greater_flag", "lvef_lte_45_flag", "rv_systolic_dysfunction_moderate_or_greater_flag", "aortic_stenosis_moderate_or_greater_flag"]
names = [n for n in names if n in P.TARGETS]
res = {"n_clean": int(clean.sum()), "prev_train_clean": float(d[d.split == "train"].shd_moderate_or_greater_flag.mean()),
       "prev_val_clean": float(d[d.split == "val"].shd_moderate_or_greater_flag.mean()), "calib": {}, "ops": {}}
rel_rows = []
for nm in names:
    j = P.TARGETS.index(nm); y = d[nm].values.astype(int); p = E[:, j]
    # isotonica fuori dal fold: fit sugli OOF degli altri fold, applicata al fold lasciato fuori
    pi = np.zeros_like(p)
    for f in range(5):
        m = fold == f
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[~m], y[~m]); pi[m] = iso.predict(p[m])
    r = {"prev": float(y.mean()), "n_pos": int(y.sum())}
    for lab, q in (("raw", p), ("iso", pi)):
        r[f"ece_{lab}"] = P.boot_stat_cluster(y, q, g, P.ece, n_boot=NB)
        r[f"brier_{lab}"] = P.boot_stat_cluster(y, q, g, P.brier, n_boot=NB)
    r["auc_raw"] = P.auc_ci_cluster(y, p, g, n_boot=NB)
    r["auc_iso"] = P.auc_ci_cluster(y, pi, g, n_boot=NB)
    # media previsto vs osservato e pendenza/intercetta logistica
    from sklearn.linear_model import LogisticRegression
    lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    lr = LogisticRegression(C=1e6).fit(lg[:, None], y)
    r["slope"], r["intercept"] = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    r["mean_pred"] = float(p.mean())
    # per split (shift di prevalenza)
    r["by_split"] = {}
    for s in ("train", "val"):
        m = (d.split == s).values
        r["by_split"][s] = {"n": int(m.sum()), "prev": float(y[m].mean()), "mean_pred": float(p[m].mean()),
                             "ece": P.boot_stat_cluster(y[m], p[m], g[m], P.ece, n_boot=NB)}
    rr = P.reliability(y, p); rr["target"] = nm; rr["tipo"] = "raw"; rel_rows.append(rr)
    ri = P.reliability(y, pi); ri["target"] = nm; ri["tipo"] = "iso"; rel_rows.append(ri)
    res["calib"][nm] = r
    print(nm, {k: v for k, v in r.items() if k.startswith("ece") or k.startswith("brier")}, flush=True)
    # punti operativi (solo SHD e LVEF<=45)
    if nm in names[:2]:
        o = {}
        order = np.argsort(-p)
        pos = y == 1; neg = ~pos
        thr_sens = np.quantile(p[pos], 0.10)  # sens ~ 90%
        thr_spec = np.quantile(p[neg], 0.90)  # spec ~ 90%
        grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 197)))
        yj = [(P.sens_spec_at(y, p, t)["sens"] + P.sens_spec_at(y, p, t)["spec"] - 1) for t in grid]
        thr_y = grid[int(np.argmax(yj))]
        for lab, t in (("sens90", thr_sens), ("spec90", thr_spec), ("youden", thr_y)):
            base = P.sens_spec_at(y, p, t); row = {"soglia": float(t)}
            for k in ("sens", "spec", "ppv", "npv"):
                row[k] = P.boot_stat_cluster(y, p, g, lambda yy, pp, k=k, t=t: P.sens_spec_at(yy, pp, t)[k], n_boot=NB)
            row["n_flag_pct"] = float((p >= t).mean())
            o[lab] = row
        # stessa soglia, due prevalenze: PPV/NPV a prevalenza 10%, 30% (calcolo analitico da sens/spec)
        res["ops"][nm] = o
        print(nm, {k: (round(v["sens"][0], 3), round(v["spec"][0], 3), round(v["ppv"][0], 3)) for k, v in o.items()}, flush=True)
pd.concat(rel_rows).to_csv(OUT / "reliability.csv", index=False)
(OUT / "calib_ops.json").write_text(json.dumps(res, indent=1))
print("OK")
