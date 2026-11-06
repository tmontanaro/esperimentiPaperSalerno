"""P2 conditions. Uso: python3 conditions.py [debug]
(a) 12 target x 3 modelli; (b) locale vs globale per strati; (c) regole singole/coppie AND (selezione su OOF train, verifica su val)."""
import sys, json, itertools
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, joblib
from sklearn.base import clone
from lib import echonext_lib as lib

debug = len(sys.argv) > 1 and sys.argv[1] == "debug"
out = ROOT / "results" / "P2" / ("debug" if debug else "")
out.mkdir(parents=True, exist_ok=True)
cfg = json.loads((ROOT / "agentic/pipeline_state.json").read_text())["config"]
thr = cfg["thresholds"]; T0 = cfg["primary_target"]
MIN_TR, MIN_VA = (30, 10) if not debug else (10, 3)
data = lib.load_data(debug_n=2000 if debug else None)
tr, va = lib.add_groups(data.train), lib.add_groups(data.val)
zoo0 = lib.model_zoo()
MODELS = ["logreg", "hist_gb", "mlp_dense"]
def newm(n): return clone(zoo0[n])
nb = 300 if not debug else 100
log = {"val_evals": 0, "rules_tried": 0}

# ---------------------------------------------------------------- (a)
rows = []; skipped = []
for T in lib.FLAG_COLS:
    Xtr, ytr, dtr = lib.get_xy(tr, T); Xva, yva, dva = lib.get_xy(va, T)
    if ytr.sum() < MIN_TR or yva.sum() < MIN_VA:
        skipped.append({"target": T, "n_pos_train": int(ytr.sum()), "n_pos_val": int(yva.sum())}); continue
    for mn in MODELS:
        cp = out / f"model_{T}_{mn}.joblib"
        if cp.exists(): m = joblib.load(cp)
        else:
            m = newm(mn).fit(Xtr, ytr); joblib.dump(m, cp)
        p = m.predict_proba(Xva)[:, 1]
        rows.append({"target": T, "model": mn, **lib.metrics_binary(yva, p, nb)})
        log["val_evals"] += 1
A = pd.DataFrame(rows); A.to_csv(out / "a_conditions.csv", index=False)
pd.DataFrame(skipped).to_csv(out / "a_skipped.csv", index=False)
best = A.sort_values("auroc", ascending=False).groupby("target").head(1).sort_values("auroc", ascending=False)
promising = best[(best.n_pos >= thr["min_pos_val"]) & (best.auroc_lo >= thr["promising_target_auroc_lo"])]
prom_t = [t for t in promising.target if t != T0]
print("A:"); lib.show(A[["target", "model", "n", "n_pos", "auroc", "auroc_lo", "auroc_hi", "auprc"]].round(3))
print("skipped:", skipped); print("promettenti:", list(promising.target))
focus = [T0] + prom_t[:2]

# ---------------------------------------------------------------- (b)
def paired_diff(y, pl, pg, n=nb, seed=1):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        a, b = lib.safe_auc(y[i], pl[i]), lib.safe_auc(y[i], pg[i])
        if not (np.isnan(a) or np.isnan(b)): d.append(a - b)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

rowsb = []
strata_vars = ["qrs_bin", "sex", "age_bin", "location_setting", "race_ethnicity"]
G = {}
for T in focus:
    G[T] = joblib.load(out / f"model_{T}_hist_gb.joblib")
    Xtr, ytr, dtr = lib.get_xy(tr, T); Xva, yva, dva = lib.get_xy(va, T)
    pg_all = G[T].predict_proba(Xva)[:, 1]
    for sv in strata_vars:
        for lev in sorted(dva[sv].dropna().unique(), key=str):
            mv = (dva[sv] == lev).values; mt = (dtr[sv] == lev).values
            nv, pv = int(mv.sum()), int(yva[mv].sum())
            row = {"target": T, "strat": sv, "level": str(lev), "n_val": nv, "pos_val": pv, "n_train": int(mt.sum()), "pos_train": int(ytr[mt].sum())}
            if (nv < thr["min_n_val"] or pv < thr["min_pos_val"] or nv - pv < 25 or ytr[mt].sum() < MIN_TR
                    or (mt.sum() - ytr[mt].sum()) < MIN_TR):
                row["nota"] = "troppo piccolo"; rowsb.append(row); continue
            ml = newm("hist_gb").fit(Xtr[mt], ytr[mt]); pl = ml.predict_proba(Xva[mv])[:, 1]
            mg = lib.metrics_binary(yva[mv], pg_all[mv], nb); mlm = lib.metrics_binary(yva[mv], pl, nb)
            lo, hi = paired_diff(yva[mv], pl, pg_all[mv])
            row.update({"auroc_glob": mg["auroc"], "glob_lo": mg["auroc_lo"], "glob_hi": mg["auroc_hi"],
                        "auroc_loc": mlm["auroc"], "loc_lo": mlm["auroc_lo"], "loc_hi": mlm["auroc_hi"],
                        "diff_loc_minus_glob": mlm["auroc"] - mg["auroc"], "diff_lo": lo, "diff_hi": hi})
            rowsb.append(row); log["val_evals"] += 2
B = pd.DataFrame(rowsb); B.to_csv(out / "b_strata.csv", index=False)
print("B:"); lib.show(B[B.target == T0].drop(columns=["target"]).round(3))

# ---------------------------------------------------------------- (c)
def atoms(d0):
    at = []
    for f in ["age_at_ecg", "ventricular_rate", "atrial_rate", "pr_interval", "qrs_duration", "qt_corrected"]:
        for q in (0.25, 0.5, 0.75):
            c = float(d0[f].quantile(q))
            at.append((f, ">=", c)); at.append((f, "<", c))
    at += [("sex", "==", 1), ("sex", "==", 0)]
    return at
def mask(d, a):
    f, op, c = a
    v = d[f].values
    return (v >= c) if op == ">=" else (v < c) if op == "<" else (v == c)   # NaN -> False
def rtxt(r): return " AND ".join(f"{f} {op} {c:.4g}" for f, op, c in r)
AT = atoms(tr)
RULES = [(a,) for a in AT] + [(a, b) for a, b in itertools.combinations(AT, 2) if a[0] != b[0]]
print("regole per target:", len(RULES))
K = 5
rowsc = []; cred = []
for T in focus:
    Xtr, ytr, dtr = lib.get_xy(tr, T); Xva, yva, dva = lib.get_xy(va, T)
    oofp = out / f"oof_{T}.npy"
    oof = np.load(oofp) if oofp.exists() else lib.oof_proba(newm("hist_gb"), Xtr, ytr, dtr.patient_key.values)
    np.save(oofp, oof)
    g_oof = lib.safe_auc(ytr, oof)
    cand = []
    for r in RULES:
        m = np.ones(len(dtr), bool)
        for a in r: m &= mask(dtr, a)
        log["rules_tried"] += 1
        n, p = int(m.sum()), int(ytr[m].sum())
        if n < 500 or p < 50 or n - p < 50: continue
        cand.append((lib.safe_auc(ytr[m], oof[m]) - g_oof, r, n, p))
    cand.sort(key=lambda x: -x[0])
    print(T, "regole eleggibili in train:", len(cand), "AUROC OOF globale train", round(g_oof, 3))
    pg_all = G[T].predict_proba(Xva)[:, 1]
    gref = lib.metrics_binary(yva, pg_all, nb)["auroc"]
    for rank, (md, r, n_t, p_t) in enumerate(cand[:K]):
        mv = np.ones(len(dva), bool); mt = np.ones(len(dtr), bool)
        for a in r: mv &= mask(dva, a); mt &= mask(dtr, a)
        row = {"target": T, "rule": rtxt(r), "rank_train": rank + 1, "n_train": n_t, "pos_train": p_t, "train_oof_margin": md,
               "n_val": int(mv.sum()), "pos_val": int(yva[mv].sum()), "auroc_val_glob_ref": gref}
        if mv.sum() >= 20 and 0 < yva[mv].sum() < mv.sum():
            mg = lib.metrics_binary(yva[mv], pg_all[mv], nb)
            ml = newm("hist_gb").fit(Xtr[mt], ytr[mt]); pl = ml.predict_proba(Xva[mv])[:, 1]
            mlm = lib.metrics_binary(yva[mv], pl, nb); dlo, dhi = paired_diff(yva[mv], pl, pg_all[mv])
            row.update({"auroc_sub_glob_model": mg["auroc"], "sub_lo": mg["auroc_lo"], "sub_hi": mg["auroc_hi"],
                        "auroc_sub_loc_model": mlm["auroc"], "locm_lo": mlm["auroc_lo"], "locm_hi": mlm["auroc_hi"],
                        "diff_loc_glob": mlm["auroc"] - mg["auroc"], "diff_lo": dlo, "diff_hi": dhi,
                        "margin_vs_global": mg["auroc"] - gref})
            ok = (row["n_val"] >= thr["min_n_val"] and row["pos_val"] >= thr["min_pos_val"] and row["n_val"] - row["pos_val"] >= 25
                  and mg["auroc"] - gref >= thr["credible_subgroup_margin"] and mg["auroc_lo"] > gref)
            row["credibile"] = bool(ok); log["val_evals"] += 2
            if ok: cred.append((T, r, row))
        else: row["credibile"] = False
        rowsc.append(row)
C = pd.DataFrame(rowsc); C.to_csv(out / "c_rules_val.csv", index=False)
lib.show(C.round(3).drop(columns=["target"]))

if not debug:
    for i, (T, r, row) in enumerate(cred):
        name = f"sub_{T.split('_')[0]}_{i+1}"
        conds = [(f, op, float(c)) for f, op, c in r]
        src = ('import joblib\nfrom pathlib import Path\nimport numpy as np, pandas as pd\n_M = joblib.load(Path(__file__).parent / "model.joblib")\n'
               f'_R = {conds!r}\n'
               'def predict(df, lib):\n    X = df[lib.ECG_FEATURES].astype(float)\n    m = np.ones(len(df), bool)\n'
               '    for f, op, c in _R:\n        v = df[f].values\n        m &= (v >= c) if op == ">=" else (v < c) if op == "<" else (v == c)\n'
               '    return pd.DataFrame({"score": _M.predict_proba(X)[:, 1], "in_scope": m}, index=df.index)\n')
        vm = {k: float(row[k]) for k in ["n_val", "pos_val", "auroc_sub_glob_model", "sub_lo", "sub_hi", "auroc_val_glob_ref"]}
        lib.save_candidate(ROOT / f"results/P2/candidates/{name}", name, T, "subgroup",
                           f"Modello globale hist_gb ristretto alla regola: {rtxt(r)}", vm, src, {"model.joblib": G[T]})
        print("candidato salvato:", name, rtxt(r))
summ = {"focus": focus, "promising": list(promising.target), "skipped": skipped, "rules_per_target": len(RULES),
        "rules_tried_total": log["rules_tried"], "val_evals": log["val_evals"], "val_rule_rows": int(len(C)),
        "credible": [(T, rtxt(r)) for T, r, _ in cred]}
(out / "summary.json").write_text(json.dumps(summ, indent=2, default=float)); print(json.dumps(summ, indent=1, default=float))
