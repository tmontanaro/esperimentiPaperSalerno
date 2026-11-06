"""P5 passi 3-5: profilo cluster, chi-quadro, modello globale vs locali sugli stessi pazienti di validation,
selezione all'indietro dentro la CV raggruppata del train del cluster, soglia di rumore con etichette mescolate.
Uso: python3 local_vs_global.py [debug] [nperm]"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, joblib
from joblib import Parallel, delayed
from scipy.stats import chi2_contingency
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from lib import echonext_lib as lib

debug = "debug" in sys.argv
args = [a for a in sys.argv[1:] if a != "debug"]
NPERM = int(args[0]) if args else (3 if debug else 30)
out = ROOT / "results" / "P5" / ("debug" if debug else ""); out.mkdir(parents=True, exist_ok=True)
cfg = json.loads((ROOT / "agentic/pipeline_state.json").read_text())["config"]
thr = cfg["thresholds"]; TARGETS = cfg["cluster_targets"]
MIN_TR, MIN_VA = (10, 3) if debug else (60, 20)
data = lib.load_data(debug_n=2000 if debug else None)
tr, va = data.train.copy(), data.val.copy()
tr["cl"] = np.load(out / "train_cluster.npy"); va["cl"] = np.load(out / "val_cluster.npy")
F = lib.ECG_FEATURES; K = int(tr.cl.max() + 1)
zoo = lib.model_zoo(); NB = 100 if debug else 300
sel = json.loads((out / "selection_summary.json").read_text())

# ---------------------------------------------------------------- profilo e chi-quadro
prof = []
pool = tr.sample(frac=1, random_state=42).drop_duplicates("patient_key")
for c in range(K):
    m = tr.cl == c; mv = va.cl == c; mp = pool.cl == c
    r = {"cluster": c, "n_train_ecg": int(m.sum()), "n_pool_pazienti": int(mp.sum()), "n_val": int(mv.sum())}
    for f in F: r[f"{f}_mean_pool"] = float(pool.loc[mp, f].mean())
    for T in TARGETS:
        r[f"prev_train_{T[:12]}"] = float(tr.loc[m, T].mean()); r[f"prev_val_{T[:12]}"] = float(va.loc[mv, T].mean())
    prof.append(r)
P = pd.DataFrame(prof); P.to_csv(out / "cluster_profile.csv", index=False)
print("Profilo cluster:"); lib.show(P.round(3))
chi = []
for T in TARGETS:
    for nm, d in (("pool_train_1ECG_per_paziente", pool), ("validation", va)):
        d = d.dropna(subset=[T]); ct = pd.crosstab(d.cl, d[T])
        c2, p, dof, _ = chi2_contingency(ct)
        chi.append({"target": T, "set": nm, "chi2": float(c2), "dof": int(dof), "p": float(p), "n": int(len(d)),
                    "prev_min": float(d.groupby("cl")[T].mean().min()), "prev_max": float(d.groupby("cl")[T].mean().max())})
pd.DataFrame(chi).to_csv(out / "chi2.csv", index=False)
print("chi-quadro:"); lib.show(pd.DataFrame(chi).round(4))

# ---------------------------------------------------------------- procedura locale vs globale
def backward_select(Xc, yc, gc, min_feat=2, tol=0.002):
    """Eliminazione all'indietro con CV raggruppata (3 fold) SOLO sul train del cluster; criterio AUROC OOF della logreg."""
    cv = GroupKFold(3); base = zoo["logreg"]
    def cvauc(cols):
        p = np.zeros(len(yc))
        for a, b in cv.split(Xc, yc, gc):
            if len(np.unique(yc[a])) < 2: return 0.5
            p[b] = clone(base).fit(Xc.iloc[a][cols], yc[a]).predict_proba(Xc.iloc[b][cols])[:, 1]
        return lib.safe_auc(yc, p)
    cur = list(Xc.columns); best = cvauc(cur)
    while len(cur) > min_feat:
        cand = [(cvauc([f for f in cur if f != j]), j) for j in cur]
        s, j = max(cand)
        if s >= best - tol: cur.remove(j); best = max(best, s) if s > best else best
        else: break
    return cur

def procedure(ytr, yva, dtr, dva, glob_model=None, keep_models=False):
    """ritorna: predizioni val per modello (globale e locali) per ogni cluster eleggibile"""
    Xtr, Xva = dtr[F].astype(float), dva[F].astype(float)
    g = clone(zoo["hist_gb"]) if glob_model is None else glob_model
    if glob_model is None: g.fit(Xtr, ytr)
    pg = g.predict_proba(Xva)[:, 1]
    res = {"global_all": pg}; per = {}
    for c in range(K):
        mt = (dtr.cl == c).values; mv = (dva.cl == c).values
        if ytr[mt].sum() < MIN_TR or yva[mv].sum() < MIN_VA or (mv.sum() - yva[mv].sum()) < MIN_VA: continue
        d = {"mv": mv, "pg": pg[mv]}
        fs = backward_select(Xtr[mt], ytr[mt], dtr.patient_key.values[mt])
        for nm in ("logreg", "lasso_logreg", "hist_gb"):
            d[nm] = clone(zoo[nm]).fit(Xtr[mt], ytr[mt]).predict_proba(Xva[mv])[:, 1]
        d["local_sel_logreg"] = clone(zoo["logreg"]).fit(Xtr[mt][fs], ytr[mt]).predict_proba(Xva[mv][fs])[:, 1]
        d["sel_features"] = fs; per[c] = d
    res["clusters"] = per
    return res, g

MODELS = ["logreg", "lasso_logreg", "hist_gb", "local_sel_logreg"]
def stat(res, yva):
    """max AUROC locale e max (locale - globale) sugli stessi pazienti"""
    mx, mg = -1, -1
    for c, d in res["clusters"].items():
        y = yva[d["mv"]]; ag = lib.safe_auc(y, d["pg"])
        for m in MODELS:
            a = lib.safe_auc(y, d[m]); mx = max(mx, a); mg = max(mg, a - ag)
    return mx, mg

def paired_diff(y, pl, pg, n=NB, seed=1):
    rg = np.random.default_rng(seed); dd = []
    for _ in range(n):
        i = rg.integers(0, len(y), len(y)); a, b = lib.safe_auc(y[i], pl[i]), lib.safe_auc(y[i], pg[i])
        if not (np.isnan(a) or np.isnan(b)): dd.append(a - b)
    return float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))

rows, perm_rows, sel_rows = [], [], []
for T in TARGETS:
    dtr, dva = tr.dropna(subset=[T]), va.dropna(subset=[T])
    ytr, yva = dtr[T].values.astype(int), dva[T].values.astype(int)
    cache = out / f"result_{T}.joblib"
    gm = out / f"global_{T}.joblib"
    if cache.exists():
        res, yv = joblib.load(cache)
    else:
        t0 = time.time(); g = joblib.load(gm) if gm.exists() else None
        res, g = procedure(ytr, yva, dtr, dva, g)
        if not gm.exists(): joblib.dump(g, gm)
        joblib.dump((res, yva), cache); print(T, "procedura:", round(time.time() - t0), "s")
    ga = lib.metrics_binary(yva, res["global_all"], NB)
    rows.append({"target": T, "cluster": "tutti(globale)", "model": "global_hist_gb", **ga, "diff_vs_global_same_pts": 0.0})
    for c, d in res["clusters"].items():
        y = yva[d["mv"]]; mg = lib.metrics_binary(y, d["pg"], NB)
        rows.append({"target": T, "cluster": c, "model": "global_hist_gb(stessi pazienti)", **mg, "diff_vs_global_same_pts": 0.0,
                     "global_full_val_auroc": ga["auroc"]})
        for m in MODELS:
            ml = lib.metrics_binary(y, d[m], NB); lo, hi = paired_diff(y, d[m], d["pg"])
            rows.append({"target": T, "cluster": c, "model": m, **ml, "diff_vs_global_same_pts": ml["auroc"] - mg["auroc"],
                         "diff_lo": lo, "diff_hi": hi, "global_full_val_auroc": ga["auroc"], "global_full_val_lo": ga["auroc_lo"]})
        sel_rows.append({"target": T, "cluster": c, "selected_features": "|".join(d["sel_features"])})
    # soglia di rumore
    obs_max, obs_gain = stat(res, yva)
    pf = out / f"perm_{T}.json"
    if pf.exists(): pr = json.loads(pf.read_text())
    else:
        def one(seed):
            from threadpoolctl import threadpool_limits
            with threadpool_limits(1):
                r = np.random.default_rng(1000 + seed)
                yt, yv_ = r.permutation(ytr), r.permutation(yva)
                rs, _ = procedure(yt, yv_, dtr, dva)
                # stessa eleggibilita' dei cluster osservata
                rs["clusters"] = {c: v for c, v in rs["clusters"].items() if c in res["clusters"]}
                return stat(rs, yv_)
        t0 = time.time()
        nj = 2 if debug else 5
        pr_ = Parallel(n_jobs=nj)(delayed(one)(s) for s in range(NPERM))
        pr = {"null_max_auc": [a for a, b in pr_], "null_max_gain": [b for a, b in pr_]}
        pf.write_text(json.dumps(pr)); print(T, "permutazioni:", NPERM, round(time.time() - t0), "s")
    nm, ng = np.array(pr["null_max_auc"]), np.array(pr["null_max_gain"])
    perm_rows.append({"target": T, "n_perm": len(nm), "obs_max_auroc": obs_max, "null_max_auroc_mean": nm.mean(), "null_max_auroc_p95": np.percentile(nm, 95),
                      "null_max_auroc_max": nm.max(), "p_emp_max_auroc": (1 + (nm >= obs_max).sum()) / (1 + len(nm)),
                      "obs_max_gain": obs_gain, "null_max_gain_mean": ng.mean(), "null_max_gain_p95": np.percentile(ng, 95),
                      "p_emp_max_gain": (1 + (ng >= obs_gain).sum()) / (1 + len(ng))})
R = pd.DataFrame(rows); R.to_csv(out / "local_vs_global.csv", index=False)
pd.DataFrame(perm_rows).to_csv(out / "permutation.csv", index=False)
pd.DataFrame(sel_rows).to_csv(out / "local_selected_features.csv", index=False)
print("Locale vs globale (validation):")
lib.show(R[["target", "cluster", "model", "n", "n_pos", "auroc", "auroc_lo", "auroc_hi", "diff_vs_global_same_pts"]].round(3), max_rows=200)
print("Permutazioni:"); lib.show(pd.DataFrame(perm_rows).round(3))
print("Selezione locale:"); lib.show(pd.DataFrame(sel_rows))
