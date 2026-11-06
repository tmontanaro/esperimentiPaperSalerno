"""Q3: il QRS aggiunge informazione sopra la CNN per LVEF? Solo aggregati. Pool dev (train+val), niente test.
Uso: python3 experiments/Q3/qrs_cnn.py [debug]   (debug = solo verifica caricamento)"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, log_loss, r2_score
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))
import pipe2_lib as P
DEBUG = len(sys.argv) > 1 and sys.argv[1] == "debug"
OUT = ROOT / "results/Q3"; OUT.mkdir(exist_ok=True)
B = 500
data, dev = P.load_dev()
ens = np.load(ROOT / "results/Q1/main/oof_ensemble.npy")
tab = np.load(ROOT / "results/Q1/main/tabular_oof.npz")["P"]
print("dev", len(dev), "ens", ens.shape, "tab", tab.shape)
assert len(ens) == len(dev) == len(tab)
clean = dev.lvef_known.values
d = dev[clean].reset_index(drop=True); E = ens[clean]; T = tab[clean]
j = P.TARGETS.index("lvef_lte_45_flag")
y = (d.lvef_value.values <= 45).astype(int)
assert (y == d.lvef_lte_45_flag.values.astype(int)).all()
if DEBUG:
    print("debug OK: clean", len(d), "pos", y.sum(), "NaN pc", np.isnan(E[:, j]).sum(), "NaN pt", np.isnan(T[:, j]).sum(),
          "NaN qrs", d.qrs_duration.isna().sum(), "NaN lvef_cnn", np.isnan(E[:, 12]).sum()); sys.exit()
pc, pt = np.clip(E[:, j], 1e-6, 1-1e-6), np.clip(T[:, j], 1e-6, 1-1e-6)
lc = np.log(pc/(1-pc))
ml, tl = E[:, 12], d.lvef_value.values
g = d.patient_key.values; fold = d.fold.values
qrs = d.qrs_duration.values.astype(float)
ok = ~np.isnan(qrs) & ~np.isnan(pt) & ~np.isnan(pc) & ~np.isnan(ml)
print("clean", len(d), "pos", y.sum(), "qrs ok", ok.sum())
lo, hi = np.nanpercentile(qrs, [1, 99]); qz = (np.clip(qrs, lo, hi) - np.nanmean(qrs)) / np.nanstd(qrs)
order, starts, counts = P._cluster_index(g)


def boot(fn, mask=None, B=B):
    """bootstrap per paziente di fn(idx) -> vettore di statistiche; ritorna lista (base, lo, hi, p)"""
    base = np.atleast_1d(fn(np.where(mask)[0] if mask is not None else np.arange(len(d))))
    r = np.random.default_rng(0); v = []
    for _ in range(B):
        i = P._boot_idx(order, starts, counts, r)
        if mask is not None:
            i = i[mask[i]]
        try:
            v.append(np.atleast_1d(fn(i)))
        except Exception:
            pass
    v = np.array(v)
    res = []
    for k in range(len(base)):
        a = v[:, k]; a = a[~np.isnan(a)]
        p = max(2*min((a <= 0).mean(), (a >= 0).mean()), 1/(len(a)+1))
        res.append((float(base[k]), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), float(p)))
    return res


R = {"n_clean": int(len(d)), "n_pos": int(y.sum()), "n_con_qrs": int(ok.sum())}
# ---------- a) per fascia di QRS
band = pd.cut(qrs, [0, 100, 120, 1e9], labels=["<100", "100-119", ">=120"], right=False)
A = {}; pa = {}
for b in ["<100", "100-119", ">=120"]:
    m = np.asarray((band == b) & ok)
    yy = y[m]
    c = P.auc_ci_cluster(yy, pc[m], g[m], n_boot=B); t = P.auc_ci_cluster(yy, pt[m], g[m], n_boot=B)
    dd = boot(lambda i: roc_auc_score(y[i], pc[i]) - roc_auc_score(y[i], pt[i]), m)[0]
    A[b] = dict(n=int(m.sum()), pos=int(yy.sum()), cnn=c, tab=t, diff=dd[:3], p=dd[3]); pa["a_"+b] = dd[3]
R["a"] = A; R["a_holm"] = P.holm(pa)
H = {}
m0 = np.asarray((band == "<100") & ok)
for b in ["100-119", ">=120"]:
    m1 = np.asarray((band == b) & ok)
    def f(i, m1=m1, m0=m0):
        a = i[m1[i]]; c = i[m0[i]]
        return roc_auc_score(y[a], pc[a]) - roc_auc_score(y[c], pc[c])
    H[b + " vs <100 (CNN)"] = boot(f, m1 | m0)[0]
R["a_eterogeneita_cnn"] = H
# ---------- b) valore aggiunto del QRS sopra la CNN (CV per paziente, stesse fold)
def cv_logit(X):
    o = np.full(len(d), np.nan)
    for f in range(5):
        tr, te = (fold != f) & ok, (fold == f) & ok
        m = LogisticRegression(C=1e4, max_iter=1000).fit(X[tr], y[tr]); o[te] = m.predict_proba(X[te])[:, 1]
    return o
o0 = cv_logit(lc[:, None]); o1 = cv_logit(np.nan_to_num(np.c_[lc, qz, qz*lc]))
def fb(i):
    return [roc_auc_score(y[i], o1[i]) - roc_auc_score(y[i], o0[i]),
            log_loss(y[i], o0[i], labels=[0, 1]) - log_loss(y[i], o1[i], labels=[0, 1])]
bb = boot(fb, ok)
R["b"] = {"auc_base": float(roc_auc_score(y[ok], o0[ok])), "auc_full": float(roc_auc_score(y[ok], o1[ok])),
          "logloss_base": float(log_loss(y[ok], o0[ok])), "logloss_full": float(log_loss(y[ok], o1[ok])),
          "delta_auc": bb[0], "delta_logloss_base_meno_full": bb[1]}
R["b_holm"] = P.holm({"delta_auc": bb[0][3], "delta_logloss": bb[1][3]})
# ---------- c) residuo della LVEF
res = tl - ml
cov = d[["age_at_ecg", "sex", "ventricular_rate", "pr_interval", "qt_corrected"]].copy()
cov["sex"] = pd.factorize(cov["sex"])[0]
okc = ok & ~cov.isna().any(axis=1).values
covv = cov.values.astype(float)
def partial(i):
    i = i[okc[i]]
    rr = rankdata(res[i]); rq = rankdata(qrs[i]); C = np.column_stack([rankdata(covv[i, k]) for k in range(covv.shape[1])])
    a = rr - LinearRegression().fit(C, rr).predict(C); b2 = rq - LinearRegression().fit(C, rq).predict(C)
    return np.corrcoef(a, b2)[0, 1]
def cv_lin(X):
    o = np.full(len(d), np.nan)
    for f in range(5):
        tr, te = (fold != f) & ok, (fold == f) & ok
        o[te] = LinearRegression().fit(X[tr], tl[tr]).predict(X[te])
    return o
q0 = cv_lin(ml[:, None]); q1 = cv_lin(np.nan_to_num(np.c_[ml, qz, qz*ml]))
ci_sp = boot(lambda i: spearmanr(res[i], qrs[i])[0], ok)[0]
ci_pa = boot(partial, okc)[0]
ci_r2 = boot(lambda i: r2_score(tl[i], q1[i]) - r2_score(tl[i], q0[i]), ok)[0]
R["c"] = {"spearman_residuo_qrs": ci_sp, "correlazione_parziale": ci_pa, "r2_base": float(r2_score(tl[ok], q0[ok])),
          "r2_full": float(r2_score(tl[ok], q1[ok])), "delta_r2": ci_r2, "mae_cnn": float(np.abs(res[ok]).mean()),
          "residuo_medio": float(res[ok].mean()), "residuo_sd": float(res[ok].std()), "n_parziale": int(okc.sum())}
R["c_holm"] = P.holm({"corr_parziale": ci_pa[3], "delta_r2": ci_r2[3]})
# ---------- d) coppie discordanti (un ECG per paziente, scelto a caso con seed 42)
perm = np.random.default_rng(42).permutation(len(d))
seen = {}
for k in perm:
    if ok[k] and g[k] not in seen:
        seen[g[k]] = k
idx = np.array(list(seen.values()))
srt = idx[np.argsort(ml[idx])]; mv = ml[srt]
used = np.zeros(len(srt), bool); pairs = []
for a in np.random.default_rng(7).permutation(len(srt)):
    if used[a]:
        continue
    l, r = np.searchsorted(mv, [mv[a]-1, mv[a]+1])
    c = np.arange(l, r)
    c = c[(~used[c]) & (c != a) & (np.abs(tl[srt[c]] - tl[srt[a]]) >= 10) & (np.abs(mv[c]-mv[a]) < 1)]
    if len(c):
        b2 = c[np.random.default_rng(int(a)).integers(len(c))]
        used[a] = used[b2] = True; pairs.append((srt[a], srt[b2]))
pairs = np.array(pairs)
sw = tl[pairs[:, 0]] < tl[pairs[:, 1]]
lowi = np.where(sw, pairs[:, 0], pairs[:, 1]); highi = np.where(sw, pairs[:, 1], pairs[:, 0])
dq = qrs[lowi] - qrs[highi]; dq = dq[~np.isnan(dq)]
r = np.random.default_rng(0); bs = np.array([dq[r.integers(0, len(dq), len(dq))].mean() for _ in range(2000)])
p_d = max(2*min((bs <= 0).mean(), (bs >= 0).mean()), 1/2001)
R["d"] = {"n_pazienti_unici": int(len(idx)), "n_coppie": int(len(dq)),
          "diff_qrs_media_(LVEF_bassa - alta)": [float(dq.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
          "p": float(p_d), "diff_lvef_vera_media": float(np.mean(np.abs(tl[pairs[:, 0]]-tl[pairs[:, 1]]))),
          "diff_lvef_cnn_media": float(np.mean(np.abs(ml[pairs[:, 0]]-ml[pairs[:, 1]])))}
R["d_holm"] = P.holm({"d": p_d})
R["n_test_totali"] = 8
json.dump(R, open(OUT / "q3_results.json", "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
