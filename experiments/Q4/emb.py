import sys, json, zlib
sys.path.insert(0, '.'); sys.path.insert(0, 'lib')
import numpy as np, pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.tree import DecisionTreeRegressor
from lib import pipe2_lib as P, echonext_lib as L
from lib.pipe2_lib import _cluster_index, _boot_idx
import os
D = 'results/Q1/main'; OUT = 'results/Q4'
os.makedirs(OUT, exist_ok=True)
rng0 = np.random.default_rng(42)
data, dev = P.load_dev()
T = P.TARGETS; iS, iL = 0, T.index('lvef_lte_45_flag')
assert T[0] == P.MAIN
auc = L.safe_auc


def boot_diff(y, p, grp, mask, nb=400, seed=0):
    """bootstrap per paziente (appaiato): AUROC(mask) - AUROC(globale) e - AUROC(resto)"""
    order, starts, counts = _cluster_index(grp); rng = np.random.default_rng(seed)
    res = {'g': [], 'r': []}
    for _ in range(nb):
        i = _boot_idx(order, starts, counts, rng)
        yi, pi, mi = y[i], p[i], mask[i]
        if mi.sum() < 20 or len(np.unique(yi[mi])) < 2: continue
        a = auc(yi[mi], pi[mi]); res['g'].append(a - auc(yi, pi))
        if (~mi).sum() > 20 and len(np.unique(yi[~mi])) == 2: res['r'].append(a - auc(yi[~mi], pi[~mi]))
    return res


def ci(v):
    v = np.array(v); return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float(v.std())


# ---------- 1. cluster sugli embedding
if os.path.exists(f'{OUT}/clusters.csv') and os.path.exists(f'{OUT}/cluster_summary.json'):
    cl = pd.read_csv(f'{OUT}/clusters.csv'); print('cluster: gia fatti, salto')
else:
    rows = []; summ = {}
    for f in range(5):
        z = np.load(f'{D}/oof_seed0_fold{f}.npz'); idx = z['idx']
        clean = dev.lvef_known.values[idx]
        idx = idx[clean]; E = z['emb'][clean]; pr = z['preds'][clean][:, 0]
        sub = dev.iloc[idx]; y = sub[P.MAIN].values.astype(int); g = sub.patient_key.values
        Xs = (E - E.mean(0)) / (E.std(0) + 1e-6); Xp = PCA(20, random_state=42).fit_transform(Xs)
        sil = {}; sam = rng0.choice(len(Xp), 5000, replace=False)
        for k in (3, 4, 6, 8):
            lab = KMeans(k, n_init=5, random_state=42).fit_predict(Xp); sil[k] = float(silhouette_score(Xp[sam], lab[sam]))
        k = max(sil, key=sil.get); lab = KMeans(k, n_init=5, random_state=42).fit(Xp).labels_
        half = np.array([zlib.crc32(f'h{x}'.encode()) % 2 for x in g]) == 1
        m1 = KMeans(k, n_init=5, random_state=1).fit(Xp[half]); m2 = KMeans(k, n_init=5, random_state=2).fit(Xp[~half])
        ari = float(adjusted_rand_score(m1.predict(Xp), m2.predict(Xp)))
        gl = auc(y, pr); gl_ci = P.auc_ci_cluster(y, pr, g, 300)
        ok = [c for c in range(k) if (lab == c).sum() >= 100 and 25 <= y[lab == c].sum() <= (lab == c).sum() - 25]
        mx = []
        for _ in range(200):
            pl = rng0.permutation(lab); dd = []
            for c in ok:
                m = pl == c
                if len(np.unique(y[m])) == 2: dd.append(auc(y[m], pr[m]) - gl)
            mx.append(max(dd))
        mx = np.array(mx)
        for c in range(k):
            m = lab == c; n = int(m.sum()); npos = int(y[m].sum())
            r = dict(fold=f, k=k, cluster=c, n=n, pos=npos, neg=n - npos, prev=npos / n,
                     age=float(sub.age_at_ecg[m].mean()), male=float((sub.sex[m].astype(str) == 'Male').mean()),
                     qrs=float(sub.qrs_duration[m].mean()), rate=float(sub.ventricular_rate[m].mean()),
                     eligible=c in ok, auc_glob=gl)
            if c in ok:
                al, lo, hi = P.auc_ci_cluster(y[m], pr[m], g[m], 300)
                bd = boot_diff(y, pr, g, m, 300, c); dl, dh, _ = ci(bd['g'])
                r.update(auc_loc=al, auc_lo=lo, auc_hi=hi, diff=al - gl, diff_lo=dl, diff_hi=dh,
                         p_emp=float((1 + (mx >= al - gl).sum()) / 201))
                r['cred_pre'] = bool(al - gl >= 0.02 and lo > gl)
            rows.append(r)
        summ[f] = dict(k=k, sil=sil, ari=ari, auc_glob=gl, glob_ci=[gl_ci[1], gl_ci[2]], perm_max_q95=float(np.quantile(mx, .95)),
                       perm_max_mean=float(mx.mean()), n=len(y))
        print(f, summ[f], flush=True)
    cl = pd.DataFrame(rows)
    best = cl[cl.eligible].sort_values('p_emp').groupby('fold').head(1)
    h = P.holm({f'f{r.fold}': r.p_emp for r in best.itertuples()})
    cl['p_holm_fold'] = np.nan
    for r in best.itertuples():
        cl.loc[(cl.fold == r.fold) & (cl.cluster == r.cluster), 'p_holm_fold'] = h[f'f{r.fold}']['p_adj']
    cl['cred_pre'] = cl['cred_pre'].fillna(False).astype(bool)
    cl['credibile'] = cl.cred_pre & (cl.p_holm_fold < 0.05)
    cl.to_csv(f'{OUT}/clusters.csv', index=False); json.dump(summ, open(f'{OUT}/cluster_summary.json', 'w'), indent=1)

# ---------- 2. strati
ens = np.load(f'{D}/oof_ensemble.npy'); cleanm = dev.lvef_known.values & ~np.isnan(ens[:, 0])
d = dev[cleanm].reset_index(drop=True); e = ens[cleanm]; g = d.patient_key.values


def terz(c): return pd.qcut(d[c], 3, labels=['T1', 'T2', 'T3'])


S = {}
S['eta'] = pd.cut(d.age_at_ecg, [0, 50, 65, 75, 200], right=False, labels=['<50', '50-64', '65-74', '>=75'])
S['sesso'] = d.sex
S['QRS'] = pd.cut(d.qrs_duration, [0, 100, 120, 1000], right=False, labels=['<100', '100-119', '>=120'])
for c, n in (('ventricular_rate', 'freq'), ('pr_interval', 'PR'), ('qt_corrected', 'QTc')): S[n] = terz(c)
S['anno'] = d.acquisition_year; S['setting'] = d.location_setting
S['razza_etnia'] = d.race_ethnicity; S['ecg_piu_recente'] = d.most_recent_ecg
S = {k: v.astype(object).where(v.notna(), 'NA').astype(str) for k, v in S.items()}
srows = []
for tn, ti in (('SHD', iS), ('LVEF<=45', iL)):
    y = d[T[ti]].values.astype(int); p = e[:, ti]; gl = auc(y, p)
    for fam, s in S.items():
        for lv in sorted(s.unique()):
            m = (s == lv).values
            if m.sum() < 100 or y[m].sum() < 25 or (m.sum() - y[m].sum()) < 25: continue
            a, lo, hi = P.auc_ci_cluster(y[m], p[m], g[m], 300)
            bd = boot_diff(y, p, g, m, 400, 7); dl, dh, sd = ci(bd['r']); dif = a - auc(y[~m], p[~m])
            pv = float(2 * stats.norm.sf(abs(dif / sd)))
            srows.append(dict(target=tn, famiglia=fam, strato=lv, n=int(m.sum()), pos=int(y[m].sum()), auc=a, lo=lo, hi=hi,
                              diff_resto=dif, d_lo=dl, d_hi=dh, p=pv, auc_glob=gl))
st = pd.DataFrame(srows); hh = P.holm({str(i): r.p for i, r in st.iterrows()})
st['p_holm'] = [hh[str(i)]['p_adj'] for i in st.index]; st['sig'] = st.p_holm < 0.05
st.to_csv(f'{OUT}/strata.csv', index=False)
print(len(st), 'strati'); print(st.round(3).to_string())

# ---------- 3. regole sull'errore (cross-fitting per paziente)
sx = d.sex.astype(str)
X = pd.DataFrame({'eta': d.age_at_ecg, 'sesso_M': (sx == 'Male').astype(int), 'freq': d.ventricular_rate, 'atriale': d.atrial_rate,
                  'PR': d.pr_interval, 'QRS': d.qrs_duration, 'QTc': d.qt_corrected, 'anno': d.acquisition_year})
for c in ('location_setting', 'race_ethnicity'): X = X.join(pd.get_dummies(d[c].astype(str), prefix=c[:4]).astype(int))
X = X.fillna(X.median()); half = np.array([zlib.crc32(f'cf{x}'.encode()) % 2 for x in g])
rrows = []; nrule = 0
for tn, ti in (('SHD', iS), ('LVEF<=45', iL)):
    y = d[T[ti]].values.astype(int); p = e[:, ti]; err = np.abs(y - p)
    for tr in (0, 1):
        A, B = half == tr, half != tr
        tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=300, random_state=42).fit(X[A], err[A])
        XA, XB = X[A], X[B]; lA, lB = tree.apply(XA), tree.apply(XB)
        eA, eB = err[A], err[B]; gB, yB, pB = g[B], y[B], p[B]
        order, starts, counts = _cluster_index(gB)
        for lf in np.unique(lA):
            nrule += 1
            mA, mB = lA == lf, lB == lf
            marA = eA[mA].mean() - eA[~mA].mean()
            rng = np.random.default_rng(int(lf)); v = []
            for _ in range(300):
                i = _boot_idx(order, starts, counts, rng); mi = mB[i]; ei = eB[i]
                if mi.sum() > 20: v.append(ei[mi].mean() - ei[~mi].mean())
            marB = eB[mB].mean() - eB[~mB].mean(); lo, hi = np.percentile(v, [2.5, 97.5]); sd = np.std(v)
            pv = float(2 * stats.norm.sf(abs(marB / sd)))
            aucin = auc(yB[mB], pB[mB]) if len(np.unique(yB[mB])) == 2 else np.nan
            node_path = list(tree.decision_path(XA.iloc[[np.where(mA)[0][0]]]).indices); desc = []
            for a_, b_ in zip(node_path[:-1], node_path[1:]):
                fe, th = tree.tree_.feature[a_], tree.tree_.threshold[a_]
                desc.append(f"{X.columns[fe]} {'<=' if b_ == tree.tree_.children_left[a_] else '>'} {th:.1f}")
            rrows.append(dict(target=tn, tr_half=tr, regola=' & '.join(desc), nA=int(mA.sum()), nB=int(mB.sum()), margine_A=marA,
                              margine_B=marB, lo=float(lo), hi=float(hi), p=pv, auc_in=aucin,
                              auc_out=auc(yB[~mB], pB[~mB]), err_in=float(eB[mB].mean()), err_out=float(eB[~mB].mean())))
rl = pd.DataFrame(rrows); hr = P.holm({str(i): r.p for i, r in rl.iterrows()})
rl['p_holm'] = [hr[str(i)]['p_adj'] for i in rl.index]
rl['regge'] = (rl.margine_A > 0) & (rl.margine_B > 0) & (rl.lo > 0) & (rl.p_holm < 0.05)
rl.to_csv(f'{OUT}/error_rules.csv', index=False); print(nrule, 'regole'); print(rl.round(3).to_string())
json.dump(dict(n_strati=len(st), n_regole=nrule, n_cluster=int(len(cl))), open(f'{OUT}/counts.json', 'w'))
