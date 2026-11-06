"""Q4 parti 2 e 3 (versione veloce): strati pre-specificati con bootstrap per paziente condiviso + regole sull'errore con cross-fitting.
(La parte 1, cluster, e' in emb.py; la parte 2 di emb.py era troppo lenta ed e' sostituita da questo script.)"""
import sys, json, zlib, os
sys.path.insert(0, '.'); sys.path.insert(0, 'lib')
import numpy as np, pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeRegressor
from lib import pipe2_lib as P, echonext_lib as L
from lib.pipe2_lib import _cluster_index, _boot_idx
D = 'results/Q1/main'; OUT = 'results/Q4'; NB = 200
data, dev = P.load_dev()
T = P.TARGETS; iS, iL = 0, T.index('lvef_lte_45_flag')
auc = L.safe_auc
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
order, starts, counts = _cluster_index(g)
srows = []
for tn, ti in (('SHD', iS), ('LVEF<=45', iL)):
    y = d[T[ti]].values.astype(int); p = e[:, ti]; gl = auc(y, p)
    lv_list = []
    for fam, s in S.items():
        for lv in sorted(s.unique()):
            m = (s == lv).values
            if m.sum() < 100 or y[m].sum() < 25 or (m.sum() - y[m].sum()) < 25: continue
            lv_list.append((fam, lv, m))
    base = {(f_, l_): (auc(y[m], p[m]), auc(y[~m], p[~m])) for f_, l_, m in lv_list}
    bs = {(f_, l_): ([], []) for f_, l_, m in lv_list}
    rng = np.random.default_rng(7)
    for b in range(NB):
        i = _boot_idx(order, starts, counts, rng); yi, pi = y[i], p[i]
        for f_, l_, m in lv_list:
            mi = m[i]
            if len(np.unique(yi[mi])) < 2 or len(np.unique(yi[~mi])) < 2: continue
            a = auc(yi[mi], pi[mi]); bs[(f_, l_)][0].append(a); bs[(f_, l_)][1].append(a - auc(yi[~mi], pi[~mi]))
    for f_, l_, m in lv_list:
        a, r_ = base[(f_, l_)]; va, vd = np.array(bs[(f_, l_)][0]), np.array(bs[(f_, l_)][1]); dif = a - r_
        sd = vd.std(); pv = float(2 * stats.norm.sf(abs(dif / sd)))
        srows.append(dict(target=tn, famiglia=f_, strato=l_, n=int(m.sum()), pos=int(y[m].sum()), auc=a,
                          lo=float(np.percentile(va, 2.5)), hi=float(np.percentile(va, 97.5)), diff_resto=dif,
                          d_lo=float(np.percentile(vd, 2.5)), d_hi=float(np.percentile(vd, 97.5)), p=pv, auc_glob=gl))
    print(tn, 'fatto', flush=True)
st = pd.DataFrame(srows); hh = P.holm({str(i): r.p for i, r in st.iterrows()})
st['p_holm'] = [hh[str(i)]['p_adj'] for i in st.index]; st['sig'] = st.p_holm < 0.05
st.to_csv(f'{OUT}/strata.csv', index=False)
print(len(st), 'strati'); print(st.round(3).to_string())

# ---------- regole sull'errore (cross-fitting per paziente)
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
        o2, s2, c2 = _cluster_index(gB)
        for lf in np.unique(lA):
            nrule += 1
            mA, mB = lA == lf, lB == lf
            marA = eA[mA].mean() - eA[~mA].mean()
            rng = np.random.default_rng(int(lf)); v = []
            for _ in range(300):
                i = _boot_idx(o2, s2, c2, rng); mi = mB[i]; ei = eB[i]
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
json.dump(dict(n_strati=len(st), n_regole=nrule, n_cluster=int(len(pd.read_csv(f'{OUT}/clusters.csv')))), open(f'{OUT}/counts.json', 'w'))
