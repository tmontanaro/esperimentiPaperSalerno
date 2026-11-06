"""P3 lvef-qrs. Uso: python3 lvef_qrs.py [debug]
(a) regressione LVEF (4 modelli + baseline) e classificazione lvef_lte_45 (5 modelli);
(b-i) modello per fascia di QRS vs globale (val, bootstrap); (b-ii) interazione qrs x feature sul train (F/Wald clusterizzato + CV raggruppata);
(b-iii/c) coppie di pazienti simili con LVEF diverso, 4 insiemi MATCH_ON (train = 1 ECG a paziente; replica su val)."""
import sys, json, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, joblib
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from lib import echonext_lib as lib
warnings.filterwarnings("ignore")

debug = len(sys.argv) > 1 and sys.argv[1] == "debug"
out = ROOT / "results" / "P3" / ("debug" if debug else "")
out.mkdir(parents=True, exist_ok=True)
SEED = 42; F = lib.ECG_FEATURES; NB = 100 if debug else 500
data = lib.load_data(debug_n=2000 if debug else None)
tr_all, va_all = lib.add_groups(data.train), lib.add_groups(data.val)
tr = tr_all.dropna(subset=["lvef_value"]).reset_index(drop=True)
va = va_all.dropna(subset=["lvef_value"]).reset_index(drop=True)
N = dict(n_train_tot=len(tr_all), n_train=len(tr), n_val_tot=len(va_all), n_val=len(va),
         pat_train=int(tr.patient_key.nunique()), pat_val=int(va.patient_key.nunique()))
print(N)
Xtr, ytr, Xva, yva = tr[F].astype(float), tr.lvef_value.values, va[F].astype(float), va.lvef_value.values
imp = lambda: SimpleImputer(strategy="median", add_indicator=True)
def reg_zoo():
    return {"ridge": make_pipeline(imp(), StandardScaler(), Ridge(alpha=1.0)),
            "random_forest": make_pipeline(imp(), RandomForestRegressor(n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=SEED)),
            "hist_gb": HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=250, random_state=SEED),
            "mlp_dense": make_pipeline(imp(), StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, max_iter=200, random_state=SEED))}
def ci(v): return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
def boot_reg(y, p, n=NB, seed=0):
    rng = np.random.default_rng(seed); m, r = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); m.append(mean_absolute_error(y[i], p[i])); r.append(r2_score(y[i], p[i]))
    return ci(m), ci(r)
log = {"val_configs": {}}

# ------------------------------------------------------------------ (a) regressione
rows, preds = [], {}
pb = np.full(len(yva), ytr.mean())
(mlo, mhi), _ = boot_reg(yva, pb)
rows.append(dict(model="baseline_media_train", mae=mean_absolute_error(yva, pb), mae_lo=mlo, mae_hi=mhi, r2=r2_score(yva, pb), r2_lo=np.nan, r2_hi=np.nan,
                 spearman=np.nan, n=len(yva)))
for name, m in reg_zoo().items():
    m.fit(Xtr, ytr); p = m.predict(Xva); preds[name] = p
    (mlo, mhi), (rlo, rhi) = boot_reg(yva, p)
    rows.append(dict(model=name, mae=mean_absolute_error(yva, p), mae_lo=mlo, mae_hi=mhi, r2=r2_score(yva, p), r2_lo=rlo, r2_hi=rhi,
                     spearman=stats.spearmanr(yva, p)[0], n=len(yva)))
    joblib.dump(m, out / f"reg_{name}.joblib")
REG = pd.DataFrame(rows); REG.to_csv(out / "a_regression.csv", index=False)
log["val_configs"]["regressione_4_modelli"] = 4
print("REG"); lib.show(REG.round(3))
# differenza di MAE vs baseline con IC (bootstrap appaiato) per il miglior modello
best_reg = REG.iloc[1:].sort_values("r2", ascending=False).model.iloc[0]
rng = np.random.default_rng(1); d = []
for _ in range(NB):
    i = rng.integers(0, len(yva), len(yva))
    d.append(mean_absolute_error(yva[i], pb[i]) - mean_absolute_error(yva[i], preds[best_reg][i]))
log["best_reg"] = best_reg; log["mae_gain_vs_baseline"] = [float(np.mean(d)), *ci(d)]

# ------------------------------------------------------------------ (a) classificazione <=45
Xc_tr, yc_tr, _ = lib.get_xy(tr_all, "lvef_lte_45_flag"); Xc_va, yc_va, dva_c = lib.get_xy(va_all, "lvef_lte_45_flag")
CLS, cpred = lib.eval_models(Xc_tr, yc_tr, Xc_va, yc_va)
CLS["n_pos_val"] = int(yc_va.sum()); CLS["prev_train"] = float(yc_tr.mean())
CLS.to_csv(out / "a_classification.csv", index=False); log["val_configs"]["classificazione_5_modelli"] = 5
print("CLS"); lib.show(CLS.round(3))
best_cls = CLS.model.iloc[0]; log["best_cls"] = best_cls

# ------------------------------------------------------------------ (b-i) stratificazione per QRS
glob = reg_zoo()["hist_gb"].fit(Xtr, ytr); pg = glob.predict(Xva)
def hgb_loc(): return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150, random_state=SEED)
rows = []; order = ["<100", "100-119", ">=120"]
for g in order:
    mt, mv = (tr.qrs_bin == g).values, (va.qrs_bin == g).values
    if mv.sum() < 30 or mt.sum() < 100: rows.append(dict(qrs=g, n_val=int(mv.sum()), n_train=int(mt.sum()), nota="troppo pochi")); continue
    pl = hgb_loc().fit(Xtr[mt], ytr[mt]).predict(Xva[mv]); y = yva[mv]; pgl = pg[mv]
    rng = np.random.default_rng(2); dm, dr = [], []
    for _ in range(NB):
        i = rng.integers(0, len(y), len(y))
        dm.append(mean_absolute_error(y[i], pl[i]) - mean_absolute_error(y[i], pgl[i]))  # <0 = locale meglio
        dr.append(r2_score(y[i], pl[i]) - r2_score(y[i], pgl[i]))                        # >0 = locale meglio
    rows.append(dict(qrs=g, n_val=int(mv.sum()), n_train=int(mt.sum()), lvef_mean_val=float(y.mean()),
                     mae_glob=mean_absolute_error(y, pgl), mae_loc=mean_absolute_error(y, pl),
                     dMAE=mean_absolute_error(y, pl) - mean_absolute_error(y, pgl), dMAE_lo=ci(dm)[0], dMAE_hi=ci(dm)[1],
                     r2_glob=r2_score(y, pgl), r2_loc=r2_score(y, pl),
                     dR2=r2_score(y, pl) - r2_score(y, pgl), dR2_lo=ci(dr)[0], dR2_hi=ci(dr)[1]))
STRAT = pd.DataFrame(rows); STRAT.to_csv(out / "b1_stratification.csv", index=False)
log["val_configs"]["stratificazione_3_modelli_locali"] = 3
print("STRAT"); lib.show(STRAT.round(3))

# ------------------------------------------------------------------ (b-ii) interazione formale (solo train)
# Ridge additivo vs Ridge con prodotti qrs x altre feature; qrs continuo e qrs>=120 (proxy di blocco di branca).
def design(df, kind):
    X = df[F].astype(float).copy()
    X["pr_missing"] = X.pr_interval.isna().astype(float); X["ar_missing"] = X.atrial_rate.isna().astype(float)
    X = X.fillna(X.median()); mu, sd = X[F].mean(), X[F].std()
    Z = (X[F] - mu) / sd
    base = pd.concat([Z, X[["pr_missing", "ar_missing"]]], axis=1)
    if kind == "add": return base
    q = Z["qrs_duration"] if kind == "cont" else (df.qrs_duration >= 120).astype(float).values
    others = [c for c in F if c != "qrs_duration"]
    inter = pd.DataFrame({f"qrsx{c}": q * Z[c] for c in others}, index=Z.index)
    if kind == "bin": base = base.assign(qrs120=q)
    return pd.concat([base, inter], axis=1)
def cluster_ols(df, kind):
    """OLS con errori standard robusti per cluster (paziente), correzione piccoli campioni; niente statsmodels."""
    D = design(df, kind).astype(float); names = ["const"] + list(D.columns)
    X = np.c_[np.ones(len(D)), D.values]; y = df.lvef_value.values
    XtXi = np.linalg.pinv(X.T @ X); beta = XtXi @ X.T @ y; e = y - X @ beta
    g = pd.factorize(df.patient_key)[0]; G = g.max() + 1
    S = np.zeros((G, X.shape[1])); np.add.at(S, g, X * e[:, None])
    V = XtXi @ (S.T @ S) @ XtXi * (G / (G - 1)) * ((len(y) - 1) / (len(y) - X.shape[1]))
    return names, beta, V, G
def cluster_wald(df, kind, cols_int):
    names, beta, V, G = cluster_ols(df, kind); idx = [names.index(c) for c in cols_int]
    b = beta[idx]; Vi = V[np.ix_(idx, idx)]; q = len(idx)
    W = float(b @ np.linalg.solve(Vi, b) / q); return W, float(stats.f.sf(W, q, G - 1))
def naive_F(df, kind):
    a, b = design(df, "add").astype(float).values, design(df, kind).astype(float).values; y = df.lvef_value.values
    def rss(X):
        X = np.c_[np.ones(len(X)), X]; beta = np.linalg.lstsq(X, y, rcond=None)[0]; return ((y - X @ beta) ** 2).sum(), X.shape[1]
    r0, k0 = rss(a); r1, k1 = rss(b); df1, df2 = k1 - k0, len(y) - k1
    Fv = ((r0 - r1) / df1) / (r1 / df2); return float(Fv), float(stats.f.sf(Fv, df1, df2))
inter_rows = []
for kind in ["cont", "bin"]:
    cols_int = [c for c in design(tr.head(50), kind).columns if c.startswith("qrsx")]
    Fn, pn = naive_F(tr, kind)
    try: Wc, pc = cluster_wald(tr, kind, cols_int)
    except Exception as e: Wc, pc = np.nan, np.nan; print("wald fail", e)
    # CV raggruppata: errore quadratico per riga, differenza additivo - interazione (>0 = interazione meglio)
    gk = GroupKFold(5); groups = tr.patient_key.values; se_a = np.zeros(len(tr)); se_i = np.zeros(len(tr))
    Da, Di = design(tr, "add").values, design(tr, kind).values
    for a_, b_ in gk.split(Da, ytr, groups):
        for D, se in ((Da, se_a), (Di, se_i)):
            m = Ridge(alpha=1.0).fit(D[a_], ytr[a_]); se[b_] = (ytr[b_] - m.predict(D[b_])) ** 2
    # bootstrap per paziente (cluster) sulla differenza di MSE
    pk = pd.factorize(groups)[0]; npat = pk.max() + 1
    s_diff = np.bincount(pk, weights=se_a - se_i, minlength=npat); cnt = np.bincount(pk, minlength=npat)
    rng = np.random.default_rng(3); dd = []
    for _ in range(NB):
        j = rng.integers(0, npat, npat); dd.append(s_diff[j].sum() / cnt[j].sum())
    inter_rows.append(dict(qrs_codifica=kind, n_termini_interazione=len(cols_int), F_naive=Fn, p_naive=pn, Wald_F_cluster=Wc, p_cluster=pc,
                           cv_mse_add=float(se_a.mean()), cv_mse_int=float(se_i.mean()), cv_dMSE=float(np.mean(se_a - se_i)),
                           cv_dMSE_lo=ci(dd)[0], cv_dMSE_hi=ci(dd)[1],
                           cv_r2_add=1 - se_a.mean() / ytr.var(), cv_r2_int=1 - se_i.mean() / ytr.var(), n=len(tr), n_pazienti=int(npat)))
# controllo non lineare: HGB con/senza qrs (non e' interazione: serve solo come riferimento)
INTER = pd.DataFrame(inter_rows); INTER.to_csv(out / "b2_interaction.csv", index=False)
print("INTER"); lib.show(INTER.round(5))
# interazioni singole (qrs continuo x feature): coefficiente cluster-robusto
names, beta, V, G = cluster_ols(tr, "cont"); se = np.sqrt(np.diag(V))
coef = pd.DataFrame({"coef": beta, "se": se, "p": 2 * stats.norm.sf(np.abs(beta / se))}, index=names)
coef = coef[coef.index.str.startswith("qrsx") | (coef.index == "qrs_duration")]
coef.to_csv(out / "b2_interaction_terms.csv"); lib.show(coef.round(4))

# ------------------------------------------------------------------ (b-iii, c) coppie
D_BIG, D_SMALL, QUANT, K = 15, 5, 0.20, 8
MATCH_SETS = {"A_eta_sesso_freq": ["age_at_ecg", "sex", "ventricular_rate"], "B_eta_sesso": ["age_at_ecg", "sex"],
              "C_eta_sesso_freq_PR_QTc": ["age_at_ecg", "sex", "ventricular_rate", "pr_interval", "qt_corrected"],
              "D_eta_freq_PR_QTc_senza_sesso": ["age_at_ecg", "ventricular_rate", "pr_interval", "qt_corrected"]}
def pairs_test(pool, M, seed=0):
    pool = pool.reset_index(drop=True)
    Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(pool[M].astype(float)))
    if "sex" in M: Z[:, M.index("sex")] *= 10   # sesso quasi esatto
    k = min(K, len(pool) - 1); dist, idx = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)
    pk = pool.patient_key.values; pr = []
    for i in range(len(pool)):
        for d_, j in zip(dist[i, 1:], idx[i, 1:]):
            if pk[j] != pk[i]: pr.append((i, j, d_)); break
    P = pd.DataFrame(pr, columns=["i", "j", "dist"])
    P = P[P.dist <= P.dist.quantile(QUANT)].copy()
    L, Q = pool.lvef_value.values, pool.qrs_duration.values
    P["dL"] = np.abs(L[P.i] - L[P.j]); P["dQ"] = np.abs(Q[P.i] - Q[P.j])
    lowi = L[P.i] < L[P.j]; P["dQs"] = np.where(lowi, Q[P.i] - Q[P.j], Q[P.j] - Q[P.i])  # QRS(LVEF bassa) - QRS(altro)
    P["dAge"] = np.abs(pool.age_at_ecg.values[P.i] - pool.age_at_ecg.values[P.j])
    P = P.dropna(subset=["dQ"]); big, small = P[P.dL >= D_BIG], P[P.dL < D_SMALL]
    r = dict(n_pairs=len(P), n_big=len(big), n_small=len(small))
    if len(big) < 20 or len(small) < 20: return r
    rng = np.random.default_rng(seed); dm, ds, fr = [], [], []
    bq, sq, bs = big.dQ.values, small.dQ.values, big.dQs.dropna().values
    for _ in range(NB):
        dm.append(np.median(rng.choice(bq, len(bq))) - np.median(rng.choice(sq, len(sq))))
        b = rng.choice(bs, len(bs)); ds.append(np.median(b)); fr.append((b > 0).mean())
    r.update(med_dQ_big=float(np.median(bq)), med_dQ_small=float(np.median(sq)), diff_med=float(np.median(bq) - np.median(sq)),
             diff_lo=ci(dm)[0], diff_hi=ci(dm)[1], p_mw=float(stats.mannwhitneyu(bq, sq, alternative="greater").pvalue),
             med_signed=float(np.median(bs)), signed_lo=ci(ds)[0], signed_hi=ci(ds)[1], frac_qrs_wider_in_low=float((bs > 0).mean()),
             frac_lo=ci(fr)[0], frac_hi=ci(fr)[1], p_wilcoxon=float(stats.wilcoxon(bs, alternative="greater").pvalue),
             p_wilcoxon_two=float(stats.wilcoxon(bs).pvalue),
             med_dAge_big=float(big.dAge.median()), med_dAge_small=float(small.dAge.median()),
             rho=float(stats.spearmanr(P.dL, P.dQ)[0]))
    return r
# train: 1 ECG casuale per paziente (indipendenza), val: gia' 1 per paziente
tr1 = tr.sample(frac=1, random_state=SEED).drop_duplicates("patient_key")
PR = []
for nm, M in MATCH_SETS.items():
    for split, pool in (("train_1ECG_per_paz", tr1), ("val_replica", va)):
        r = pairs_test(pool, M); r.update(match=nm, split=split, match_on=",".join(M)); PR.append(r)
    log["val_configs"]["pairs_" + nm] = 1
PRD = pd.DataFrame(PR); PRD.to_csv(out / "b3_pairs.csv", index=False)
print("PAIRS"); lib.show(PRD.round(4))

# ------------------------------------------------------------------ candidato lvef_lte_45 (modello globale, train soltanto)
if not debug:
    best_m = lib.model_zoo()[best_cls].fit(Xc_tr, yc_tr); pc = best_m.predict_proba(Xc_va)[:, 1]
    mb = lib.metrics_binary(yc_va, pc)
    src = ("import joblib\nfrom pathlib import Path\nimport pandas as pd\n_M = joblib.load(Path(__file__).parent / 'model.joblib')\n"
           "def predict(df, lib):\n    X = df[lib.ECG_FEATURES].astype(float)\n"
           "    return pd.DataFrame({'score': _M.predict_proba(X)[:, 1], 'in_scope': True}, index=df.index)\n")
    lib.save_candidate(out / "candidates" / "lvef_lte45_global", "lvef_lte45_global", "lvef_lte_45_flag", "global",
                       f"Modello globale {best_cls} su 7 feature ECG, addestrato sul train (solo righe con LVEF). Accuratezza moderata (<0.80).",
                       {**mb, "split": "val", "model": best_cls}, src, {"model.joblib": best_m})
    log["candidate_auroc"] = mb
json.dump({**log, "N": N}, open(out / "summary.json", "w"), indent=2, default=float)
print("OK", log["val_configs"])
