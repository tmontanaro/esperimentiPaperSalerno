"""R1-P3: coppie senza reciproche duplicate, bootstrap per paziente (multiway), filtro 20% descritto correttamente,
e riscrittura descrizione del candidato lvef_lte45_global. Uso: python3 pairs_dedup.py [debug]"""
import sys, json, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from lib import echonext_lib as lib
warnings.filterwarnings("ignore")
debug = len(sys.argv) > 1 and sys.argv[1] == "debug"
out = ROOT / "results" / "R1-P3" / ("debug" if debug else ""); out.mkdir(parents=True, exist_ok=True)
SEED = 42; NB = 100 if debug else 500; D_BIG, D_SMALL, QUANT, K = 15, 5, 0.20, 8
data = lib.load_data(debug_n=2000 if debug else None)
tr_all, va_all = lib.add_groups(data.train), lib.add_groups(data.val)
tr = tr_all.dropna(subset=["lvef_value"]).reset_index(drop=True); va = va_all.dropna(subset=["lvef_value"]).reset_index(drop=True)
MATCH_SETS = {"A_eta_sesso_freq": ["age_at_ecg", "sex", "ventricular_rate"], "B_eta_sesso": ["age_at_ecg", "sex"],
              "C_eta_sesso_freq_PR_QTc": ["age_at_ecg", "sex", "ventricular_rate", "pr_interval", "qt_corrected"],
              "D_eta_freq_PR_QTc_senza_sesso": ["age_at_ecg", "ventricular_rate", "pr_interval", "qt_corrected"]}
def ci(v): return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
def wmedian(x, w):
    o = np.argsort(x); x, w = x[o], w[o]; c = np.cumsum(w); return x[np.searchsorted(c, c[-1] / 2)]
def build(pool, M):
    pool = pool.reset_index(drop=True)
    Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(pool[M].astype(float)))
    if "sex" in M: Z[:, M.index("sex")] *= 10
    k = min(K, len(pool) - 1); dist, idx = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)
    pk = pool.patient_key.values; pr = []
    for i in range(len(pool)):
        for d_, j in zip(dist[i, 1:], idx[i, 1:]):
            if pk[j] != pk[i]: pr.append((i, j, d_)); break
    P = pd.DataFrame(pr, columns=["i", "j", "dist"]); n_dir = len(P)
    P["a"], P["b"] = np.minimum(P.i, P.j), np.maximum(P.i, P.j)
    P = P.sort_values("dist").drop_duplicates(["a", "b"]).reset_index(drop=True)   # elimina coppie reciproche
    n_unique = len(P); cut = float(P.dist.quantile(QUANT)); n_zero = int((P.dist == 0).sum())
    P = P[P.dist <= cut].copy(); n_after = len(P)
    L, Q = pool.lvef_value.values, pool.qrs_duration.values
    P["dL"] = np.abs(L[P.i] - L[P.j]); P["dQ"] = np.abs(Q[P.i] - Q[P.j])
    lowi = L[P.i] < L[P.j]; P["dQs"] = np.where(lowi, Q[P.i] - Q[P.j], Q[P.j] - Q[P.i])
    P = P.dropna(subset=["dQ"]).copy()
    codes = pd.factorize(pk)[0]; P["ia"] = codes[P.i.values]; P["ib"] = codes[P.j.values]
    info = dict(n_pool=len(pool), n_dirette=n_dir, n_uniche=n_unique, n_dist_zero=n_zero, frac_dist_zero=n_zero / n_unique,
                cutoff_q20=cut, n_dopo_filtro=n_after, frac_tenute=n_after / n_unique, n_pazienti_pool=int(codes.max() + 1),
                n_pazienti_in_coppie=int(len(np.unique(np.r_[P.ia, P.ib]))))
    return P, info
def test(P, npat):
    big, small = P[P.dL >= D_BIG], P[P.dL < D_SMALL]; r = dict(n_big=len(big), n_small=len(small))
    if len(big) < 20 or len(small) < 20: return r
    bs = big[big.dQs.notna()]; rng = np.random.default_rng(SEED)
    def stat(w):
        wb = w[big.ia.values] * w[big.ib.values]; ws = w[small.ia.values] * w[small.ib.values]; wsg = w[bs.ia.values] * w[bs.ib.values]
        return (wmedian(big.dQ.values, wb) - wmedian(small.dQ.values, ws), wmedian(bs.dQs.values, wsg),
                (wsg * (bs.dQs.values > 0)).sum() / wsg.sum())
    dm, ds, fr = [], [], []
    for _ in range(NB):   # bootstrap multiway: si ricampionano i PAZIENTI; peso coppia = w_i * w_j
        w = np.bincount(rng.integers(0, npat, npat), minlength=npat).astype(float); a, b, c = stat(w); dm.append(a); ds.append(b); fr.append(c)
    bq, sq, sg = big.dQ.values, small.dQ.values, bs.dQs.values
    r.update(med_dQ_big=float(np.median(bq)), med_dQ_small=float(np.median(sq)), diff_med=float(np.median(bq) - np.median(sq)),
             diff_lo=ci(dm)[0], diff_hi=ci(dm)[1], med_signed=float(np.median(sg)), signed_lo=ci(ds)[0], signed_hi=ci(ds)[1],
             frac_qrs_wider_in_low=float((sg > 0).mean()), frac_lo=ci(fr)[0], frac_hi=ci(fr)[1],
             frac_dQs_zero=float((sg == 0).mean()), rho=float(stats.spearmanr(P.dL, P.dQ)[0]))
    return r
tr1 = tr.sample(frac=1, random_state=SEED).drop_duplicates("patient_key"); rows = []
for nm, M in MATCH_SETS.items():
    for split, pool in (("train_1ECG_per_paz", tr1), ("val_replica", va)):
        P, info = build(pool, M); r = {**info, **test(P, info["n_pazienti_pool"])}
        r.update(match=nm, split=split, match_on=",".join(M)); rows.append(r)
R = pd.DataFrame(rows); R.to_csv(out / "pairs_dedup.csv", index=False); lib.show(R.round(3))
if not debug:
    Xc, yc, _ = lib.get_xy(tr_all, "lvef_lte_45_flag"); nfl = len(yc)
    nf_nolv = int((tr_all.lvef_value.isna() & tr_all.lvef_lte_45_flag.notna()).sum())
    cd = ROOT / "results/P3/candidates/lvef_lte45_global"; old = json.load(open(cd / "candidate.json"))
    desc = (f"Modello globale hist_gb su 7 feature ECG, addestrato sul solo train su tutte le righe con lvef_lte_45_flag disponibile "
            f"(n={nfl} ECG, di cui {nf_nolv} senza lvef_value), NON solo sulle righe con LVEF numerico. Valutato su tutte le {old['val_metrics']['n']} righe val con flag. "
            f"Accuratezza moderata (<0.80). Modello scelto tra 5 guardando il validation: AUROC ottimistica.")
    lib.save_candidate(cd, old["name"], old["target"], old["kind"], desc, old["val_metrics"], (cd / "candidate.py").read_text())
    json.dump(dict(n_train_flag=nfl, n_train_flag_senza_lvef_value=nf_nolv), open(out / "candidate_fix.json", "w"))
    print(desc)
