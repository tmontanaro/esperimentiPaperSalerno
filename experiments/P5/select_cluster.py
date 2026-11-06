"""P5 passo 1-3: selezione feature per il clustering (SOLO train), clustering finale, assegnazione train/val.
Uso: python3 select_cluster.py [debug]. Non usa il target ne' il validation per scegliere (il validation e' solo assegnato ai cluster)."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from lib import echonext_lib as lib

debug = len(sys.argv) > 1 and sys.argv[1] == "debug"
out = ROOT / "results" / "P5" / ("debug" if debug else ""); out.mkdir(parents=True, exist_ok=True)
data = lib.load_data(debug_n=2000 if debug else None)
tr, va = data.train, data.val
F = lib.ECG_FEATURES
rng = np.random.default_rng(42)
pool = tr.sample(frac=1, random_state=42).drop_duplicates("patient_key")  # 1 ECG per paziente
med = pool[F].median()
mu = ((pool[F].fillna(med)).mean()); sd = (pool[F].fillna(med)).std()
def prep(df): return ((df[F].fillna(med) - mu) / sd).clip(-4, 4).values
Z = prep(pool); n = len(Z)
sil_idx = rng.choice(n, min(4000, n), replace=False)
perm = rng.permutation(n); halves = [(perm[:n // 2], perm[n // 2:])]
perm2 = rng.permutation(n); halves.append((perm2[:n // 2], perm2[n // 2:]))
print("pool pazienti:", n)

def km(X, k, seed=42, n_init=5): return KMeans(k, n_init=n_init, random_state=seed).fit(X)
def score(cols, k, w=None):
    X = Z[:, cols] * (1 if w is None else w[cols])
    m = km(X, k); sil = silhouette_score(X[sil_idx], m.labels_[sil_idx])
    aris = []
    for a, b in halves:
        ma, mb = km(X[a], k, 1, 3), km(X[b], k, 2, 3)
        aris.append(adjusted_rand_score(ma.predict(X), mb.predict(X)))
    ari = float(np.mean(aris)); return float(sil), ari, (sil + ari) / 2

rows = []
paths = {}
for k in (3, 4, 5):
    cur = list(range(len(F))); path = []
    s = score(cur, k); rows.append({"method": "backward", "k": k, "n_feat": len(cur), "features": "|".join(F[i] for i in cur), "silhouette": s[0], "ari": s[1], "score": s[2], "on_path": True})
    path.append((cur[:], s))
    while len(cur) > 3:
        cand = []
        for j in cur:
            c = [i for i in cur if i != j]; s = score(c, k)
            rows.append({"method": "backward", "k": k, "n_feat": len(c), "features": "|".join(F[i] for i in c), "silhouette": s[0], "ari": s[1], "score": s[2], "on_path": False})
            cand.append((s[2], c, s))
        best = max(cand, key=lambda t: t[0]); cur = best[1]; path.append((cur[:], best[2]))
        for r in rows[::-1]:
            if r["k"] == k and r["features"] == "|".join(F[i] for i in cur): r["on_path"] = True; break
    paths[k] = path
    print("k", k, [(len(p), round(s[0], 3), round(s[1], 3), round(s[2], 3)) for p, s in path])

# sparse k-means (Witten & Tibshirani 2010) su sotto-campione
def bcss_vec(X, lab, k):
    tot = ((X - X.mean(0)) ** 2).sum(0)
    wss = sum(((X[lab == c] - X[lab == c].mean(0)) ** 2).sum(0) for c in range(k) if (lab == c).any())
    return tot - wss
def soft_w(a, s):
    a = np.maximum(a, 0)
    if np.linalg.norm(a) == 0: return np.ones_like(a) / np.sqrt(len(a))
    w = a / np.linalg.norm(a)
    if w.sum() <= s: return w
    lo, hi = 0, a.max()
    for _ in range(60):
        d = (lo + hi) / 2; w = np.maximum(a - d, 0); w = w / (np.linalg.norm(w) + 1e-12)
        if w.sum() > s: lo = d
        else: hi = d
    return w
def sparse_km(X, k, s, it=15):
    w = np.ones(X.shape[1]) / np.sqrt(X.shape[1]); lab = None
    for _ in range(it):
        m = KMeans(k, n_init=5, random_state=42).fit(X * np.sqrt(w)); lab = m.labels_
        wn = soft_w(bcss_vec(X, lab, k), s)
        if np.abs(wn - w).sum() < 1e-4: w = wn; break
        w = wn
    return w
sub = rng.choice(n, min(8000, n), replace=False)
srows = []
for k in (3, 4, 5):
    for s_ in (1.5, 2.0, 2.4):
        w = sparse_km(Z[sub], k, s_)
        cols = [i for i in range(len(F)) if w[i] > 1e-3]
        wv = np.sqrt(w)
        sil, ari, sc = score(list(range(len(F))), k, wv)
        srows.append({"method": "sparse_kmeans", "k": k, "s": s_, "n_feat": len(cols), "features": "|".join(F[i] for i in cols),
                      "weights": "|".join(f"{F[i]}:{w[i]:.2f}" for i in range(len(F))), "silhouette": sil, "ari": ari, "score": sc})
        print("sparse k", k, "s", s_, [f"{F[i]}:{w[i]:.2f}" for i in range(len(F)) if w[i] > 1e-3], round(sil, 3), round(ari, 3))
R = pd.concat([pd.DataFrame(rows), pd.DataFrame(srows)]); R.to_csv(out / "selection_log.csv", index=False)
n_conf = len(R); print("configurazioni di selezione provate:", n_conf)

# scelta automatica: miglior score tra i set (>=3 feature) del cammino all'indietro, per ciascun k; poi miglior k
best_per_k = {k: max(p, key=lambda t: t[1][2]) for k, p in paths.items()}
kbest = max(best_per_k, key=lambda k: best_per_k[k][1][2])
cols, sc = best_per_k[kbest]
feat = [F[i] for i in cols]
print("scelta: k", kbest, feat, "sil %.3f ari %.3f score %.3f" % sc)
fin = km(Z[:, cols], kbest, 42, 30)
order = np.argsort(-np.bincount(fin.labels_, minlength=kbest)); centers = fin.cluster_centers_[order]
def assign(df):
    X = prep(df)[:, cols]; d = ((X[:, None, :] - centers[None]) ** 2).sum(-1); return d.argmin(1)
tr_lab, va_lab = assign(tr), assign(va)
joblib.dump({"features": feat, "all_features": F, "median": med.to_dict(), "mean": mu.to_dict(), "std": sd.to_dict(), "clip": 4.0,
             "centers": centers, "k": kbest, "cols": cols}, out / "cluster_model.joblib")
np.save(out / "train_cluster.npy", tr_lab); np.save(out / "val_cluster.npy", va_lab)
alt = [{"k": k, "n_feat": len(c), "features": "|".join(F[i] for i in c), "silhouette": s[0], "ari": s[1], "score": s[2]} for k, (c, s) in best_per_k.items()]
pd.DataFrame(alt).to_csv(out / "best_per_k.csv", index=False)
json.dump({"k": int(kbest), "features": feat, "silhouette": sc[0], "ari": sc[1], "score": sc[2], "n_selection_configs": int(n_conf), "n_pool": int(n),
           "full_set_ari_same_k": float(paths[kbest][0][1][1]), "full_set_sil_same_k": float(paths[kbest][0][1][0])}, open(out / "selection_summary.json", "w"), indent=1)
print(pd.DataFrame(alt).round(3).to_string())
