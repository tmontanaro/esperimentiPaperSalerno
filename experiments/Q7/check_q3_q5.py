"""Q7: controlli 5-6. Ricalcolo indipendente (codice proprio) di numeri chiave di Q3, Q4, Q5 e di Holm. Solo aggregati."""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))
import echonext_lib as lib  # noqa: E402

dbg = int(sys.argv[1]) if len(sys.argv) > 1 else None
res = {}


def my_holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (m - r) * p[i])); adj[i] = run
    return adj


S = pd.read_csv(ROOT / "results/Q4/strata.csv")
res["q4_strata_n"] = len(S)
res["q4_strata_holm_maxdiff"] = float(np.nanmax(np.abs(my_holm(S.p.values) - S.p_holm.values)))
res["q4_strata_n_sig_mine"] = int((my_holm(S.p.values) < 0.05).sum()); res["q4_strata_n_sig_reported"] = int(S.sig.sum())
Rr = pd.read_csv(ROOT / "results/Q4/error_rules.csv")
res["q4_rules_holm_maxdiff"] = float(np.nanmax(np.abs(my_holm(Rr.p.values) - Rr.p_holm.values)))
res["q4_rules_n_regge"] = int(Rr.regge.sum())
q3 = json.loads((ROOT / "results/Q3/q3_results.json").read_text())
fam = {"a": [q3["a"][k]["p"] for k in q3["a"]], "b": [q3["b"]["delta_auc"][3], q3["b"]["delta_logloss_base_meno_full"][3]],
       "c": [q3["c"]["delta_r2"][3], q3["c"]["correlazione_parziale"][3]]}
res["q3_holm_mine"] = {k: my_holm(v).round(4).tolist() for k, v in fam.items()}

data = lib.load_data(debug_n=dbg)
dev = pd.concat([data.train, data.val], ignore_index=True)
if dbg:
    print(res); sys.exit(0)
pk = dev.patient_key.values
fold = np.array([zlib.crc32(f"42-{k}".encode()) % 5 for k in pk])
clean = dev.lvef_value.notna().values
T = list(lib.FLAG_COLS)
E = np.load(ROOT / "results/Q1/main/oof_ensemble.npy")
jS, jL = T.index("shd_moderate_or_greater_flag"), T.index("lvef_lte_45_flag")

# Q4: two strata reproduced with sklearn
qrs = dev.qrs_duration.values
for nm, m0 in (("SHD_QRS<100", qrs < 100), ("SHD_QRS100-119", (qrs >= 100) & (qrs < 120))):
    m = clean & m0; y = dev.shd_moderate_or_greater_flag.values[m].astype(int)
    res[f"q4_{nm}"] = [int(m.sum()), int(y.sum()), float(roc_auc_score(y, E[m, jS]))]
# case-mix: tabular in QRS<100 for SHD too (does the CNN keep its advantage?)
Tab = np.load(ROOT / "results/Q1/main/tabular_oof.npz")["P"]
m = clean & (qrs < 100); y = dev.shd_moderate_or_greater_flag.values[m].astype(int)
res["q4_SHD_QRS<100_tab"] = float(roc_auc_score(y, Tab[m, jS]))

# Q3 b: own CV for y ~ logit(cnn) vs + qrs + qrs*logit(cnn), same patient folds, clean
m = clean & ~np.isnan(qrs)
y = dev.lvef_lte_45_flag.values[m].astype(int)
p = np.clip(E[m, jL], 1e-6, 1 - 1e-6); lg = np.log(p / (1 - p)); q = (qrs[m] - np.nanmean(qrs[m])) / np.nanstd(qrs[m])
X0 = lg[:, None]; X1 = np.c_[lg, q, q * lg]
o0, o1 = np.zeros(len(y)), np.zeros(len(y)); fm = fold[m]
for f in range(5):
    tr, te = fm != f, fm == f
    o0[te] = LogisticRegression(C=1e6, max_iter=1000).fit(X0[tr], y[tr]).predict_proba(X0[te])[:, 1]
    o1[te] = LogisticRegression(C=1e6, max_iter=1000).fit(X1[tr], y[tr]).predict_proba(X1[te])[:, 1]
res["q3b_mine"] = {"auc_base": roc_auc_score(y, o0), "auc_full": roc_auc_score(y, o1), "delta": roc_auc_score(y, o1) - roc_auc_score(y, o0)}

# Q5: ECE (10 quantile bins, own code) and sens-90% operating point for SHD
def ece(y, p, b=10):
    e = np.quantile(p, np.linspace(0, 1, b + 1)); k = np.clip(np.searchsorted(e, p, side="right") - 1, 0, b - 1)
    return float(sum(abs(p[k == i].mean() - y[k == i].mean()) * (k == i).sum() for i in range(b) if (k == i).any()) / len(y))


for nm, j in (("shd", jS), ("lvef45", jL)):
    yy = dev[T[j]].values[clean].astype(int); pp = E[clean, j]
    res[f"q5_ece_raw_{nm}"] = ece(yy, pp); res[f"q5_brier_raw_{nm}"] = float(np.mean((pp - yy) ** 2))
    thr = np.quantile(pp[yy == 1], 0.10)  # 90% sensitivity
    res[f"q5_sens90_{nm}"] = {"thr": float(thr), "sens": float((pp[yy == 1] >= thr).mean()), "spec": float((pp[yy == 0] < thr).mean()),
                             "ppv": float(yy[pp >= thr].mean())}
# prevalence train vs val (clean)
for s in ("train", "val"):
    mm = clean & (dev.split.values == s)
    res[f"prev_shd_{s}_clean"] = float(dev.shd_moderate_or_greater_flag.values[mm].mean())
# Q1 sensitivity: all_rows vs main seed0, on clean rows
A = np.load(ROOT / "results/Q1/all_rows/oof_seed0.npy"); M0 = np.load(ROOT / "results/Q1/main/oof_seed0.npy")
for nm, j in (("shd", jS), ("lvef45", jL)):
    yy = dev[T[j]].values[clean].astype(int)
    res[f"q1_allrows_minus_clean_{nm}"] = float(roc_auc_score(yy, A[clean, j]) - roc_auc_score(yy, M0[clean, j]))
# Q2 'eight' vs all seed0
B = np.load(ROOT / "results/Q2/eight/oof_ensemble.npy")
for nm, j in (("shd", jS), ("lvef45", jL)):
    yy = dev[T[j]].values[clean].astype(int)
    res[f"q2_eight_minus_all_{nm}"] = float(roc_auc_score(yy, B[clean, j]) - roc_auc_score(yy, M0[clean, j]))
(ROOT / "results/Q7/check_q3_q5.json").write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float))
