"""Q7: controlli indipendenti 1-3 (fold per paziente, riproduzione AUROC con sklearn, popolazioni clean).
Codice proprio: fold ricalcolate con zlib.crc32 qui (non con pipe2_lib.fold_of), AUROC con sklearn.roc_auc_score,
bootstrap per paziente scritto qui. Stampa e salva SOLO aggregati.
Uso: python3 experiments/Q7/check_folds_auc.py [--debug-n 2000]
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))
import echonext_lib as lib  # noqa: E402  (solo per caricare train+val in modo sicuro)

ap = argparse.ArgumentParser()
ap.add_argument("--debug-n", type=int, default=None)
a = ap.parse_args()
OUT = ROOT / "results" / "Q7"
OUT.mkdir(parents=True, exist_ok=True)

data = lib.load_data(debug_n=a.debug_n)
dev = pd.concat([data.train, data.val], ignore_index=True)
pk = dev.patient_key.values
myfold = np.array([zlib.crc32(f"42-{k}".encode()) % 5 for k in pk])
clean = dev.lvef_value.notna().values
targets = list(lib.FLAG_COLS)
res = {"n_dev": int(len(dev)), "n_clean": int(clean.sum()), "n_patients": int(pd.Series(pk).nunique())}

# patient -> one fold only
pf = pd.DataFrame({"p": pk, "f": myfold}).groupby("p").f.nunique()
res["patients_in_more_than_one_fold"] = int((pf > 1).sum())
# rows without LVEF: labels all zero?
lab = dev.loc[~clean, targets].fillna(0).values
res["noLVEF_rows_any_positive_label"] = int((lab.sum(1) > 0).sum())

if a.debug_n:
    print(json.dumps(res, indent=1)); sys.exit(0)

runs = sorted({p.parent for p in (ROOT / "results").glob("Q*/**/oof_seed*_fold*.npz")})
fold_checks = []
for r in runs:
    cfg = json.loads((r / "config.json").read_text())
    for z in sorted(r.glob("oof_seed*_fold*.npz")):
        f = int(z.stem.split("fold")[1]); d = np.load(z)
        idx = d["idx"]
        te_set = set(pk[idx])
        exp_tr = (myfold != f) & (clean if cfg["population"] == "clean" else True)
        tr_pat = set(pk[exp_tr])
        fold_checks.append({
            "run": str(r.relative_to(ROOT)), "file": z.name, "population": cfg["population"],
            "idx_all_in_fold": bool((myfold[idx] == f).all()),
            "idx_equals_full_fold": bool(len(idx) == int((myfold == f).sum())),
            "patient_overlap_train_test": len(te_set & tr_pat),
            "n_train_file": int(d["n_train"]), "n_train_expected": int(exp_tr.sum()),
            "n_train_ok": bool(int(d["n_train"]) == int(exp_tr.sum())),
        })
FC = pd.DataFrame(fold_checks)
FC.to_csv(OUT / "fold_checks.csv", index=False)
res["n_npz_checked"] = len(FC)
res["n_runs_checked"] = len(runs)
res["all_idx_in_fold"] = bool(FC.idx_all_in_fold.all())
res["all_idx_full_fold"] = bool(FC.idx_equals_full_fold.all())
res["max_patient_overlap"] = int(FC.patient_overlap_train_test.max())
res["all_n_train_ok"] = bool(FC.n_train_ok.all())
res["n_train_bad"] = FC.loc[~FC.n_train_ok, ["run", "file", "n_train_file", "n_train_expected"]].to_dict("records")


def boot_ci(y, p, g, B=300, seed=7):
    codes, inv = np.unique(g, return_inverse=True)
    rows_by = pd.Series(np.arange(len(y))).groupby(inv).apply(np.array).values
    rng = np.random.default_rng(seed); v = []
    for _ in range(B):
        s = rng.integers(0, len(rows_by), len(rows_by))
        i = np.concatenate([rows_by[k] for k in s])
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# reproduction of AUROC from oof_ensemble.npy vs metrics_clean.csv / metrics_all.csv
rep = []
for r in runs:
    if not (r / "oof_ensemble.npy").exists():
        continue
    E = np.load(r / "oof_ensemble.npy")
    for pop, mask in (("clean", clean), ("all", np.ones(len(dev), bool))):
        M = pd.read_csv(r / f"metrics_{pop}.csv").set_index("target")
        for j, t in enumerate(targets):
            short = t.replace("_moderate_or_greater_flag", "").replace("_flag", "")
            if short not in M.index:
                continue
            m = mask & dev[t].notna().values & ~np.isnan(E[:, j])
            y = dev.loc[m, t].astype(int).values
            mine = roc_auc_score(y, E[m, j])
            rep.append({"run": str(r.relative_to(ROOT)), "pop": pop, "target": short, "n": int(m.sum()), "n_pos": int(y.sum()),
                        "auc_q7": mine, "auc_reported": float(M.loc[short, "cnn_auroc"]),
                        "abs_diff": abs(mine - float(M.loc[short, "cnn_auroc"])), "n_reported": int(M.loc[short, "n"])})
R = pd.DataFrame(rep)
R.to_csv(OUT / "auc_reproduction.csv", index=False)
res["auc_reproduced_n"] = len(R)
res["auc_max_abs_diff"] = float(R.abs_diff.max())
res["auc_n_over_0005"] = int((R.abs_diff > 0.005).sum())
res["n_mismatch"] = int((R.n != R.n_reported).sum())

# CI with my own patient bootstrap for main run, 4 key labels, clean
E = np.load(ROOT / "results/Q1/main/oof_ensemble.npy")
T = np.load(ROOT / "results/Q1/main/tabular_oof.npz")["P"]
ci = {}
for t in ["shd_moderate_or_greater_flag", "lvef_lte_45_flag", "rv_systolic_dysfunction_moderate_or_greater_flag",
          "aortic_stenosis_moderate_or_greater_flag", "aortic_regurgitation_moderate_or_greater_flag"]:
    j = targets.index(t); m = clean & dev[t].notna().values
    y = dev.loc[m, t].astype(int).values
    lo, hi = boot_ci(y, E[m, j], pk[m])
    tlo, thi = boot_ci(y, T[m, j], pk[m])
    ci[t] = {"cnn": [roc_auc_score(y, E[m, j]), lo, hi], "tab": [roc_auc_score(y, T[m, j]), tlo, thi]}
res["ci_main_clean"] = ci
# seed sd (main)
ss = {}
for t in ["shd_moderate_or_greater_flag", "lvef_lte_45_flag"]:
    j = targets.index(t); m = clean & dev[t].notna().values; y = dev.loc[m, t].astype(int).values
    ss[t] = [roc_auc_score(y, np.load(ROOT / f"results/Q1/main/oof_seed{s}.npy")[m, j]) for s in range(3)]
res["per_seed_auc_main"] = ss
# LVEF regression
mr = clean
yv = dev.lvef_value.values[mr]
for nm, pr in (("cnn", E[mr, 12]), ("tab", T[mr, 12])):
    res[f"lvef_{nm}_mae"] = float(np.mean(np.abs(yv - pr)))
    res[f"lvef_{nm}_r2"] = float(1 - np.sum((yv - pr) ** 2) / np.sum((yv - yv.mean()) ** 2))
# Q2: I_II_V1 vs all seed0 (clean)
A0 = np.load(ROOT / "results/Q1/main/oof_seed0.npy"); B = np.load(ROOT / "results/Q2/I_II_V1/oof_ensemble.npy")
q2 = {}
for t in ["shd_moderate_or_greater_flag", "lvef_lte_45_flag"]:
    j = targets.index(t); m = clean & dev[t].notna().values; y = dev.loc[m, t].astype(int).values
    q2[t] = {"all_seed0": roc_auc_score(y, A0[m, j]), "I_II_V1": roc_auc_score(y, B[m, j])}
    q2[t]["diff"] = q2[t]["I_II_V1"] - q2[t]["all_seed0"]
res["q2_I_II_V1"] = q2
(OUT / "check_folds_auc.json").write_text(json.dumps(res, indent=1, default=float))
print(json.dumps({k: v for k, v in res.items()}, indent=1, default=float))
