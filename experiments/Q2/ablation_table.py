"""Q2: tabella di ablazione derivazioni (seed 0, popolazione clean). Solo aggregati."""
import sys
sys.path.insert(0, "lib")
import numpy as np, pandas as pd
import pipe2_lib as P

_, dev = P.load_dev()
clean = dev.lvef_known.values
y_all = dev[P.TARGETS].values
groups = dev.patient_key.values[clean]
sets = ["all", "eight", "limb", "chest", "I_II", "I_II_V1", "II", "I"]
nleads = {"all": 12, "eight": 8, "limb": 6, "chest": 6, "I_II": 2, "I_II_V1": 3, "II": 1, "I": 1}
tg = ["shd_moderate_or_greater_flag", "lvef_lte_45_flag"]
cond = [t for t in P.TARGETS if t.startswith("rv_systolic") or t.startswith("aortic_stenosis")]
tg += cond
print("target:", tg)


def load(s):
    f = "results/Q1/main/oof_seed0.npy" if s == "all" else f"results/Q2/{s}/oof_seed0.npy"
    return np.load(f)


preds = {s: load(s) for s in sets}
rows = []
for t in tg:
    j = P.TARGETS.index(t)
    y = y_all[clean, j]
    pf = preds["all"][clean, j] if preds["all"].shape[0] == len(dev) else preds["all"][:, j]
    for s in sets:
        p = preds[s][clean, j] if preds[s].shape[0] == len(dev) else preds[s][:, j]
        a, lo, hi = P.auc_ci_cluster(y, p, groups)
        if s == "all":
            d = dlo = dhi = 0.0
        else:
            d, dlo, dhi = P.auc_diff_ci_cluster(y, p, pf, groups)
        rows.append(dict(target=t, set=s, n_leads=nleads[s], auroc=a, lo=lo, hi=hi, diff_vs_all=d, diff_lo=dlo, diff_hi=dhi))
df = pd.DataFrame(rows)
df.to_csv("results/Q2/ablation_table.csv", index=False)
pd.set_option("display.width", 200)
print(df.round(3).to_string())
