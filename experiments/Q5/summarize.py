import json, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
R = Path(__file__).resolve().parents[2] / "results/Q5"
c = json.loads((R / "calib_ops.json").read_text())
f = {i: json.loads((R / f"occlusion_fold{i}.json").read_text()) for i in (0, 1)}
f3 = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
print("prev", c["prev_train_clean"], c["prev_val_clean"])
for n, r in c["calib"].items():
    print(n, "prev", round(r["prev"], 3), "pos", r["n_pos"], "meanp", round(r["mean_pred"], 3), "slope", round(r["slope"], 3), "int", round(r["intercept"], 3))
    for k in ("ece_raw", "ece_iso", "brier_raw", "brier_iso", "auc_raw", "auc_iso"):
        print("  ", k, f3(r[k]))
    for s, v in r["by_split"].items():
        print("   split", s, v["n"], round(v["prev"], 3), round(v["mean_pred"], 3), "ece", f3(v["ece"]))
for n, o in c["ops"].items():
    for k, r in o.items():
        print(n, k, "thr", round(r["soglia"], 3), "flag%", round(r["n_flag_pct"], 3), *[f"{m} {f3(r[m])}" for m in ("sens", "spec", "ppv", "npv")])
for t in ("shd", "lvef45"):
    for part in ("lead", "window"):
        keys = list(f[0][part])
        a = [f[0][part][k][t][0] for k in keys]; b = [f[1][part][k][t][0] for k in keys]
        rho = spearmanr(a, b)
        print(t, part, "spearman fold0-fold1", round(rho[0], 2), "p", round(rho[1], 3))
        for k in keys:
            x, y = f[0][part][k][t], f[1][part][k][t]
            print("  ", k, "f0", f3(x), "f1", f3(y))
