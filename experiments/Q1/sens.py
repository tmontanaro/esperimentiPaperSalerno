import sys, json
import numpy as np
sys.path.insert(0, "lib")
import pipe2_lib as P

_, dev = P.load_dev()
a = np.load("results/Q1/main/oof_seed0.npy")
b = np.load("results/Q1/all_rows/oof_seed0.npy")
cl = dev.lvef_known.values
out = {}
for t, name in [("shd", "shd_moderate_or_greater_flag"), ("lvef_lte_45", "lvef_lte_45_flag")]:
    j = P.TARGETS.index(name)
    y = dev[name].values[cl]
    g = dev.patient_key.values[cl]
    out[t] = dict(main=P.auc_ci_cluster(y, a[cl, j], g),
                  allrows=P.auc_ci_cluster(y, b[cl, j], g),
                  diff_all_minus_main=P.auc_diff_ci_cluster(y, b[cl, j], a[cl, j], g))
print(json.dumps(out, indent=1))
json.dump(out, open("results/Q1/sensitivity.json", "w"), indent=1)
