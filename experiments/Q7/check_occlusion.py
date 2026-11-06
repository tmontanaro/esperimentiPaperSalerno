"""Q7: riproduzione indipendente (ridotta) dell'occlusione di Q5: modello seed0/fold0 di results/Q1/main, N righe clean del fold 0,
azzero un canale alla volta e misuro il calo di AUROC SHD (sklearn) con IC bootstrap per paziente appaiato. Solo aggregati."""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib")); sys.path.insert(0, str(ROOT / "tools"))
import echonext_lib as lib  # noqa: E402
from cnn_cv import make_net  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
data = lib.load_data()
dev = pd.concat([data.train, data.val], ignore_index=True)
fold = np.array([zlib.crc32(f"42-{k}".encode()) % 5 for k in dev.patient_key.values])
idx = np.where((fold == 0) & dev.lvef_value.notna().values)[0]
idx = np.sort(np.random.default_rng(1).choice(idx, min(N, len(idx)), replace=False))
maps = {s: np.load(data.wave_path(s), mmap_mode="r") for s in ("train", "val")}
X = np.stack([np.asarray(maps[s][r, 0]) for s, r in zip(dev.split.values[idx], dev.row_in_split.values[idx].astype(int))]).astype(np.float32)
y = dev.shd_moderate_or_greater_flag.values[idx].astype(int); g = dev.patient_key.values[idx]
dv = "cuda" if torch.cuda.is_available() else "cpu"
net = make_net(12, 13).to(dv); net.load_state_dict(torch.load(ROOT / "results/Q1/main/weights_seed0_fold0.pt", map_location=dv)); net.eval()


def pred(Xa):
    x = torch.from_numpy(Xa).permute(0, 2, 1).contiguous().to(dv)
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(net(x[a:a + 256])[:, 0]).cpu().numpy() for a in range(0, len(x), 256)])


p0 = pred(X)
z = np.load(ROOT / "results/Q1/main/oof_seed0_fold0.npz")
pos = {v: i for i, v in enumerate(z["idx"])}
res = {"n": len(idx), "n_pos": int(y.sum()), "auc_base": float(roc_auc_score(y, p0)),
       "max_abs_diff_vs_saved_oof": float(np.max(np.abs(p0 - z["preds"][[pos[i] for i in idx], 0])))}
codes, inv = np.unique(g, return_inverse=True)
groups = pd.Series(np.arange(len(y))).groupby(inv).apply(np.array).values
rng = np.random.default_rng(0)
boots = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]) for _ in range(200)]
res["drop"] = {}
for c in range(12):
    Xc = X.copy(); Xc[:, :, c] = 0; pc = pred(Xc)
    d = roc_auc_score(y, pc) - res["auc_base"]
    v = [roc_auc_score(y[b], pc[b]) - roc_auc_score(y[b], p0[b]) for b in boots]
    res["drop"][c] = [float(d), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    print(c, [round(x, 4) for x in res["drop"][c]], flush=True)
(ROOT / "results/Q7/check_occlusion.json").write_text(json.dumps(res, indent=1))
print({k: v for k, v in res.items() if k != "drop"})
