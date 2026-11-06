"""Q5 c): occlusione per derivazione e per finestra (0.5 s = 250 campioni) sul modello (seed 0, fold F). Solo aggregati."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib")); sys.path.insert(0, str(ROOT / "tools"))
import pipe2_lib as P
import cnn_cv as C
import torch

F = int(sys.argv[1]); NROWS = 5000
OUT = ROOT / "results/Q5" / f"occlusion_fold{F}.json"
if OUT.exists():
    print("già fatto"); sys.exit(0)
data, dev = P.load_dev()
net, mu, sd, cfg = C.load_fold_model("results/Q1/main", 0, F)
rng = np.random.default_rng(42)
cand = np.where((dev.fold.values == F) & dev.lvef_known.values)[0]
idx = np.sort(rng.choice(cand, min(NROWS, len(cand)), replace=False))
reader = P.WaveReader(data, dev, leads=cfg["leads"])
X = reader.get(idx)  # (n,2500,12)
g = dev.patient_key.values[idx]
tg = {"shd": P.TARGETS.index("shd_moderate_or_greater_flag"), "lvef45": P.TARGETS.index("lvef_lte_45_flag")}
Y = {k: dev[P.TARGETS[j]].values[idx].astype(int) for k, j in tg.items()}
dv = C._device()


def pred(Xm):
    outs = []
    with torch.no_grad():
        for a in range(0, len(Xm), 256):
            x = torch.from_numpy(Xm[a:a + 256]).permute(0, 2, 1).contiguous().to(dv)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dv == "cuda"):
                o = net(x)
            outs.append(torch.sigmoid(o.float()[:, :12]).cpu().numpy())
    return np.vstack(outs)


base = pred(X)
res = {"fold": F, "n_rows": int(len(idx)), "n_patients": int(len(np.unique(g))),
       "prev": {k: float(Y[k].mean()) for k in Y}, "base_auc": {}, "lead": {}, "window": {}}
for k, j in tg.items():
    res["base_auc"][k] = P.auc_ci_cluster(Y[k], base[:, j], g, n_boot=200)
print("base", res["base_auc"], flush=True)


def run(label, Xo, store):
    po = pred(Xo)
    store[label] = {k: P.auc_diff_ci_cluster(Y[k], po[:, j], base[:, j], g, n_boot=200) for k, j in tg.items()}
    print(label, {k: round(v[0], 4) for k, v in store[label].items()}, flush=True)


for c in range(12):
    Xo = X.copy(); Xo[:, :, c] = 0.0; run(f"ch{c}", Xo, res["lead"])
for w in range(10):
    Xo = X.copy(); Xo[:, w * 250:(w + 1) * 250, :] = 0.0; run(f"{w*0.5:.1f}-{(w+1)*0.5:.1f}s", Xo, res["window"])
OUT.write_text(json.dumps(res, indent=1)); print("OK")
