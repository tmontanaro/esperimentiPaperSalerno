"""P4 - CNN 1D multi-etichetta sulle forme d'onda (porting di legacy_notebooks/04).
Uso:  python3 experiments/P4/train.py --debug-n 2000 --epochs 2 --out results/P4/debug
      python3 experiments/P4/train.py --epochs 12 --max-epochs-this-call 3     (riprende dal checkpoint)
      python3 experiments/P4/train.py --epochs 12 --eval-only                  (valutazione + candidato)
Input: SOLO forme d'onda (+ opzionale --use-tab: ECG_FEATURES). Nessun test. Epoca finale usata (nessuna selezione sul validation).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from lib import echonext_lib as lib  # noqa: E402
from model_def import ECGNet  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--debug-n", type=int, default=None)
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--use-tab", action="store_true")
ap.add_argument("--out", default=str(ROOT / "results/P4"))
ap.add_argument("--max-epochs-this-call", type=int, default=None)
ap.add_argument("--eval-only", action="store_true")
ap.add_argument("--workers", type=int, default=6)
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
PROG = ROOT / "reports/P4_waveform.progress.md"
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
TARGETS = [lib.TARGET] + [c for c in lib.FLAG_COLS if c != lib.TARGET]

data = lib.load_data(debug_n=a.debug_n)
train, val = data.train, data.val
imp = lib.SimpleImputer(strategy="median").fit(train[lib.ECG_FEATURES].astype(float))
sc = lib.StandardScaler().fit(imp.transform(train[lib.ECG_FEATURES].astype(float)))
def tab(df): return np.clip(sc.transform(imp.transform(df[lib.ECG_FEATURES].astype(float))), -5, 5).astype(np.float32)

class ECGSet(Dataset):
    def __init__(self, split, df):
        self.path = data.wave_path(split); self.W = None
        self.rows = df.row_in_split.values; self.y = df[TARGETS].values.astype(np.float32); self.t = tab(df)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        if self.W is None: self.W = np.load(self.path, mmap_mode="r")  # un memmap per worker
        x = np.ascontiguousarray(self.W[self.rows[i], 0].T, dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(self.t[i]), torch.from_numpy(self.y[i])

ds = {"train": ECGSet("train", train), "val": ECGSet("val", val)}
NW = a.workers if dev == "cuda" else 0
dl_train = DataLoader(ds["train"], batch_size=a.batch, shuffle=True, num_workers=NW, drop_last=True, pin_memory=dev == "cuda",
                      persistent_workers=NW > 0, multiprocessing_context="fork" if NW else None, generator=torch.Generator().manual_seed(SEED))
dl_val = DataLoader(ds["val"], batch_size=256, shuffle=False, num_workers=NW,
                    multiprocessing_context="fork" if NW else None)

model = ECGNet(len(TARGETS), len(lib.ECG_FEATURES), a.use_tab).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl_train))
scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
amp = dev == "cuda"

def masked_bce(lo, y):
    m = ~torch.isnan(y); l = F.binary_cross_entropy_with_logits(lo, torch.nan_to_num(y), reduction="none")
    return (l * m).sum() / m.sum().clamp(min=1)

@torch.no_grad()
def predict():
    model.eval(); P, E = [], []
    for x, t, _ in dl_val:
        with torch.autocast(device_type=dev, dtype=torch.float16, enabled=amp):
            o, e = model(x.to(dev), t.to(dev), return_emb=True)
        P.append(torch.sigmoid(o.float()).cpu().numpy()); E.append(e.float().cpu().numpy())
    return np.vstack(P), np.vstack(E)

ck, hist_f = OUT / "ckpt.pt", OUT / "history.json"
hist, start = [], 0
if ck.exists():
    s = torch.load(ck, map_location=dev)
    model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"]); scaler.load_state_dict(s["scaler"])
    start, hist = s["epoch"], s["hist"]
    torch.set_rng_state(s["rng"].cpu()); print("ripreso dall'epoca", start)

tix = 0; yv = val[lib.TARGET].values; vm = ~np.isnan(yv)
if not a.eval_only:
    done_now = 0
    for ep in range(start, a.epochs):
        if a.max_epochs_this_call and done_now >= a.max_epochs_this_call: break
        model.train(); tl, t0 = 0.0, time.time()
        for x, t, y in dl_train:
            x, t, y = x.to(dev, non_blocking=True), t.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=amp):
                loss = masked_bce(model(x, t).float(), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); tl += loss.item()
        P, _ = predict(); auc = lib.safe_auc(yv[vm], P[vm, tix])
        hist.append(dict(epoch=ep + 1, train_loss=tl / len(dl_train), val_auroc_monitor=auc, sec=round(time.time() - t0)))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": ep + 1, "hist": hist, "rng": torch.get_rng_state()}, ck)
        hist_f.write_text(json.dumps(hist, indent=1))
        msg = f"epoca {ep + 1}/{a.epochs} loss={hist[-1]['train_loss']:.4f} val AUROC composito={auc:.4f} ({hist[-1]['sec']}s)"
        print(msg, flush=True)
        if not a.debug_n:
            PROG.write_text(PROG.read_text() + f"\n- [{time.strftime('%H:%M')}] {msg} (checkpoint {ck})" if PROG.exists() else msg)
        done_now += 1

if len(hist) < a.epochs:
    print(f"incompleto: {len(hist)}/{a.epochs} epoche; rilanciare lo stesso comando"); sys.exit(0)

# ---------------- valutazione (epoca finale, nessuna selezione) ----------------
Pva, Eva = predict()
torch.save(model.state_dict(), OUT / "weights.pt")
np.savez_compressed(OUT / "val_pred_emb.npz", pred=Pva, emb=Eva, row_in_split=val.row_in_split.values, targets=np.array(TARGETS))
vg = lib.add_groups(val)
rows = []
for j, tg in enumerate(TARGETS):
    y = val[tg].values; m = ~np.isnan(y)
    if y[m].sum() < 10 or (m.sum() - y[m].sum()) < 10: continue
    Xtr, ytr, _ = lib.get_xy(train, tg)
    tp = lib.model_zoo()["hist_gb"].fit(Xtr, ytr).predict_proba(val.loc[m, lib.ECG_FEATURES].astype(float))[:, 1]
    c = lib.metrics_binary(y[m], Pva[m, j]); tb = lib.metrics_binary(y[m], tp)
    d = np.random.default_rng(0); rr = []
    for _ in range(300):  # IC bootstrap della differenza CNN - tabulare (stesse righe)
        i = d.integers(0, m.sum(), m.sum()); yy = y[m][i]
        if len(np.unique(yy)) == 2: rr.append(lib.safe_auc(yy, Pva[m, j][i]) - lib.safe_auc(yy, tp[i]))
    rows.append({"target": tg, "n": c["n"], "n_pos": c["n_pos"], "auroc_cnn": c["auroc"], "cnn_lo": c["auroc_lo"], "cnn_hi": c["auroc_hi"],
                 "auprc_cnn": c["auprc"], "auroc_tab": tb["auroc"], "tab_lo": tb["auroc_lo"], "tab_hi": tb["auroc_hi"],
                 "gain": c["auroc"] - tb["auroc"], "gain_lo": float(np.percentile(rr, 2.5)), "gain_hi": float(np.percentile(rr, 97.5))})
pd.DataFrame(rows).to_csv(OUT / "val_per_target.csv", index=False)
rows = []
tp_main = lib.model_zoo()["hist_gb"].fit(*lib.get_xy(train, lib.TARGET)[:2]).predict_proba(val.loc[vm, lib.ECG_FEATURES].astype(float))[:, 1]
vgm = vg[vm]
for g in ["<100", "100-119", ">=120"]:
    k = (vgm.qrs_bin == g).values
    if k.sum() > 50 and len(np.unique(yv[vm][k])) == 2:
        c = lib.metrics_binary(yv[vm][k], Pva[vm, tix][k]); tb = lib.metrics_binary(yv[vm][k], tp_main[k])
        rows.append({"qrs": g, "n": c["n"], "n_pos": c["n_pos"], "auroc_cnn": c["auroc"], "cnn_lo": c["auroc_lo"], "cnn_hi": c["auroc_hi"],
                     "auroc_tab": tb["auroc"], "tab_lo": tb["auroc_lo"], "tab_hi": tb["auroc_hi"], "gain": c["auroc"] - tb["auroc"]})
pd.DataFrame(rows).to_csv(OUT / "val_by_qrs.csv", index=False)
print(pd.read_csv(OUT / "val_per_target.csv").round(3).to_string()); print(pd.read_csv(OUT / "val_by_qrs.csv").round(3).to_string())

# ---------------- candidato ----------------
if not a.debug_n:
    cand_dir = ROOT / "results/P4/candidates/cnn_multilabel"
    src = '''import os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
_D = Path(__file__).parent
sys.path.insert(0, str(_D))
from model_def import ECGNet  # definizione della rete accanto a questo file
_CFG = __import__("json").loads((_D / "config.json").read_text())
_NET = None
def predict(df, lib):
    global _NET
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if _NET is None:
        _NET = ECGNet(_CFG["n_out"], _CFG["n_tab"], False).to(dev)
        _NET.load_state_dict(torch.load(_D / "weights.pt", map_location=dev)); _NET.eval()
    data = lib.load_data(allow_test=os.environ.get("ECHONEXT_ALLOW_TEST") == "1")
    out = np.zeros(len(df), dtype=np.float32)
    for split in df["split"].unique():
        idx = np.where((df["split"] == split).values)[0]
        for s in range(0, len(idx), 256):
            b = idx[s:s + 256]
            x = torch.from_numpy(np.ascontiguousarray(data.waveforms(split, df.row_in_split.values[b]).transpose(0, 2, 1), dtype=np.float32)).to(dev)
            t = torch.zeros(len(b), _CFG["n_tab"], device=dev)
            with torch.no_grad(), torch.autocast(device_type=dev, dtype=torch.float16, enabled=dev == "cuda"):
                out[b] = torch.sigmoid(_NET(x, t).float()[:, 0]).cpu().numpy()
    return pd.DataFrame({"score": out, "in_scope": True}, index=df.index)
'''
    pt = pd.read_csv(OUT / "val_per_target.csv").iloc[0]
    assert a.use_tab is False, "il candidato supporta solo il modello senza feature tabulari"
    lib.save_candidate(cand_dir, "cnn_multilabel", lib.TARGET, "model", "ResNet 1D multi-etichetta su forme d'onda, epoca finale",
                       {k: float(pt[k]) for k in ["auroc_cnn", "cnn_lo", "cnn_hi"]}, src)
    import shutil
    shutil.copy(OUT / "weights.pt", cand_dir / "weights.pt"); shutil.copy(Path(__file__).parent / "model_def.py", cand_dir / "model_def.py")
    (cand_dir / "config.json").write_text(json.dumps({"n_out": len(TARGETS), "n_tab": len(lib.ECG_FEATURES), "targets": TARGETS}))
    print("candidato salvato in", cand_dir)
