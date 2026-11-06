"""Q7: contratto dei candidati, controllo indipendente. Su N righe del pool dev (train+val, mai test):
(a) predict(df, lib) del candidato vs ricostruzione propria (pesi caricati qui, input letti via np.load mmap delle forme d'onda);
(b) correlazione con le predizioni OOF di Q1 (stessi ECG) per verificare che l'uscita sia quella giusta (non un'altra colonna).
Le righe sono nel training del final-fit: l'AUROC qui è ottimistica e serve solo come sanity check. Solo aggregati."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib")); sys.path.insert(0, str(ROOT / "tools"))
import echonext_lib as lib  # noqa: E402
from cnn_cv import make_net  # noqa: E402  (solo la definizione dell'architettura)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
data = lib.load_data()
dev = pd.concat([data.train, data.val], ignore_index=True)
rng = np.random.default_rng(42)
sel = np.sort(rng.choice(len(dev), N, replace=False))
df = dev.iloc[sel]
E = np.load(ROOT / "results/Q1/main/oof_ensemble.npy")[sel]
targets = list(lib.FLAG_COLS)
F = json.loads((ROOT / "agentic/FROZEN2.json").read_text())
maps = {s: np.load(data.wave_path(s), mmap_mode="r") for s in ("train", "val")}
dv = "cuda" if torch.cuda.is_available() else "cpu"
out = {}
for c in F["candidates"]:
    spec = importlib.util.spec_from_file_location("cand_" + c["name"], ROOT / c["dir"] / "candidate.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    pr = mod.predict(df, lib)
    meta = json.loads((ROOT / c["weights_dir"] / "config.json").read_text())
    j = targets.index(c["target"]); leads = meta["leads"]
    X = np.stack([np.asarray(maps[s][r, 0])[:, leads] for s, r in zip(df.split.values, df.row_in_split.values.astype(int))]).astype(np.float32)
    x = torch.from_numpy(X).permute(0, 2, 1).contiguous().to(dv)
    ps = []
    for s in meta["seeds"]:
        net = make_net(len(leads), 13).to(dv); net.load_state_dict(torch.load(ROOT / c["weights_dir"] / f"weights_seed{s}.pt", map_location=dv)); net.eval()
        with torch.no_grad():
            ps.append(np.concatenate([torch.sigmoid(net(x[a:a + 256])[:, j]).float().cpu().numpy() for a in range(0, len(x), 256)]))
    mine = np.mean(ps, 0)
    y = df[c["target"]].values; ok = ~pd.isna(y) & df.lvef_value.notna().values
    corr_same = float(np.corrcoef(pr.score.values, E[:, j])[0, 1])
    corr_other = {targets[k][:12]: round(float(np.corrcoef(pr.score.values, E[:, k])[0, 1]), 3) for k in range(12) if k != j}
    out[c["name"]] = {
        "shape_ok": bool(len(pr) == N and set(pr.columns) >= {"score", "in_scope"} and pr.index.equals(df.index)),
        "score_in_01": bool(((pr.score >= 0) & (pr.score <= 1)).all()), "nan": int(pr.score.isna().sum()),
        "max_abs_diff_vs_mine": float(np.max(np.abs(pr.score.values - mine))),
        "corr_with_oof_same_target": corr_same, "max_corr_with_oof_other_target": max(corr_other.values()),
        "auc_in_sample_clean(optimistic)": float(roc_auc_score(y[ok].astype(int), pr.score.values[ok])) if y[ok].sum() > 10 else None,
        "auc_oof_same_rows": float(roc_auc_score(y[ok].astype(int), E[ok, j])) if y[ok].sum() > 10 else None,
        "n_pos_clean": int(y[ok].sum()),
    }
    print(c["name"], json.dumps(out[c["name"]]), flush=True)
(ROOT / "results/Q7/check_contract.json").write_text(json.dumps(out, indent=1))
