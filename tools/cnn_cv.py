#!/usr/bin/env python3
"""CNN 1D sulle forme d'onda con cross-validation per paziente su train+validation (il test non viene mai toccato).

Sottocomandi
  run         addestra la CNN in CV (fold x seed), salva le predizioni "fuori dal fold" (OOF) e gli embedding.
              È RIPRENDIBILE: ogni coppia (seed, fold) è un job con il suo file; i job già fatti si saltano.
              Con --time-budget-min si ferma in modo ordinato (codice di uscita 3) e basta rilanciare lo stesso comando.
  summarize   (gli embedding NON si assemblano tra fold: ogni fold ha la sua rete e il suo spazio)
              AUROC (IC95% bootstrap per paziente) per le 12 etichette + regressione LVEF, confronto col modello tabulare
              sugli stessi pazienti e sulle stesse fold, popolazioni «clean» (LVEF noto) e «all».
  final-fit   addestra sul pool dev intero (seed multipli) e salva i pesi, per costruire i candidati del secondo test.
  make-candidate  scrive candidate.json/candidate.py (contratto di agentic/PLAYBOOK.md) a partire da un final-fit.

Esempio
  python3 tools/cnn_cv.py run --out results/Q1/main --seeds 0,1,2 --epochs 12 --time-budget-min 7
  python3 tools/cnn_cv.py summarize --out results/Q1/main

Codici di uscita: 0 completo · 3 parziale (rilancia lo stesso comando) · 2 errore.
Derivazioni: ordine ASSUNTO standard (I, II, III, aVR, aVL, aVF, V1..V6); NON è documentato da PhysioNet: lo verifica la fase Q0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))
import pipe2_lib as P2  # noqa: E402

lib = P2.lib

LEADSETS = {
    "all": list(range(12)), "limb": [0, 1, 2, 3, 4, 5], "chest": [6, 7, 8, 9, 10, 11],
    "I": [0], "II": [1], "I_II": [0, 1], "eight": [0, 1, 6, 7, 8, 9, 10, 11], "I_II_V1": [0, 1, 6],
}
N_BIN = len(P2.TARGETS)  # 12 uscite binarie + 1 regressione (LVEF) = 13


def parse_leads(s: str) -> list[int]:
    if s in LEADSETS:
        return LEADSETS[s]
    return [int(x) for x in s.split(",")]


# --------------------------------------------------------------------------- modello
def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def make_net(n_in: int, n_out: int):
    torch, nn, F = _torch()

    class Block(nn.Module):
        def __init__(s, cin, cout, stride):
            super().__init__()
            s.c1 = nn.Conv1d(cin, cout, 7, stride, 3, bias=False); s.b1 = nn.BatchNorm1d(cout)
            s.c2 = nn.Conv1d(cout, cout, 7, 1, 3, bias=False); s.b2 = nn.BatchNorm1d(cout)
            s.sk = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout)) if (stride != 1 or cin != cout) else nn.Identity()

        def forward(s, x):
            h = F.relu(s.b1(s.c1(x))); h = s.b2(s.c2(h)); return F.relu(h + s.sk(x))

    class ECGNet(nn.Module):
        def __init__(s):
            super().__init__()
            s.stem = nn.Sequential(nn.Conv1d(n_in, 32, 15, 2, 7, bias=False), nn.BatchNorm1d(32), nn.ReLU())
            s.body = nn.Sequential(Block(32, 32, 2), Block(32, 64, 2), Block(64, 128, 2), Block(128, 128, 2))
            s.drop = nn.Dropout(0.3); s.head = nn.Linear(256, n_out)

        def forward(s, x, return_emb=False):
            h = s.body(s.stem(x)); z = torch.cat([h.mean(-1), h.amax(-1)], 1)
            o = s.head(s.drop(z))
            return (o, z) if return_emb else o

    return ECGNet()


def _device():
    torch, _, _ = _torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_targets(dev: pd.DataFrame) -> np.ndarray:
    Y = np.full((len(dev), N_BIN + 1), np.nan, dtype=np.float32)
    Y[:, :N_BIN] = dev[P2.TARGETS].astype(float).values
    Y[:, N_BIN] = dev["lvef_value"].astype(float).values
    return Y


def train_net(reader, Y, tr_idx, cfg: dict, seed: int, beat=None):
    """Addestra una CNN su tr_idx. Ritorna (rete, mu_lvef, sd_lvef)."""
    torch, nn, F = _torch()
    dev_ = _device()
    torch.manual_seed(seed); np.random.seed(seed)
    if dev_ == "cuda":
        torch.backends.cudnn.benchmark = True
    Ytr = Y[tr_idx].copy()
    lv = Ytr[:, N_BIN]
    mu, sd = float(np.nanmean(lv)), float(np.nanstd(lv) + 1e-6)
    Y2 = Y.copy(); Y2[:, N_BIN] = (Y2[:, N_BIN] - mu) / sd
    net = make_net(len(reader.leads), N_BIN + 1).to(dev_)
    B, E = cfg["batch"], cfg["epochs"]
    nb = max(1, len(tr_idx) // B)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=E * nb)
    rng = np.random.default_rng(seed)
    for ep in range(E):
        net.train(); perm = rng.permutation(tr_idx); t0 = time.time(); tl = 0.0
        for s in range(nb):
            b = np.sort(perm[s * B:(s + 1) * B])
            x = torch.from_numpy(reader.get(b)).permute(0, 2, 1).contiguous().to(dev_)
            y = torch.from_numpy(Y2[b]).to(dev_)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev_ == "cuda"):
                out = net(x)
            out = out.float()
            m = ~torch.isnan(y)
            yb = torch.nan_to_num(y)
            bce = (F.binary_cross_entropy_with_logits(out[:, :N_BIN], yb[:, :N_BIN], reduction="none") * m[:, :N_BIN]).sum() / m[:, :N_BIN].sum().clamp(min=1)
            reg = (F.smooth_l1_loss(out[:, N_BIN], yb[:, N_BIN], reduction="none") * m[:, N_BIN]).sum() / m[:, N_BIN].sum().clamp(min=1)
            loss = bce + cfg["reg_weight"] * reg
            loss.backward(); opt.step(); sched.step(); tl += float(loss.item())
        if beat:
            beat(ep + 1, E, tl / nb, time.time() - t0)
    return net, mu, sd


def load_fold_model(out, seed: int, fold: int):
    """Carica la rete addestrata nel job (seed, fold) di una run: ritorna (net, mu, sd, cfg). Gli embedding e le predizioni di quel job
    stanno in oof_seed{seed}_fold{fold}.npz ed appartengono allo STESSO spazio: non mescolare fold diversi."""
    torch, _, _ = _torch()
    out = Path(out); out = out if out.is_absolute() else ROOT / out
    cfg = json.loads((out / "config.json").read_text())
    net = make_net(len(cfg["leads"]), N_BIN + 1).to(_device())
    net.load_state_dict(torch.load(out / f"weights_seed{seed}_fold{fold}.pt", map_location=_device()))
    net.eval()
    z = np.load(out / f"oof_seed{seed}_fold{fold}.npz")
    return net, float(z["mu"]), float(z["sd"]), cfg


def predict_net(net, reader, idx, mu, sd, want_emb=True):
    torch, _, _ = _torch()
    dev_ = _device(); net.eval()
    P, Z = [], []
    with torch.no_grad():
        for a in range(0, len(idx), 256):
            b = idx[a:a + 256]
            x = torch.from_numpy(reader.get(b)).permute(0, 2, 1).contiguous().to(dev_)
            o, z = net(x, return_emb=True)
            o = o.float().cpu().numpy()
            p = np.empty_like(o)
            p[:, :N_BIN] = 1 / (1 + np.exp(-o[:, :N_BIN]))
            p[:, N_BIN] = o[:, N_BIN] * sd + mu
            P.append(p)
            if want_emb:
                Z.append(z.float().cpu().numpy().astype(np.float16))
    return np.vstack(P), (np.vstack(Z) if want_emb else None)


# --------------------------------------------------------------------------- progresso e configurazione
class Progress:
    def __init__(self, out: Path, total: int, cfg: dict):
        self.f = out / "progress.json"
        self.d = {"status": "running", "jobs_total": total, "jobs_done": 0, "current": None, "epoch": None,
                  "pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S"), "updated": None, "job_seconds": [], "config": cfg}

    def save(self, **kw):
        self.d.update(kw); self.d["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp = self.f.with_suffix(".tmp"); tmp.write_text(json.dumps(self.d, indent=1)); os.replace(tmp, self.f)


CFG_KEYS = ("leads", "epochs", "batch", "lr", "reg_weight", "population", "folds", "debug_n")


def make_cfg(a) -> dict:
    return {"leads": parse_leads(a.leads), "epochs": a.epochs, "batch": a.batch, "lr": a.lr, "reg_weight": a.reg_weight,
            "population": a.population, "folds": a.folds, "debug_n": a.debug_n}


def check_cfg(out: Path, cfg: dict):
    f = out / "config.json"
    if f.exists():
        old = json.loads(f.read_text())
        diff = [k for k in CFG_KEYS if old.get(k) != cfg.get(k)]
        if diff:
            sys.exit(f"ERRORE: {out} contiene già una run con parametri diversi ({', '.join(diff)}). Usa un'altra cartella --out.")
    out.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(cfg, indent=1))


def _prep(a):
    data, dev = P2.load_dev(debug_n=a.debug_n, n_folds=a.folds)
    cfg = make_cfg(a)
    Y = build_targets(dev)
    clean = dev.lvef_known.values
    reader = P2.WaveReader(data, dev, leads=cfg["leads"], preload=a.preload)
    return data, dev, cfg, Y, clean, reader


def _train_idx(dev, clean, cfg, fold=None):
    m = np.ones(len(dev), dtype=bool) if fold is None else (dev.fold.values != fold)
    if cfg["population"] == "clean":
        m &= clean
    return np.where(m)[0]


# --------------------------------------------------------------------------- run
def cmd_run(a):
    out = Path(a.out); out = out if out.is_absolute() else ROOT / out
    data, dev, cfg, Y, clean, reader = _prep(a)
    check_cfg(out, cfg)
    seeds = [int(s) for s in a.seeds.split(",")]
    jobs = [(s, f) for s in seeds for f in range(a.folds)]
    todo = [(s, f) for (s, f) in jobs if not (out / f"oof_seed{s}_fold{f}.npz").exists()]
    prog = Progress(out, len(jobs), cfg)
    prog.save(jobs_done=len(jobs) - len(todo))
    print(f"[cnn_cv] {out.relative_to(ROOT)}: {len(jobs)} job, già fatti {len(jobs) - len(todo)}, dispositivo {_device()}, derivazioni {cfg['leads']}", flush=True)
    t_start, ran = time.time(), 0
    for (s, f) in todo:
        est = float(np.mean(prog.d["job_seconds"])) if prog.d["job_seconds"] else 0.0
        if a.time_budget_min and ran > 0 and (time.time() - t_start + est) > a.time_budget_min * 60:
            prog.save(status="partial", current=None)
            print(f"STATO: PARZIALE ({len(jobs) - len(todo) + ran}/{len(jobs)} job). Rilancia lo stesso comando per continuare.", flush=True)
            return 3
        t0 = time.time()
        prog.save(current=f"seed{s}_fold{f}")

        def beat(ep, E, loss, secs, s=s, f=f):
            prog.save(epoch=f"{ep}/{E}")
            print(f"  seed{s} fold{f} epoca {ep}/{E} loss {loss:.4f} ({secs:.0f}s)", flush=True)
        tr = _train_idx(dev, clean, cfg, fold=f)
        te = np.where(dev.fold.values == f)[0]
        net, mu, sd = train_net(reader, Y, tr, cfg, seed=s, beat=beat)
        P, Z = predict_net(net, reader, te, mu, sd)
        torch, _, _ = _torch()
        torch.save(net.state_dict(), out / f"weights_seed{s}_fold{f}.pt")
        np.savez_compressed(out / f"oof_seed{s}_fold{f}.npz", idx=te, preds=P.astype(np.float32), emb=Z, mu=mu, sd=sd, n_train=len(tr))
        ran += 1
        prog.d["job_seconds"].append(time.time() - t0)
        prog.save(jobs_done=len(jobs) - len(todo) + ran)
        print(f"[cnn_cv] job seed{s} fold{f} finito in {time.time() - t0:.0f}s ({len(jobs) - len(todo) + ran}/{len(jobs)})", flush=True)
    prog.save(status="done", current=None)
    print("STATO: COMPLETO", flush=True)
    return 0


# --------------------------------------------------------------------------- summarize
def _tabular_oof(dev, clean, population, cache: Path):
    """OOF hist_gb sulle 7 feature ECG, stesse fold; 12 classificazioni + regressione LVEF."""
    if cache.exists():
        z = np.load(cache)
        if z["P"].shape[0] == len(dev):
            return z["P"]
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    X = dev[P2.FEATURES].astype(float)
    P = np.full((len(dev), N_BIN + 1), np.nan, dtype=np.float32)
    for f in sorted(dev.fold.unique()):
        te = np.where(dev.fold.values == f)[0]
        base = (dev.fold.values != f) & (clean if population == "clean" else True)
        for j, t in enumerate(P2.TARGETS):
            m = base & dev[t].notna().values
            if len(np.unique(dev.loc[m, t])) < 2:
                continue
            clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=250, random_state=lib.SEED)
            clf.fit(X[m], dev.loc[m, t].astype(int))
            P[te, j] = clf.predict_proba(X.iloc[te])[:, 1]
        m = base & dev.lvef_value.notna().values
        reg = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=250, random_state=lib.SEED)
        reg.fit(X[m], dev.loc[m, "lvef_value"].values)
        P[te, N_BIN] = reg.predict(X.iloc[te])
    np.savez_compressed(cache, P=P)
    return P


def cmd_summarize(a):
    out = Path(a.out); out = out if out.is_absolute() else ROOT / out
    cfg = json.loads((out / "config.json").read_text())
    data, dev = P2.load_dev(debug_n=cfg["debug_n"], n_folds=cfg["folds"])
    seeds = sorted({int(p.name.split("_")[1][4:]) for p in out.glob("oof_seed*_fold*.npz")})
    full = [s for s in seeds if all((out / f"oof_seed{s}_fold{f}.npz").exists() for f in range(cfg["folds"]))]
    if not full:
        sys.exit("Nessun seed completo: finisci prima `run`.")
    oof = {}
    for s in full:
        P = np.full((len(dev), N_BIN + 1), np.nan, dtype=np.float32)
        for f in range(cfg["folds"]):
            z = np.load(out / f"oof_seed{s}_fold{f}.npz")
            P[z["idx"]] = z["preds"]
        oof[s] = P
        np.save(out / f"oof_seed{s}.npy", P)
    ens = np.nanmean(np.stack([oof[s] for s in full]), axis=0)
    np.save(out / "oof_ensemble.npy", ens)
    tab = _tabular_oof(dev, dev.lvef_known.values, cfg["population"], out / "tabular_oof.npz")
    g = dev.patient_key.values
    rows, summary = [], {"seeds": full, "config": cfg, "populations": {}}
    for pop, mask in (("clean", dev.lvef_known.values), ("all", np.ones(len(dev), dtype=bool))):
        rows = []
        for j, t in enumerate(P2.TARGETS):
            m = mask & dev[t].notna().values
            y = dev.loc[m, t].astype(int).values
            if len(np.unique(y)) < 2 or y.sum() < 20:
                continue
            gm = g[m]
            auc, lo, hi = P2.auc_ci_cluster(y, ens[m, j], gm, n_boot=a.boot)
            per_seed = [lib.safe_auc(y, oof[s][m, j]) for s in full]
            tauc, tlo, thi = P2.auc_ci_cluster(y, tab[m, j], gm, n_boot=a.boot)
            d, dlo, dhi = P2.auc_diff_ci_cluster(y, ens[m, j], tab[m, j], gm, n_boot=a.boot)
            rows.append({"target": t.replace("_moderate_or_greater_flag", "").replace("_flag", ""), "n": int(m.sum()), "n_pos": int(y.sum()),
                         "cnn_auroc": auc, "cnn_lo": lo, "cnn_hi": hi, "seed_mean": float(np.mean(per_seed)), "seed_sd": float(np.std(per_seed)),
                         "seed_min": float(np.min(per_seed)), "seed_max": float(np.max(per_seed)),
                         "tab_auroc": tauc, "tab_lo": tlo, "tab_hi": thi, "diff": d, "diff_lo": dlo, "diff_hi": dhi})
        T = pd.DataFrame(rows)
        T.to_csv(out / f"metrics_{pop}.csv", index=False)
        mr = mask & dev.lvef_value.notna().values
        yv = dev.loc[mr, "lvef_value"].values
        reg = {}
        for nm, pr in (("cnn", ens[mr, N_BIN]), ("tab", tab[mr, N_BIN])):
            mae = P2.boot_stat_cluster(yv, pr, g[mr], lambda y, p: np.mean(np.abs(y - p)), n_boot=min(a.boot, 200))
            r2 = P2.boot_stat_cluster(yv, pr, g[mr], lambda y, p: 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2), n_boot=min(a.boot, 200))
            reg[nm] = {"mae": mae, "r2": r2}
        summary["populations"][pop] = {"n_rows": int(mask.sum()), "lvef_regression": reg, "table": rows}
        print(f"\n== popolazione «{pop}» (n={int(mask.sum())}): AUROC fuori-fold, ensemble di {len(full)} seed, IC95% per paziente ==")
        lib.show(T.round(3))
        print(f"LVEF continua (n={int(mr.sum())}): CNN MAE {reg['cnn']['mae'][0]:.2f} R2 {reg['cnn']['r2'][0]:.3f} | tabulare MAE {reg['tab']['mae'][0]:.2f} R2 {reg['tab']['r2'][0]:.3f}")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"\nScritti: metrics_clean.csv, metrics_all.csv, summary.json, oof_ensemble.npy, oof_seed*.npy in {out.relative_to(ROOT)}")
    return 0


# --------------------------------------------------------------------------- final-fit e candidati
def cmd_final_fit(a):
    out = Path(a.out); out = out if out.is_absolute() else ROOT / out
    data, dev, cfg, Y, clean, reader = _prep(a)
    out.mkdir(parents=True, exist_ok=True)
    cf = out / "config.json"
    if cf.exists():
        old = json.loads(cf.read_text())
        diff = [k for k in CFG_KEYS if old.get(k) != cfg.get(k)]
        if diff:
            sys.exit(f"ERRORE: {out} contiene già un final-fit con parametri diversi ({', '.join(diff)}).")
    seeds = [int(s) for s in a.seeds.split(",")]
    tr = _train_idx(dev, clean, cfg, fold=None)
    t_start, ran = time.time(), 0
    meta = json.loads(cf.read_text()) if cf.exists() else {**cfg, "targets": P2.TARGETS, "n_bin": N_BIN, "seeds": [], "mu_sd": {}}
    for s in seeds:
        wf = out / f"weights_seed{s}.pt"
        if wf.exists():
            continue
        if a.time_budget_min and ran > 0 and (time.time() - t_start) > a.time_budget_min * 60 * 0.6:
            cf.write_text(json.dumps(meta, indent=1))
            print("STATO: PARZIALE. Rilancia lo stesso comando per continuare.", flush=True)
            return 3

        def beat(ep, E, loss, secs, s=s):
            print(f"  final-fit seed{s} epoca {ep}/{E} loss {loss:.4f} ({secs:.0f}s)", flush=True)
        net, mu, sd = train_net(reader, Y, tr, cfg, seed=s, beat=beat)
        torch, _, _ = _torch()
        torch.save(net.state_dict(), wf)
        meta["seeds"] = sorted(set(meta["seeds"]) | {s}); meta["mu_sd"][str(s)] = [mu, sd]
        cf.write_text(json.dumps(meta, indent=1))
        ran += 1
    cf.write_text(json.dumps(meta, indent=1))
    print("STATO: COMPLETO", flush=True)
    return 0


CAND_PY = '''"""Candidato CNN generato da tools/cnn_cv.py make-candidate. Contratto: predict(df, lib) -> DataFrame[score, in_scope]."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "tools" / "cnn_cv.py").exists():
    ROOT = ROOT.parent
WEIGHTS = ROOT / "{weights}"
TARGET = "{target}"


def predict(df, lib):
    sys.path.insert(0, str(ROOT / "tools"))
    import cnn_cv
    return cnn_cv.candidate_predict(df, lib, WEIGHTS, TARGET)
'''


def candidate_predict(df, lib_, wdir, target):
    """Usata dai candidati: ensemble dei seed salvati, uscita binaria `target`. Legge le forme d'onda dello split di ogni riga."""
    torch, _, _ = _torch()
    wdir = Path(wdir)
    meta = json.loads((wdir / "config.json").read_text())
    j = meta["targets"].index(target)
    dev_ = _device()
    splits = set(df["split"].unique())
    data = lib_.load_data(allow_test=("test" in splits))
    leads = meta["leads"]
    nets = []
    for s in meta["seeds"]:
        net = make_net(len(leads), N_BIN + 1).to(dev_)
        net.load_state_dict(torch.load(wdir / f"weights_seed{s}.pt", map_location=dev_))
        net.eval(); nets.append(net)
    score = np.zeros(len(df))
    split_v, row_v = df["split"].values, df["row_in_split"].values.astype(int)
    with torch.no_grad():
        for a0 in range(0, len(df), 256):
            sl = slice(a0, a0 + 256)
            xs = []
            for sp, r in zip(split_v[sl], row_v[sl]):
                xs.append(data.waveforms(sp, [r])[0][:, leads])
            x = torch.from_numpy(np.asarray(xs, dtype=np.float32)).permute(0, 2, 1).contiguous().to(dev_)
            p = np.mean([torch.sigmoid(n(x)[:, j]).float().cpu().numpy() for n in nets], axis=0)
            score[sl] = p
    return pd.DataFrame({"score": score, "in_scope": True}, index=df.index)


def cmd_make_candidate(a):
    wd = Path(a.weights); wd = wd if wd.is_absolute() else ROOT / wd
    cd = Path(a.cand_out); cd = cd if cd.is_absolute() else ROOT / cd
    if not (wd / "config.json").exists():
        sys.exit("final-fit non trovato in " + str(wd))
    cd.mkdir(parents=True, exist_ok=True)
    (cd / "candidate.py").write_text(CAND_PY.format(weights=str(wd.relative_to(ROOT)), target=a.target))
    (cd / "candidate.json").write_text(json.dumps({
        "name": a.name, "target": a.target, "kind": "cnn_waveform", "description": a.description or f"CNN forme d'onda, uscita {a.target}",
        "val_metrics": {"auroc": a.cv_auroc, "note": "AUROC fuori-fold in CV su train+validation (non validation singolo)"}}, indent=2))
    print("Candidato scritto in", cd.relative_to(ROOT))
    return 0


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--out", required=True, help="cartella dei risultati, es. results/Q1/main")
        s.add_argument("--leads", default="all", help="all|limb|chest|I|II|I_II|eight|I_II_V1 oppure indici separati da virgola (0-11)")
        s.add_argument("--epochs", type=int, default=12)
        s.add_argument("--batch", type=int, default=128)
        s.add_argument("--lr", type=float, default=2e-3)
        s.add_argument("--reg-weight", type=float, default=0.5)
        s.add_argument("--population", choices=["clean", "all"], default="clean",
                       help="clean = addestra solo su righe con lvef_value noto (default); all = tutte le righe")
        s.add_argument("--folds", type=int, default=P2.N_FOLDS)
        s.add_argument("--debug-n", type=int, default=None, help="prova veloce: prime N righe per split")
        s.add_argument("--preload", action="store_true", help="carica in RAM (float16) le derivazioni scelte: più veloce, serve memoria")
        s.add_argument("--time-budget-min", type=float, default=0, help="esci in modo ordinato (codice 3) dopo circa N minuti; rilancia per continuare")
    s = sp.add_parser("run"); common(s); s.add_argument("--seeds", default="0,1,2"); s.set_defaults(fn=cmd_run)
    s = sp.add_parser("summarize"); s.add_argument("--out", required=True); s.add_argument("--boot", type=int, default=300)
    s.set_defaults(fn=cmd_summarize)
    s = sp.add_parser("final-fit"); common(s); s.add_argument("--seeds", default="0,1,2"); s.set_defaults(fn=cmd_final_fit)
    s = sp.add_parser("make-candidate"); s.add_argument("--weights", required=True); s.add_argument("--target", required=True)
    s.add_argument("--name", required=True); s.add_argument("--cand-out", required=True); s.add_argument("--description")
    s.add_argument("--cv-auroc", type=float, default=None); s.set_defaults(fn=cmd_make_candidate)
    a = ap.parse_args()
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
