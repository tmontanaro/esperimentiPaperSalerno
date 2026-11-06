"""Funzioni comuni della PIPELINE 2 (fasi Q0-Q9). Si appoggia a lib/echonext_lib.py senza modificarlo.

Principi:
  * il TEST resta chiuso (qui non si carica mai): ogni scelta si fa con la cross-validation sul pool "dev" = train + validation;
  * le fold sono per PAZIENTE e dipendono solo da patient_key (hash), quindi sono identiche in ogni script e in ogni ripresa;
  * gli intervalli di confidenza sono bootstrap per PAZIENTE (i molti ECG dello stesso paziente non sono indipendenti).
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))
import echonext_lib as lib  # noqa: E402

TARGETS = list(lib.FLAG_COLS)            # 12 etichette binarie; la prima è il composito SHD
MAIN = lib.TARGET
FEATURES = list(lib.ECG_FEATURES)
N_FOLDS = 5
FOLD_SEED = 42


def fold_of(patient_key, n_folds: int = N_FOLDS, seed: int = FOLD_SEED) -> int:
    return zlib.crc32(f"{seed}-{patient_key}".encode()) % n_folds


def load_dev(data_dir=None, debug_n: int | None = None, n_folds: int = N_FOLDS):
    """Pool di sviluppo = train + validation (mai il test). Ritorna (data, dev) con dev indicizzato 0..n-1.

    Colonne aggiunte: fold (per paziente), lvef_known (True se lvef_value è presente)."""
    data = lib.load_data(data_dir, debug_n=debug_n)
    dev = pd.concat([data.train, data.val], ignore_index=True)
    dev["fold"] = [fold_of(k, n_folds) for k in dev.patient_key.values]
    dev["lvef_known"] = dev["lvef_value"].notna()
    dev = lib.add_groups(dev)
    return data, dev


class WaveReader:
    """Legge le forme d'onda del pool dev (train+val) con mmap; opzionale caricamento in RAM (float16) delle sole derivazioni scelte."""

    def __init__(self, data, dev: pd.DataFrame, leads=None, preload: bool = False):
        self.data, self.dev = data, dev
        self.leads = list(range(12)) if leads is None else list(leads)
        self.maps = {s: np.load(data.wave_path(s), mmap_mode="r") for s in ("train", "val")
                     if (dev.split == s).any()}
        self.split = dev.split.values
        self.row = dev.row_in_split.values.astype(np.int64)
        self.cache = None
        if preload:
            n = len(dev)
            self.cache = np.empty((n, 2500, len(self.leads)), dtype=np.float16)
            for s, W in self.maps.items():
                ix = np.where(self.split == s)[0]
                order = ix[np.argsort(self.row[ix])]
                for a in range(0, len(order), 2048):
                    b = order[a:a + 2048]
                    x = np.asarray(W[self.row[b], 0])[:, :, self.leads]
                    self.cache[b] = x.astype(np.float16)

    def get(self, idx) -> np.ndarray:
        """idx = posizioni nel DataFrame dev. Ritorna (n, 2500, n_leads) float32 nello stesso ordine di idx."""
        idx = np.asarray(idx)
        if self.cache is not None:
            return self.cache[idx].astype(np.float32)
        out = np.empty((len(idx), 2500, len(self.leads)), dtype=np.float32)
        for s, W in self.maps.items():
            m = np.where(self.split[idx] == s)[0]
            if len(m) == 0:
                continue
            order = m[np.argsort(self.row[idx[m]])]
            out[order] = np.asarray(W[self.row[idx[order]], 0])[:, :, self.leads]
        return out


# --------------------------------------------------------------------------- statistica
def _cluster_index(groups):
    g = pd.factorize(np.asarray(groups))[0]
    order = np.argsort(g, kind="stable")
    counts = np.bincount(g)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return order, starts, counts


def _boot_idx(order, starts, counts, rng):
    sel = rng.integers(0, len(counts), len(counts))
    lens = counts[sel]
    total = int(lens.sum())
    pos = np.repeat(starts[sel] - np.cumsum(lens) + lens, lens) + np.arange(total)
    return order[pos]


def auc_ci_cluster(y, p, groups, n_boot: int = 300, seed: int = 0):
    """AUROC con IC95% bootstrap per paziente. Ritorna (auc, lo, hi)."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    ok = ~np.isnan(p)
    y, p, groups = y[ok], p[ok], np.asarray(groups)[ok]
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    base = lib.safe_auc(y, p)
    order, starts, counts = _cluster_index(groups)
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n_boot):
        i = _boot_idx(order, starts, counts, rng)
        a = lib.safe_auc(y[i], p[i])
        if not np.isnan(a):
            v.append(a)
    return base, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def auc_diff_ci_cluster(y, p1, p2, groups, n_boot: int = 300, seed: int = 0):
    """Differenza AUROC(p1) - AUROC(p2) sugli stessi pazienti, IC95% bootstrap appaiato per paziente."""
    y = np.asarray(y).astype(int)
    p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    ok = ~(np.isnan(p1) | np.isnan(p2))
    y, p1, p2, groups = y[ok], p1[ok], p2[ok], np.asarray(groups)[ok]
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    base = lib.safe_auc(y, p1) - lib.safe_auc(y, p2)
    order, starts, counts = _cluster_index(groups)
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n_boot):
        i = _boot_idx(order, starts, counts, rng)
        if len(np.unique(y[i])) == 2:
            v.append(lib.safe_auc(y[i], p1[i]) - lib.safe_auc(y[i], p2[i]))
    return base, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def boot_stat_cluster(y, p, groups, fn, n_boot: int = 300, seed: int = 0):
    """IC95% bootstrap per paziente di una statistica qualsiasi fn(y, p). Ritorna (valore, lo, hi)."""
    y, p, groups = np.asarray(y), np.asarray(p, dtype=float), np.asarray(groups)
    base = float(fn(y, p))
    order, starts, counts = _cluster_index(groups)
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n_boot):
        i = _boot_idx(order, starts, counts, rng)
        try:
            x = float(fn(y[i], p[i]))
        except Exception:
            continue
        if not np.isnan(x):
            v.append(x)
    return base, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def reliability(y, p, bins: int = 10) -> pd.DataFrame:
    """Tabella di calibrazione (bin a quantili): probabilità media prevista vs frequenza osservata."""
    y, p = np.asarray(y).astype(float), np.asarray(p, dtype=float)
    q = pd.qcut(p, bins, duplicates="drop")
    d = pd.DataFrame({"y": y, "p": p, "bin": q})
    g = d.groupby("bin", observed=True).agg(n=("y", "size"), prevista=("p", "mean"), osservata=("y", "mean")).reset_index(drop=True)
    return g


def ece(y, p, bins: int = 10) -> float:
    g = reliability(y, p, bins)
    return float((g.n * (g.prevista - g.osservata).abs()).sum() / g.n.sum())


def brier(y, p) -> float:
    y, p = np.asarray(y).astype(float), np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Correzione di Holm per confronti multipli. Ritorna {nome: {p, p_adj, significativo}}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        prev = adj
        out[k] = {"p": float(p), "p_adj": float(adj), "significativo": bool(adj < alpha)}
    return out


def sens_spec_at(y, p, thr: float) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    pred = p >= thr
    tp, fp = int(((pred) & (y == 1)).sum()), int(((pred) & (y == 0)).sum())
    fn, tn = int(((~pred) & (y == 1)).sum()), int(((~pred) & (y == 0)).sum())
    return {"soglia": float(thr), "sens": tp / max(tp + fn, 1), "spec": tn / max(tn + fp, 1),
            "ppv": tp / max(tp + fp, 1), "npv": tn / max(tn + fn, 1), "n_pos_previsti": tp + fp}
