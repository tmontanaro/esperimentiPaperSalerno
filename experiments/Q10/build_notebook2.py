"""Q10: costruisce reports/Q10_notebook/replica_completa.ipynb partendo dal notebook P9 (cap. 0-8, pipeline 1)
e aggiungendo i capitoli della pipeline 2 (Q0-Q8). Rieseguibile: python3 experiments/Q10/build_notebook2.py
Non modifica il notebook di P9 (lo legge soltanto). Nessun output salvato nel notebook generato. Non carica il test."""
import ast, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "reports/P9_notebook/replica_analisi.ipynb"
OUTD = ROOT / "reports/Q10_notebook"
OUTD.mkdir(parents=True, exist_ok=True)

p9 = json.loads(SRC.read_text())
txt = lambda c: "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
P9 = [(c["cell_type"], txt(c)) for c in p9["cells"]]
assert len(P9) == 40

cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

# ------------------------------------------------------------------ intestazione (nuova) + P9 cap. 0-8
md(r"""
# Replica completa dell'analisi EchoNext (pipeline 1 + pipeline 2)

**Che cosa è**: un notebook autonomo con cui rifare, capitolo per capitolo, **tutte** le analisi: pipeline 1 (cap. 1-8: audit dati → baseline → condizioni → LVEF e QRS → forme d'onda → cluster → controlli → test finale) e pipeline 2 (cap. 9-17: protocollo e verifica delle derivazioni → CNN in cross-validation → ablazione delle derivazioni → LVEF e QRS con la CNN → embedding, cluster e sottogruppi → calibrazione e interpretabilità → candidati, revisione, secondo test → sintesi).
**Fonte dei numeri «attesi»**: i report in `reports/` (P0…P8, Q0…Q9); ogni numero è accompagnato dal report di origine.

**Parametri** (cella 0 · Preparazione): `DATA_DIR` (cartella dei dati), `QUICK` (prova veloce: campione piccolo, 1 seed, 2 epoche, solo per verificare che il codice giri), `SEED`, `RUN_HEAVY` (addestra le CNN, permutazioni, occlusione; richiede `torch`, meglio con GPU).

**Tempi**: vedere la tabella nel capitolo 17 (misurati in una prova, GPU RTX 5080). Senza `RUN_HEAVY` il notebook gira in pochi minuti ma i capitoli 11-15 mostrano solo i numeri dei report. **La replica completa delle CNN (Q1: 15 addestramenti, Q2: 7 insiemi × 5 fold) richiede ore**: non è stata rieseguita per intero in questo notebook (vedi «Cosa non è replicabile»).

**Promemoria importante**: *SHD* (cardiopatia strutturale di grado almeno moderato) **non coincide con scompenso cardiaco**: è un'etichetta ricavata dall'ecocardiogramma.
Il **test set resta chiuso**: questo notebook non lo carica (le righe di test vengono scartate subito dopo la lettura) e non lo valuta. Le tabelle dei due test (P7 e Q8) sono **testo copiato dai report**.
""")
for i in range(1, 39):
    t, s = P9[i]
    if i == 1:
        s = s.replace("## Come leggere questo notebook", "## Come leggere questo notebook (valido per tutti i capitoli, anche 9-17)")
    if i == 3:
        s = s.replace("# ----------------------------- PARAMETRI (modificare qui) -----------------------------",
                      "# ----------------------------- PARAMETRI (modificare qui) -----------------------------\n# (QUICK/RUN_HEAVY valgono anche per i capitoli 9-17 della pipeline 2)")
        s = s.replace("# True = addestra la CNN (cap. 5) e fa le permutazioni dei cluster (cap. 6)", "# True = addestra le CNN (cap. 5, 10-14), permutazioni (cap. 6, 13), occlusione (cap. 14)")
    if i == 38:
        s = s.replace("## 8 · Test finale, sintesi e limiti (fasi P7-P8)", "## 8 · Test finale della pipeline 1, sintesi e limiti (fasi P7-P8)")
        s = s.replace("Sintesi completa, glossario e prossimi passi: `reports/P8_synthesis.md` (e `reports/FINAL_REPORT.md`).",
                      "Sintesi completa, glossario e prossimi passi: `reports/P8_synthesis.md` (e `reports/FINAL_REPORT.md`).\n- La pipeline 2 (cap. 9-17) ha risolto parte di questi limiti (più seed, CV per la CNN, calibrazione), ne ha aggiunti altri (due letture del test).")
    cells.append((t, s))

# ------------------------------------------------------------------ cap. 9 · Q0
md(r"""
## 9 · Pipeline 2 — protocollo comune e verifica dell'ordine delle derivazioni (fase Q0)
**Obiettivo / perché**: la pipeline 2 fa i test che restano possibili con gli *stessi* dati. Regole del protocollo:
1. **Pool di sviluppo = train + validation**; ogni scelta si fa con la **cross-validation (CV) per paziente a 5 fold**, con il fold calcolato da un hash di `patient_key` (così un paziente sta sempre in un solo fold, in ogni script e in ogni ripresa).
2. **Popolazione «clean»** = righe con `lvef_value` noto (circa 91%). Le righe senza LVEF hanno tutte le etichette a 0 e PhysioNet dichiara che le eco senza LVEF sono state escluse: sono anomale. Il risultato principale è sul *clean*; *all* è una sensibilità.
3. Ogni numero ha un **IC95% con bootstrap per paziente**; i confronti multipli sono corretti con **Holm**.
4. Il test è chiuso. Le derivazioni sono nei canali 0-11; l'ordine standard (I, II, III, aVR, aVL, aVF, V1-V6) **non è documentato** da PhysioNet: qui si verifica con le relazioni fisiche tra le derivazioni degli arti (II = I + III; aVR = −(I+II)/2; aVL = I − II/2; aVF = II − I/2). Le toraciche non sono verificabili.

Questa cella costruisce il pool di sviluppo, le fold e gli strumenti statistici (copiati da `lib/pipe2_lib.py`: fold per paziente, bootstrap per paziente, Holm, ECE).
""")
code(r'''
# ---------------- Pool di sviluppo, fold per paziente, strumenti statistici (da lib/pipe2_lib.py) ----------------
import zlib, json
N_FOLDS, FOLD_SEED = 5, 42
def fold_of(patient_key, n_folds=N_FOLDS, seed=FOLD_SEED):
    """Fold di un paziente = hash del suo patient_key: identico in ogni script."""
    return zlib.crc32(f"{seed}-{patient_key}".encode()) % n_folds

DEV = pd.concat([train, val], ignore_index=True)           # pool di sviluppo (il test non c'è: scartato in load_data)
if QUICK:                                                   # sottoinsieme di PAZIENTI, solo per la prova di funzionamento
    pk_q = np.random.default_rng(SEED).choice(DEV.patient_key.unique(), 1500, replace=False)
    DEV = DEV[DEV.patient_key.isin(pk_q)].reset_index(drop=True)
DEV["fold"] = [fold_of(k) for k in DEV.patient_key.values]
DEV["lvef_known"] = DEV.lvef_value.notna()
DEV = add_groups(DEV)
CLEAN = DEV.lvef_known.values
QT = list(FLAG_COLS)                                        # 12 etichette, la prima è SHD
print(f"pool di sviluppo: {len(DEV)} ECG, {DEV.patient_key.nunique()} pazienti | clean (LVEF noto): {CLEAN.sum()} ({CLEAN.mean():.3f})")
print("pazienti per fold:", DEV.groupby("fold").patient_key.nunique().to_dict())
assert DEV.groupby("patient_key").fold.nunique().max() == 1, "un paziente è in più fold!"
CHECKS_PRE = [("Q0 N ECG pool dev (solo QUICK=False)", 77101, len(DEV)), ("Q0 pazienti pool dev", 30844, DEV.patient_key.nunique()), ("Q0 frazione clean", 0.910, CLEAN.mean())]

def _cluster_index(groups):
    g = pd.factorize(np.asarray(groups))[0]
    order = np.argsort(g, kind="stable"); counts = np.bincount(g)
    return order, np.concatenate([[0], np.cumsum(counts)[:-1]]), counts
def _boot_idx(order, starts, counts, rng):
    """Un campione bootstrap che ricampiona i PAZIENTI (e tutti i loro ECG)."""
    sel = rng.integers(0, len(counts), len(counts)); lens = counts[sel]; total = int(lens.sum())
    pos = np.repeat(starts[sel] - np.cumsum(lens) + lens, lens) + np.arange(total)
    return order[pos]
def auc_ci_cluster(y, p, groups, n_boot=None, seed=0):
    n_boot = n_boot or NBOOT
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float); ok = ~np.isnan(p)
    y, p, groups = y[ok], p[ok], np.asarray(groups)[ok]
    if len(np.unique(y)) < 2: return float("nan"), float("nan"), float("nan")
    order, starts, counts = _cluster_index(groups); rng = np.random.default_rng(seed); v = []
    for _ in range(n_boot):
        i = _boot_idx(order, starts, counts, rng); a = safe_auc(y[i], p[i])
        if not np.isnan(a): v.append(a)
    return safe_auc(y, p), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
def auc_diff_ci_cluster(y, p1, p2, groups, n_boot=None, seed=0):
    """AUROC(p1) - AUROC(p2) sugli stessi pazienti, IC95% bootstrap appaiato per paziente."""
    n_boot = n_boot or NBOOT
    y = np.asarray(y).astype(int); p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    ok = ~(np.isnan(p1) | np.isnan(p2)); y, p1, p2, groups = y[ok], p1[ok], p2[ok], np.asarray(groups)[ok]
    order, starts, counts = _cluster_index(groups); rng = np.random.default_rng(seed); v = []
    for _ in range(n_boot):
        i = _boot_idx(order, starts, counts, rng)
        if len(np.unique(y[i])) == 2: v.append(safe_auc(y[i], p1[i]) - safe_auc(y[i], p2[i]))
    return safe_auc(y, p1) - safe_auc(y, p2), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
def boot_stat_cluster(y, p, groups, fn, n_boot=None, seed=0):
    n_boot = n_boot or NBOOT
    y, p, groups = np.asarray(y), np.asarray(p, float), np.asarray(groups)
    order, starts, counts = _cluster_index(groups); rng = np.random.default_rng(seed); v = []
    for _ in range(n_boot):
        i = _boot_idx(order, starts, counts, rng)
        try:
            x = float(fn(y[i], p[i]))
            if not np.isnan(x): v.append(x)
        except Exception: pass
    return float(fn(y, p)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
def reliability(y, p, bins=10):
    d = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)}); d["bin"] = pd.qcut(d.p, bins, duplicates="drop")
    return d.groupby("bin", observed=True).agg(n=("y", "size"), prevista=("p", "mean"), osservata=("y", "mean")).reset_index(drop=True)
def ece(y, p, bins=10):
    g = reliability(y, p, bins); return float((g.n * (g.prevista - g.osservata).abs()).sum() / g.n.sum())
def brier(y, p): return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))
def holm(pvals, alpha=0.05):
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p)); prev = adj
        out[k] = {"p": float(p), "p_adj": float(adj), "significativo": bool(adj < alpha)}
    return out
def classe(a, lo, n_pos):
    """Classe di accuratezza: buona = AUROC>=0.80 con IC inf>=0.75; moderata 0.70-0.80; debole <0.70; non valutabile se <100 positivi."""
    return "non valutabile" if n_pos < 100 else ("buona" if (a >= 0.8 and lo >= 0.75) else ("moderata" if a >= 0.7 else "debole"))
def fmt3(t): return f"{t[0]:.3f} [{t[1]:.3f}-{t[2]:.3f}]"
CHECKS = []   # (nome, atteso dal report, ottenuto qui)
CHECKS += CHECKS_PRE
''')
md(r"""
### Verifica dell'ordine delle derivazioni
Su ~2000 ECG di train (500 con `QUICK`) si regredisce ogni derivazione degli arti sulle altre. Ipotesi standard: canale1 ≈ a·c0 + b·c2 (a, b > 0); canale3 ≈ a·c0 + b·c1 (a, b < 0); canale4 ≈ a·c0 + b·c2 (a > 0, b < 0); canale5 ≈ a·c1 + b·c2 (a, b > 0). `LEAD_ORDER_OK = yes` se tutte e 4 hanno R² ≥ 0.95 e segni attesi.
""")
code(r'''
from sklearn.linear_model import LinearRegression
Wtr_ = np.load(WAVE_DIR / "EchoNext_train_waveforms.npy", mmap_mode="r")
n_chk = 500 if QUICK else 2000
rows_ = np.sort(np.random.default_rng(SEED).choice(len(Wtr_), n_chk, replace=False))
Xw = np.asarray(Wtr_[rows_, 0], dtype=np.float32)[:, ::5, :].reshape(-1, 12)       # (campioni, 12 canali), sottocampionati
rel = {"c1 ~ c0, c2": (1, [0, 2], (+1, +1)), "c3 ~ c0, c1": (3, [0, 1], (-1, -1)),
       "c4 ~ c0, c2": (4, [0, 2], (+1, -1)), "c5 ~ c1, c2": (5, [1, 2], (+1, +1))}
rec, ok_all = [], True
for nome, (tg_, pr_, segni) in rel.items():
    lr_ = LinearRegression().fit(Xw[:, pr_], Xw[:, tg_]); r2_ = lr_.score(Xw[:, pr_], Xw[:, tg_])
    sg_ = tuple(int(np.sign(c)) for c in lr_.coef_); ok_ = (r2_ >= 0.95) and sg_ == segni; ok_all &= ok_
    rec.append({"relazione": nome, "R2": r2_, "coef": np.round(lr_.coef_, 2).tolist(), "segni attesi": segni, "ok": ok_})
show(pd.DataFrame(rec))
LEAD_ORDER_OK = "yes" if ok_all else "partial/no"
print("LEAD_ORDER_OK =", LEAD_ORDER_OK, "(le derivazioni toraciche, canali 6-11, non sono verificabili con questo metodo)")
print("Atteso (reports/Q0_inventory.md): R2 c1 0.976, c3 0.994, c4 0.995, c5 0.996, LEAD_ORDER_OK = yes")
CHECKS.append(("Q0 R2 aVF (c5)", 0.996, rec[3]["R2"]))
del Xw
''')
md(r"""
### Risultato che abbiamo ottenuto (fonte: `reports/Q0_inventory.md`, handoff Q0)
- `LEAD_ORDER_OK = yes`: canali 0-5 = derivazioni degli arti nell'ordine atteso (R² 0.976 / 0.994 / 0.995 / 0.996, segni attesi). Le toraciche non sono verificabili: nei report non si nominano.
- Pool di sviluppo: **77 101 ECG, 30 844 pazienti**; popolazione clean 70 130 ECG (**91.0%**). Pesi della CNN, `test_results.json`, candidati P1/P3/P4 e registro presenti.
- Riferimenti pipeline 1: SHD CNN test 0.830 [0.820-0.841], tabulare 0.762 [0.750-0.774].
- Tempi: 9.5 s per epoca con GPU; stime: Q1 ≈ 30 min, Q2 ≈ 70 min (limite superiore), Q6 ≈ 8 min.

### Decisione
Si procede con CV per paziente sulla popolazione clean (cella sopra: nessun paziente in due fold). I nomi delle derivazioni degli arti sono ammessi; per le toraciche si usano solo gli indici di canale (nei report un insieme come «I_II_V1» contiene il canale 6, il cui nome non è verificato).
""")

# ------------------------------------------------------------------ cap. 10 · Q1
md(r"""
## 10 · CNN in cross-validation (fase Q1)
**Obiettivo / perché**: nella pipeline 1 la CNN era stata addestrata una volta sola (validation 0.823). Qui si misura la prestazione con **CV per paziente (5 fold) e 3 seed**, sulla popolazione clean, e la si confronta con il **modello tabulare** (hist_gb sulle 7 feature) sulle *stesse* righe e fold. Per ogni etichetta si dà la classe di accuratezza (buona / moderata / debole).
**Scelte**: CNN 1D residua (4 blocchi), 12 derivazioni, 12 epoche, lr 0.002 (OneCycle), batch 128, peso della regressione LVEF 0.5 (uscite: 12 etichette + LVEF continua), addestramento sulle sole righe clean dei fold di sviluppo; le predizioni «fuori dal fold» (OOF) delle 3 seed si mediano (*ensemble*).
Le celle seguenti definiscono la rete (copiata da `tools/cnn_cv.py`) e il lettore delle forme d'onda: **richiedono `RUN_HEAVY=True`** e `torch`.
""")
code(r'''
# ---------------- CNN, lettore di forme d'onda, CV (da tools/cnn_cv.py) — solo con RUN_HEAVY ----------------
CFG = dict(epochs=2 if QUICK else 12, lr=2e-3, batch=128, reg_weight=0.5)
SEEDS = [0] if QUICK else [0, 1, 2]
LEADSETS = {"all": list(range(12)), "eight": [0, 1, 6, 7, 8, 9, 10, 11], "limb": [0, 1, 2, 3, 4, 5], "chest": [6, 7, 8, 9, 10, 11],
            "I_II": [0, 1], "II": [1], "I": [0], "I_II_V1": [0, 1, 6]}
NB_BIN = 12
OOF1 = EMB1 = MODELS1 = None
if not RUN_HEAVY:
    print("RUN_HEAVY=False: le CNN non vengono addestrate; i capitoli 10-14 mostrano i numeri dei report.")
else:
    import torch, torch.nn as nn, torch.nn.functional as F
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("dispositivo:", DEVICE, "| config:", CFG, "| seed:", SEEDS)
    class WaveReader:
        """Legge le forme d'onda del pool DEV (train+val) con mmap; ritorna (n, 2500, n_leads) float32 nell'ordine di idx."""
        def __init__(self, dev, leads):
            self.leads = list(leads); self.split = dev.split.values; self.row = dev.row_in_split.values.astype(np.int64)
            self.maps = {s: np.load(WAVE_DIR / f"EchoNext_{s}_waveforms.npy", mmap_mode="r") for s in ("train", "val") if (self.split == s).any()}
        def get(self, idx):
            idx = np.asarray(idx); out = np.empty((len(idx), 2500, len(self.leads)), dtype=np.float32)
            for s, W in self.maps.items():
                m = np.where(self.split[idx] == s)[0]
                if len(m) == 0: continue
                o = m[np.argsort(self.row[idx[m]])]
                out[o] = np.asarray(W[self.row[idx[o]], 0])[:, :, self.leads]
            return out
    def make_net(n_in, n_out):
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
                h = s.body(s.stem(x)); z = torch.cat([h.mean(-1), h.amax(-1)], 1); o = s.head(s.drop(z))
                return (o, z) if return_emb else o
        return ECGNet()
    Y_ALL = np.full((len(DEV), NB_BIN + 1), np.nan, dtype=np.float32)
    Y_ALL[:, :NB_BIN] = DEV[QT].astype(float).values; Y_ALL[:, NB_BIN] = DEV["lvef_value"].astype(float).values
    def train_net(reader, tr_idx, seed, cfg=CFG):
        torch.manual_seed(seed); np.random.seed(seed)
        lv = Y_ALL[tr_idx, NB_BIN]; mu, sd = float(np.nanmean(lv)), float(np.nanstd(lv) + 1e-6)
        Y2 = Y_ALL.copy(); Y2[:, NB_BIN] = (Y2[:, NB_BIN] - mu) / sd
        net = make_net(len(reader.leads), NB_BIN + 1).to(DEVICE); B, E = cfg["batch"], cfg["epochs"]; nb = max(1, len(tr_idx) // B)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=E * nb); rng = np.random.default_rng(seed)
        for ep in range(E):
            net.train(); perm = rng.permutation(tr_idx)
            for s in range(nb):
                b = np.sort(perm[s * B:(s + 1) * B])
                x = torch.from_numpy(reader.get(b)).permute(0, 2, 1).contiguous().to(DEVICE); y = torch.from_numpy(Y2[b]).to(DEVICE)
                opt.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                    out = net(x)
                out = out.float(); m = ~torch.isnan(y); yb = torch.nan_to_num(y)
                bce = (F.binary_cross_entropy_with_logits(out[:, :NB_BIN], yb[:, :NB_BIN], reduction="none") * m[:, :NB_BIN]).sum() / m[:, :NB_BIN].sum().clamp(min=1)
                reg = (F.smooth_l1_loss(out[:, NB_BIN], yb[:, NB_BIN], reduction="none") * m[:, NB_BIN]).sum() / m[:, NB_BIN].sum().clamp(min=1)
                (bce + cfg["reg_weight"] * reg).backward(); opt.step(); sched.step()
        return net, mu, sd
    def predict_net(net, reader, idx, mu, sd, want_emb=True):
        net.eval(); P, Z = [], []
        with torch.no_grad():
            for a in range(0, len(idx), 256):
                x = torch.from_numpy(reader.get(idx[a:a + 256])).permute(0, 2, 1).contiguous().to(DEVICE)
                o, z = net(x, return_emb=True); o = o.float().cpu().numpy(); p = np.empty_like(o)
                p[:, :NB_BIN] = 1 / (1 + np.exp(-o[:, :NB_BIN])); p[:, NB_BIN] = o[:, NB_BIN] * sd + mu
                P.append(p); Z.append(z.float().cpu().numpy().astype(np.float16)) if want_emb else None
        return np.vstack(P), (np.vstack(Z) if want_emb else None)
    def run_cnn_cv(leads, seeds, keep=False):
        """CV per paziente: per ogni (seed, fold) addestra sulle righe CLEAN degli altri fold, predice TUTTE le righe del fold."""
        reader = WaveReader(DEV, leads); oof = {s: np.full((len(DEV), NB_BIN + 1), np.nan, np.float32) for s in seeds}; emb, models = {}, {}
        for s in seeds:
            for f in range(N_FOLDS):
                t0 = time.time(); tr = np.where((DEV.fold.values != f) & CLEAN)[0]; te = np.where(DEV.fold.values == f)[0]
                net, mu, sd = train_net(reader, tr, s); P, Z = predict_net(net, reader, te, mu, sd); oof[s][te] = P
                if keep and s == seeds[0]: emb[f] = (te, Z); models[f] = (net, mu, sd)
                print(f"  seed {s} fold {f}: {time.time() - t0:.0f}s", flush=True)
        return oof, emb, models
    t0 = time.time(); OOF1, EMB1, MODELS1 = run_cnn_cv(LEADSETS["all"], SEEDS, keep=True)
    print("CNN CV completata in %.0f s" % (time.time() - t0))
''')
code(r'''
# ---------------- Modello tabulare in CV (stesse fold, stessa popolazione clean) e confronto con la CNN ----------------
TAB = np.full((len(DEV), 2), np.nan)       # colonne: SHD, LVEF<=45
for f in range(N_FOLDS):
    tr_, te_ = (DEV.fold.values != f) & CLEAN, (DEV.fold.values == f)
    for j, tg in enumerate([T, "lvef_lte_45_flag"]):
        m_ = new_hgb().fit(DEV.loc[tr_, ECG_FEATURES].astype(float), DEV.loc[tr_, tg].astype(int))
        TAB[te_, j] = m_.predict_proba(DEV.loc[te_, ECG_FEATURES].astype(float))[:, 1]
dc = DEV[CLEAN].reset_index(drop=True); g_c = dc.patient_key.values
tab_shd = auc_ci_cluster(dc[T], TAB[CLEAN, 0], g_c); tab_lv = auc_ci_cluster(dc["lvef_lte_45_flag"], TAB[CLEAN, 1], g_c)
print("TABULARE clean, SHD:", fmt3(tab_shd), "| LVEF<=45%:", fmt3(tab_lv))
print("Atteso (reports/Q1_cnn-cv.md, campione completo): SHD 0.712 [0.705-0.721]; LVEF<=45% 0.720 [0.710-0.730]")
CHECKS.append(("Q1 tabulare SHD clean (solo QUICK=False)", 0.712, tab_shd[0]))
if OOF1 is None:
    print("\nRisultati attesi per la CNN (Q1, 3 seed, clean): SHD 0.815 [0.809-0.822] (seed singoli 0.808-0.809, SD < 0.001); differenza vs tabulare +0.103 [0.096-0.109];")
    print("LVEF<=45% 0.869; VD 0.844; stenosi aortica 0.811; LVEF continua: CNN MAE 7.48 / R2 0.482 contro tabulare MAE 10.45 / R2 0.156.")
else:
    ENS = np.mean([OOF1[s] for s in SEEDS], axis=0)          # ensemble dei seed (tutte le righe)
    rows = []
    for j, tg in enumerate(QT):
        y_ = dc[tg].values.astype(int); a = auc_ci_cluster(y_, ENS[CLEAN, j], g_c)
        rows.append({"target": tg.replace("_moderate_or_greater_flag", "").replace("_flag", ""), "n_pos": int(y_.sum()), "auroc": a[0], "lo": a[1], "hi": a[2], "classe": classe(a[0], a[1], int(y_.sum()))})
    Q1TAB = pd.DataFrame(rows); show(Q1TAB)
    shd_c = (Q1TAB.auroc[0], Q1TAB.lo[0], Q1TAB.hi[0])
    dif = auc_diff_ci_cluster(dc[T], ENS[CLEAN, 0], TAB[CLEAN, 0], g_c); print("SHD: CNN - tabulare:", fmt3(dif))
    dif2 = auc_diff_ci_cluster(dc["lvef_lte_45_flag"], ENS[CLEAN, 1], TAB[CLEAN, 1], g_c); print("LVEF<=45%: CNN - tabulare:", fmt3(dif2))
    lv_ = dc.lvef_value.values; est = ENS[CLEAN, NB_BIN]
    print("LVEF continua, CNN: MAE %.2f, R2 %.3f" % (mean_absolute_error(lv_, est), r2_score(lv_, est)))
    CHECKS.append(("Q1 CNN SHD clean (QUICK: solo confronto di plausibilità)", 0.815, shd_c[0]))
    CHECKS.append(("Q1 CNN LVEF<=45% clean", 0.869, Q1TAB.auroc[1]))
    CHECKS.append(("Q1 differenza CNN-tabulare SHD", 0.103, dif[0]))
''')
md(r"""
### Risultato che abbiamo ottenuto (fonte: `reports/Q1_cnn-cv.md`, handoff Q1)
- **CNN SHD (clean, 3 seed): 0.815 [0.809-0.822]**; variabilità tra seed trascurabile (SD < 0.001; seed singoli 0.808-0.809). Tabulare sulle stesse righe: **0.712 [0.705-0.721]**; differenza **+0.103 [0.096-0.109]**.
- Classi: **buone** SHD, LVEF ≤ 45% (0.869), disfunzione sistolica del VD (0.844), stenosi aortica (0.811); **moderate** spessore parietale ≥ 13, rigurgito mitralico, tricuspidale, polmonare, versamento pericardico, PASP ≥ 45, TR max ≥ 32; **debole** rigurgito aortico (0.692 [0.652-0.732]).
- LVEF continua: CNN MAE 7.48 / R² 0.482 contro tabulare MAE 10.45 / R² 0.156. Stenosi aortica: il **tabulare è migliore** della CNN (0.841 contro 0.811).
- Sensibilità alle righe senza LVEF (addestrare anche su di esse): ΔAUROC SHD +0.0020 [0.0001-0.0038], LVEF ≤ 45% +0.0026 [0.0003-0.0048]: trascurabile.

### Decisione
Nessuna variante (epoche, lr, peso della regressione) ha battuto la configurazione principale con la soglia pre-specificata (+0.01 su entrambe e IC appaiato che esclude 0): **configurazione finale** = quella sopra (12 epoche, lr 0.002, peso 0.5, 12 derivazioni, popolazione clean). Le fasi successive usano queste predizioni. Con `QUICK=True` i numeri sono molto più bassi (campione e addestramento ridotti): servono solo a verificare il codice.
""")

# ------------------------------------------------------------------ cap. 11 · Q2
md(r"""
## 11 · Ablazione delle derivazioni (fase Q2)
**Obiettivo / perché**: l'idea originale era «togliere input finché funziona». Si riaddestra la CNN (stessa configurazione, seed 0, 5 fold) usando solo alcune derivazioni e si guarda di quanto cala l'AUROC rispetto a tutte e 12. **Regola pre-specificata**: un insieme è «sufficiente» se la differenza puntuale è ≥ −0.01 **sia** per SHD sia per LVEF ≤ 45%; si riporta il più piccolo.
Insiemi: `all` (12), `eight` (8), `limb` (arti, 6), `chest` (6 canali toracici), `I_II`, `II`, `I`, `I_II_V1` (3: canali 0, 1 e 6; il nome della toracica non è verificato). Dietro `RUN_HEAVY` (con `QUICK` solo `all` e `I_II_V1`).
""")
code(r'''
def minimal_sufficient(tab, thr=-0.01):
    """tab: DataFrame con colonne set, n_leads, d_shd, d_lvef (differenze vs 'all'). Ritorna il più piccolo insieme con entrambe >= thr."""
    ok = tab[(tab.set != "all") & (tab.d_shd >= thr) & (tab.d_lvef >= thr)]
    return None if ok.empty else ok.sort_values(["n_leads", "set"]).iloc[0].set

# (a) Numeri del report (reports/Q2_lead-ablation.md; results/Q2/ablation_table.csv): differenza dall'insieme completo, seed 0, clean
EXP_ABL = pd.DataFrame([("eight", 8, +0.002, +0.000), ("chest", 6, -0.002, -0.007), ("I_II_V1", 3, -0.006, -0.008), ("limb", 6, -0.016, -0.011),
                        ("I_II", 2, -0.017, -0.014), ("I", 1, -0.024, -0.024), ("II", 1, -0.038, -0.044)], columns=["set", "n_leads", "d_shd", "d_lvef"])
print("REGOLA applicata ai numeri del report ->", "insieme minimo sufficiente:", minimal_sufficient(EXP_ABL))
show(EXP_ABL)
if OOF1 is not None:
    sets = ["I_II_V1"] if QUICK else ["eight", "limb", "chest", "I_II", "II", "I", "I_II_V1"]
    base0 = OOF1[0]; rows = []
    for nm in sets:
        o, _, _ = run_cnn_cv(LEADSETS[nm], [0]); o = o[0]
        r = {"set": nm, "n_leads": len(LEADSETS[nm])}
        for lab, j, tg in (("shd", 0, T), ("lvef", 1, "lvef_lte_45_flag")):
            d_ = auc_diff_ci_cluster(dc[tg], o[CLEAN, j], base0[CLEAN, j], g_c)
            r[f"auroc_{lab}"] = safe_auc(dc[tg], o[CLEAN, j]); r[f"d_{lab}"] = d_[0]; r[f"d_{lab}_lo"], r[f"d_{lab}_hi"] = d_[1], d_[2]
        rows.append(r)
    ABL = pd.DataFrame(rows); show(ABL.round(3)); print("\nricalcolato qui -> insieme minimo sufficiente:", minimal_sufficient(ABL))
    if "I_II_V1" in ABL.set.values: CHECKS.append(("Q2 Δ SHD con I_II_V1", -0.006, float(ABL[ABL.set == "I_II_V1"].d_shd.iloc[0])))
''')
md(r"""
### Risultato che abbiamo ottenuto (fonte: `reports/Q2_lead-ablation.md`, handoff Q2; tabella `results/Q2/ablation_table.csv`)
- Riferimento (tutte, seed 0, clean): SHD 0.808 [0.801-0.814], LVEF ≤ 45% 0.862 [0.855-0.869].
- **Insieme minimo sufficiente = 3 derivazioni (I_II_V1)**: SHD 0.802 [0.795-0.809], Δ −0.006 [−0.009, −0.003]; LVEF ≤ 45% 0.854 [0.846-0.862], Δ −0.008 [−0.011, −0.004].
- Calo per insieme (Δ SHD / Δ LVEF): `eight` +0.002/+0.000; `chest` −0.002/−0.007; `I_II_V1` −0.006/−0.008; `limb` −0.016/−0.011; `I_II` −0.017/−0.014; `I` −0.024/−0.024; `II` −0.038/−0.044.
- Per stenosi aortica e disfunzione del VD, `I_II_V1` perde −0.009 e −0.013 (oltre la soglia per il VD).

### Decisione
«Sufficiente» significa *perdita ≤ 0.01 su un seed*, **non** assenza di perdita (la piccola perdita è reale: l'IC esclude 0). Si congela un candidato con 3 derivazioni per SHD (C5) accanto a quello con tutte; `eight` è il set senza perdita. Cautela: con una sola seed e senza correzione per confronti multipli la gerarchia fine tra insiemi vicini non va sovra-interpretata.
""")

# ------------------------------------------------------------------ cap. 12 · Q3
md(r"""
## 12 · LVEF e durata del QRS con la CNN (fase Q3)
**Obiettivo / perché**: i medici ipotizzano che il QRS modifichi la relazione tra le misure ECG e la LVEF. Nella pipeline 1 il tabulare era troppo debole per concludere (inconclusivo). Qui si ripete il test con un predittore molto migliore (la CNN): **se il QRS contasse ancora, dovrebbe aggiungere informazione sopra la CNN**.
**Scelte** (clean, predizioni OOF, stesse fold): (a) AUROC LVEF ≤ 45% per fascia di QRS (<100, 100-119, ≥120), CNN e tabulare, con differenza appaiata; (b) regressione logistica `y ~ logit(cnn)` contro `y ~ logit(cnn)+QRS+QRS×logit(cnn)` in CV per paziente: ΔAUROC e Δlog-loss; (c) residuo LVEF (vera − stimata dalla CNN): correlazione di Spearman col QRS e ΔR² aggiungendo QRS e interazione; (d) coppie discordanti (stessa LVEF stimata ±1, LVEF vera che differisce di ≥ 10 punti; un ECG per paziente): differenza di QRS. **Holm** per famiglia. **Verdetto**: `supportata` se (b) ΔAUROC con IC inf > 0 dopo Holm **e** (c) ΔR² ≥ 0.01 con IC inf > 0; `non_supportata` se in (b) e (c) l'IC sup < 0.01; altrimenti `inconclusiva`.
""")
code(r'''
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
def verdetto_qrs(b_auc_hi, b_auc_lo_holm_sig, c_r2_hi, c_r2_lo):
    """Regola pre-specificata del verdetto sul QRS (PLAYBOOK2, Q3)."""
    if b_auc_lo_holm_sig and c_r2_lo > 0 and c_r2_hi >= 0.01: return "supportata"
    if b_auc_hi < 0.01 and c_r2_hi < 0.01: return "non_supportata"
    return "inconclusiva"
print("REGOLA applicata ai numeri del report (ΔAUROC IC sup +0.0006, ΔR² IC sup +0.0010) ->", verdetto_qrs(0.0006, False, 0.0010, -0.0002))
if OOF1 is None:
    print("RUN_HEAVY=False: numeri del report -> vedi sotto.")
else:
    NBQ = 50 if QUICK else 300
    jL = QT.index("lvef_lte_45_flag"); yL = dc.lvef_known.values.astype(int) * 0 + (dc.lvef_value.values <= 45).astype(int)
    pc = np.clip(ENS[CLEAN, jL], 1e-6, 1 - 1e-6); pt = np.clip(TAB[CLEAN, 1], 1e-6, 1 - 1e-6); lc = np.log(pc / (1 - pc))
    ml, tl = ENS[CLEAN, NB_BIN], dc.lvef_value.values; qrs = dc.qrs_duration.values.astype(float); fold_c = dc.fold.values
    ok = ~np.isnan(qrs) & ~np.isnan(pc) & ~np.isnan(pt) & ~np.isnan(ml)
    lo_, hi_ = np.nanpercentile(qrs, [1, 99]); qz = (np.clip(qrs, lo_, hi_) - np.nanmean(qrs)) / np.nanstd(qrs)
    order, starts, counts = _cluster_index(g_c)
    def boot(fn, mask=None, B=NBQ):
        """bootstrap per paziente di fn(indici) -> statistiche; ritorna (valore, lo, hi, p) per ciascuna"""
        base = np.atleast_1d(fn(np.where(mask)[0] if mask is not None else np.arange(len(dc)))); r = np.random.default_rng(0); v = []
        for _ in range(B):
            i = _boot_idx(order, starts, counts, r)
            if mask is not None: i = i[mask[i]]
            try: v.append(np.atleast_1d(fn(i)))
            except Exception: pass
        v = np.array(v); res = []
        for k in range(len(base)):
            a = v[:, k]; a = a[~np.isnan(a)]; p = max(2 * min((a <= 0).mean(), (a >= 0).mean()), 1 / (len(a) + 1))
            res.append((float(base[k]), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), float(p)))
        return res
    # (a) per fascia di QRS
    band = np.asarray(pd.cut(qrs, [0, 100, 120, 1e9], labels=["<100", "100-119", ">=120"], right=False)); pa = {}; rows = []
    for b in ["<100", "100-119", ">=120"]:
        m = (band == b) & ok
        c = auc_ci_cluster(yL[m], pc[m], g_c[m]); t = auc_ci_cluster(yL[m], pt[m], g_c[m])
        dd = boot(lambda i: safe_auc(yL[i], pc[i]) - safe_auc(yL[i], pt[i]), m)[0]; pa["a_" + b] = dd[3]
        rows.append({"fascia QRS": b, "n": int(m.sum()), "pos": int(yL[m].sum()), "CNN": fmt3(c), "tabulare": fmt3(t), "diff": f"{dd[0]:+.3f} [{dd[1]:+.3f}, {dd[2]:+.3f}]"})
    show(pd.DataFrame(rows)); ha = holm(pa)
    # (b) valore aggiunto del QRS sopra la CNN (CV per paziente)
    def cv_logit(X):
        o = np.full(len(dc), np.nan)
        for f in range(N_FOLDS):
            tr, te = (fold_c != f) & ok, (fold_c == f) & ok
            o[te] = LogisticRegression(C=1e4, max_iter=1000).fit(X[tr], yL[tr]).predict_proba(X[te])[:, 1]
        return o
    o0 = cv_logit(lc[:, None]); o1 = cv_logit(np.nan_to_num(np.c_[lc, qz, qz * lc]))
    bb = boot(lambda i: [safe_auc(yL[i], o1[i]) - safe_auc(yL[i], o0[i]), log_loss(yL[i], o0[i], labels=[0, 1]) - log_loss(yL[i], o1[i], labels=[0, 1])], ok)
    hb = holm({"delta_auc": bb[0][3], "delta_logloss": bb[1][3]})
    print("(b) ΔAUROC con QRS: %+.4f [%+.4f, %+.4f] (p Holm %.2f) | Δlog-loss %+.4f [%+.4f, %+.4f]" % (*bb[0][:3], hb["delta_auc"]["p_adj"], *bb[1][:3]))
    # (c) residuo della LVEF
    res = tl - ml
    def cv_lin(X):
        o = np.full(len(dc), np.nan)
        for f in range(N_FOLDS):
            tr, te = (fold_c != f) & ok, (fold_c == f) & ok; o[te] = LinearRegression().fit(X[tr], tl[tr]).predict(X[te])
        return o
    q0 = cv_lin(ml[:, None]); q1 = cv_lin(np.nan_to_num(np.c_[ml, qz, qz * ml]))
    cr = boot(lambda i: r2_score(tl[i], q1[i]) - r2_score(tl[i], q0[i]), ok)[0]
    sp = boot(lambda i: spearmanr(res[i], qrs[i])[0], ok)[0]
    print("(c) ΔR² con QRS: %+.4f [%+.4f, %+.4f] | Spearman residuo-QRS: %+.3f [%+.3f, %+.3f]" % (*cr[:3], *sp[:3]))
    # (d) coppie discordanti: un ECG per paziente, accoppiamento greedy
    perm = np.random.default_rng(42).permutation(len(dc)); seen = {}
    for k in perm:
        if ok[k] and g_c[k] not in seen: seen[g_c[k]] = k
    idx = np.array(list(seen.values())); srt = idx[np.argsort(ml[idx])]; mv = ml[srt]; used = np.zeros(len(srt), bool); pairs = []
    for a in np.random.default_rng(7).permutation(len(srt)):
        if used[a]: continue
        l, r_ = np.searchsorted(mv, [mv[a] - 1, mv[a] + 1]); c = np.arange(l, r_)
        c = c[(~used[c]) & (c != a) & (np.abs(tl[srt[c]] - tl[srt[a]]) >= 10) & (np.abs(mv[c] - mv[a]) < 1)]
        if len(c):
            b2 = c[np.random.default_rng(int(a)).integers(len(c))]; used[a] = used[b2] = True; pairs.append((srt[a], srt[b2]))
    if pairs:
        pairs = np.array(pairs); sw = tl[pairs[:, 0]] < tl[pairs[:, 1]]
        lowi = np.where(sw, pairs[:, 0], pairs[:, 1]); highi = np.where(sw, pairs[:, 1], pairs[:, 0]); dq = qrs[lowi] - qrs[highi]; dq = dq[~np.isnan(dq)]
        r = np.random.default_rng(0); bs = np.array([dq[r.integers(0, len(dq), len(dq))].mean() for _ in range(2000)])
        print("(d) %d coppie; QRS (LVEF bassa - alta): %+.1f ms [%+.1f, %+.1f]" % (len(dq), dq.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
    V = verdetto_qrs(bb[0][2], bool(hb["delta_auc"]["significativo"] and bb[0][1] > 0), cr[2], cr[1])
    print("VERDETTO ricalcolato qui:", V, "| 8 confronti totali nel report (3 + 2 + 2 + 1, Holm per famiglia)")
    CHECKS.append(("Q3 ΔAUROC QRS", 0.0001, bb[0][0])); CHECKS.append(("Q3 ΔR² QRS", 0.0004, cr[0]))
''')
md(r"""
### Risultato che abbiamo ottenuto (fonte: `reports/Q3_lvef-qrs-cnn.md`, handoff Q3)
- **`QRS_MODIFICA = non_supportata`.** AUROC LVEF ≤ 45% per fascia (CNN / tabulare): QRS < 100: 0.840 / 0.669; 100-119: 0.867 / 0.662; ≥ 120: 0.883 / 0.683 (la CNN è meglio ovunque, +0.17 / +0.20 / +0.20).
- Valore aggiunto del QRS sopra la CNN: **ΔAUROC +0.0001 [−0.0005, +0.0006]** (p Holm 0.79); **ΔR² +0.0004 [−0.0002, +0.0010]** (p Holm 0.46); correlazione parziale residuo-QRS +0.006 [−0.012, +0.022].
- Coppie discordanti: differenza di QRS +1.5 ms [0.9, 2.1] su 8 992 coppie (significativa ma trascurabile). 8 confronti, Holm per famiglia.

### Decisione
Ipotesi dei medici **non supportata sopra la CNN**: effetti esclusi oltre ±0.001 di AUROC. Cautela: la CNN vede già il QRS nel tracciato, quindi questo non prova che il QRS sia irrilevante, ma che non aggiunge nulla a ciò che la rete ha già estratto. Nessun candidato nuovo da questo capitolo. (La regola del verdetto è ricalcolata nella cella: sui numeri del report dà `non_supportata`.)
""")

# ------------------------------------------------------------------ cap. 13 · Q4
md(r"""
## 13 · Embedding, cluster e sottogruppi (fase Q4)
**Obiettivo / perché**: ripetere la ricerca di gruppi di pazienti con la rappresentazione *appresa dalla CNN* (embedding a 256 numeri, penultimo strato) e capire dove la CNN funziona meglio o peggio.
**Scelte**: (1) **cluster fold per fold** (gli embedding di fold diversi stanno in spazi diversi: non si mescolano): standardizzazione, PCA a 20 componenti, k-means con k ∈ {3,4,6,8} scelto con la silhouette (senza etichette), stabilità tra due metà di pazienti (ARI); AUROC *locale* della stessa predizione contro quella globale; **rumore** con permutazioni delle etichette di cluster (200; 20 con `QUICK`). Cluster «credibile» = n ≥ 100, ≥ 25 positivi e ≥ 25 negativi, AUROC locale ≥ globale + 0.02 con IC inferiore sopra il globale e p empirico < 0.05 dopo Holm. (2) **Strati pre-specificati** (età, sesso, QRS, terzili di frequenza/PR/QTc e colonne di contesto presenti), differenza appaiata dal resto, Holm. (3) **Regole sugli errori** con cross-fitting (albero profondità ≤ 3, foglia minima 300, addestrato su metà dei pazienti e verificato sull'altra).
Dietro `RUN_HEAVY` (servono gli embedding della CNN).
""")
code(r'''
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.tree import DecisionTreeRegressor
def credibile(auc_loc, auc_lo, auc_glob, p_holm, n, pos):
    """Definizione di sottogruppo/cluster credibile (PLAYBOOK §8)."""
    return bool(n >= 100 and pos >= 25 and (n - pos) >= 25 and auc_loc >= auc_glob + 0.02 and auc_lo > auc_glob and p_holm < 0.05)
print("regola di credibilità: es. locale 0.84 [0.82-0.86] contro globale 0.82, p Holm 0.01 ->", credibile(0.84, 0.82, 0.82, 0.01, 500, 200),
      "(IC inf non sopra il globale)  | es. locale 0.86 [0.84-0.88], p Holm 0.01 ->", credibile(0.86, 0.84, 0.82, 0.01, 500, 200))
if EMB1 is None:
    print("RUN_HEAVY=False: nessun embedding disponibile; risultati attesi nel testo sotto.")
else:
    NPERM = 20 if QUICK else 200; NBQ4 = 50 if QUICK else 300; rng0 = np.random.default_rng(42); crow = []
    for f in range(N_FOLDS):
        te, Z = EMB1[f]; cl_ = CLEAN[te]; idx = te[cl_]; E = Z[cl_].astype(np.float32); pr = OOF1[0][idx, 0]
        sub = DEV.iloc[idx]; y = sub[T].values.astype(int); g = sub.patient_key.values
        Xp = PCA(20, random_state=42).fit_transform((E - E.mean(0)) / (E.std(0) + 1e-6))
        sam = rng0.choice(len(Xp), min(5000, len(Xp)), replace=False); sil = {}
        for k in (3, 4, 6, 8): sil[k] = silhouette_score(Xp[sam], KMeans(k, n_init=5, random_state=42).fit_predict(Xp)[sam])
        k = max(sil, key=sil.get); lab = KMeans(k, n_init=5, random_state=42).fit(Xp).labels_
        half = np.array([zlib.crc32(f"h{x}".encode()) % 2 for x in g]) == 1
        ari = adjusted_rand_score(KMeans(k, n_init=5, random_state=1).fit(Xp[half]).predict(Xp), KMeans(k, n_init=5, random_state=2).fit(Xp[~half]).predict(Xp))
        gl = safe_auc(y, pr); elig = [c for c in range(k) if (lab == c).sum() >= 100 and 25 <= y[lab == c].sum() <= (lab == c).sum() - 25]
        mx = np.array([max([safe_auc(y[pl == c], pr[pl == c]) - gl for c in elig] or [0]) for pl in (rng0.permutation(lab) for _ in range(NPERM))])
        for c in elig:
            m = lab == c; a, lo, hi = auc_ci_cluster(y[m], pr[m], g[m], NBQ4)
            crow.append(dict(fold=f, k=k, cluster=c, n=int(m.sum()), pos=int(y[m].sum()), prev=y[m].mean(), auc_loc=a, auc_lo=lo, auc_glob=gl, diff=a - gl,
                             p_emp=float((1 + (mx >= a - gl).sum()) / (NPERM + 1)), ari=ari, perm_q95=float(np.quantile(mx, .95))))
        print(f"fold {f}: k={k}, ARI={ari:.3f}, silhouette={sil[k]:.3f}, rumore (max locale-globale): media {mx.mean():+.3f}, q95 {np.quantile(mx, .95):+.3f}", flush=True)
    CL = pd.DataFrame(crow)
    best = CL.sort_values("p_emp").groupby("fold").head(1); hh = holm({f"f{r.fold}": r.p_emp for r in best.itertuples()})
    CL["p_holm"] = [hh[f"f{r.fold}"]["p_adj"] if ((best.fold == r.fold) & (best.cluster == r.cluster)).any() else 1.0 for r in CL.itertuples()]
    CL["credibile"] = [credibile(r.auc_loc, r.auc_lo, r.auc_glob, r.p_holm, r.n, r.pos) for r in CL.itertuples()]
    show(CL.round(3)); print("cluster credibili:", int(CL.credibile.sum()), "su", len(CL), "(atteso: 0 su 15, k=3 in ogni fold, ARI 0.90-0.92)")
    CHECKS.append(("Q4 cluster credibili", 0, int(CL.credibile.sum())))
''')
code(r'''
if OOF1 is not None:
    ok4 = CLEAN & ~np.isnan(OOF1[0][:, 0]); d4 = DEV[ok4].reset_index(drop=True); e4 = ENS[ok4]; g4 = d4.patient_key.values
    terz = lambda c: pd.qcut(d4[c], 3, labels=["T1", "T2", "T3"])
    S = {"eta": pd.cut(d4.age_at_ecg, [0, 50, 65, 75, 200], right=False, labels=["<50", "50-64", "65-74", ">=75"]), "sesso": d4.sex,
         "QRS": pd.cut(d4.qrs_duration, [0, 100, 120, 1000], right=False, labels=["<100", "100-119", ">=120"]),
         "freq": terz("ventricular_rate"), "PR": terz("pr_interval"), "QTc": terz("qt_corrected")}
    for c in ("acquisition_year", "location_setting", "race_ethnicity", "most_recent_ecg"):   # colonne di contesto, se presenti
        if c in d4.columns: S[c] = d4[c]
    S = {k: v.astype(object).where(v.notna(), "NA").astype(str) for k, v in S.items()}
    NBS = 50 if QUICK else 300; srows = []
    for tn, ti in (("SHD", 0), ("LVEF<=45", QT.index("lvef_lte_45_flag"))):
        y = d4[QT[ti]].values.astype(int); p = e4[:, ti]
        order4, starts4, counts4 = _cluster_index(g4)
        for fam, s in S.items():
            for lv in sorted(s.unique()):
                m = (s == lv).values
                if m.sum() < 100 or y[m].sum() < 25 or (m.sum() - y[m].sum()) < 25 or (~m).sum() < 100: continue
                a, lo, hi = auc_ci_cluster(y[m], p[m], g4[m], NBS); r = np.random.default_rng(7); v = []
                for _ in range(NBS):
                    i = _boot_idx(order4, starts4, counts4, r); mi = m[i]
                    if mi.sum() > 20 and (~mi).sum() > 20 and len(np.unique(y[i][mi])) == 2 and len(np.unique(y[i][~mi])) == 2:
                        v.append(safe_auc(y[i][mi], p[i][mi]) - safe_auc(y[i][~mi], p[i][~mi]))
                dif = a - safe_auc(y[~m], p[~m]); sd = np.std(v)
                srows.append(dict(target=tn, famiglia=fam, strato=lv, n=int(m.sum()), pos=int(y[m].sum()), auc=a, lo=lo, hi=hi, diff_resto=dif,
                                  d_lo=float(np.percentile(v, 2.5)), d_hi=float(np.percentile(v, 97.5)), p=float(2 * stats.norm.sf(abs(dif / sd)))))
    ST = pd.DataFrame(srows); hs = holm({str(i): r.p for i, r in ST.iterrows()}); ST["p_holm"] = [hs[str(i)]["p_adj"] for i in ST.index]
    print(len(ST), "strati confrontati (Holm su tutti). Strati con p_Holm < 0.05:"); show(ST[ST.p_holm < 0.05].round(3) if (ST.p_holm < 0.05).any() else pd.DataFrame({"nessuno": []}))
    # regole sugli errori con cross-fitting
    X = pd.DataFrame({"eta": d4.age_at_ecg, "sesso": d4.sex, "freq": d4.ventricular_rate, "atriale": d4.atrial_rate, "PR": d4.pr_interval, "QRS": d4.qrs_duration, "QTc": d4.qt_corrected})
    X = X.fillna(X.median()); half = np.array([zlib.crc32(f"cf{x}".encode()) % 2 for x in g4]); rr, nrule = [], 0
    for tn, ti in (("SHD", 0), ("LVEF<=45", QT.index("lvef_lte_45_flag"))):
        y = d4[QT[ti]].values.astype(int); p = e4[:, ti]; err = np.abs(y - p)
        for tr in (0, 1):
            A, Bm = half == tr, half != tr
            tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=300 if not QUICK else 60, random_state=42).fit(X[A], err[A]); lA, lB = tree.apply(X[A]), tree.apply(X[Bm])
            eA, eB = err[A], err[Bm]; oB, sB, cB = _cluster_index(g4[Bm])
            for lf in np.unique(lA):
                nrule += 1; mA, mB = lA == lf, lB == lf; marA = eA[mA].mean() - eA[~mA].mean(); r = np.random.default_rng(int(lf)); v = []
                for _ in range(200):
                    i = _boot_idx(oB, sB, cB, r); mi = mB[i]
                    if mi.sum() > 20: v.append(eB[i][mi].mean() - eB[i][~mi].mean())
                marB = eB[mB].mean() - eB[~mB].mean(); sd = np.std(v)
                rr.append(dict(target=tn, nA=int(mA.sum()), nB=int(mB.sum()), margine_A=marA, margine_B=marB, lo=float(np.percentile(v, 2.5)), p=float(2 * stats.norm.sf(abs(marB / sd)))))
    RL = pd.DataFrame(rr); hr = holm({str(i): r.p for i, r in RL.iterrows()}); RL["p_holm"] = [hr[str(i)]["p_adj"] for i in RL.index]
    RL["regge"] = (RL.margine_A > 0) & (RL.margine_B > 0) & (RL.lo > 0) & (RL.p_holm < 0.05)
    print(nrule, "regole provate; reggono nella metà di verifica:", int(RL.regge.sum()))
''')
md(r"""
### Risultato che abbiamo ottenuto (fonte: `reports/Q4_embedding-subgroups.md`, handoff Q4)
- **Cluster credibili: 0 su 15** (k = 3 in tutti i fold; AUROC locale sempre sotto il globale, differenza massima con IC superiore −0.007; stabilità ARI 0.899-0.915; silhouette 0.21). I 3 gruppi (prevalenza 0.3 / 0.65 / 0.92) si ripetono in 5 fold su 5 ma sono solo **fasce di punteggio della rete**, non tipi di paziente. Rumore: massimo delle permutazioni +0.005 in media, +0.011-0.014 al 95° percentile. Cluster replicati: nessuno.
- **Strati (90, Holm): dove la CNN è peggiore** — SHD con QRS < 100: 0.787 [0.778-0.795], −0.052 [−0.064, −0.040]; SHD con PR basso: 0.771 [0.760-0.782], −0.054; LVEF ≤ 45% con QRS < 100: 0.840, −0.038; ambulatoriale SHD 0.784, −0.029 (p 0.008). **Dove è migliore**: SHD con QRS 100-119: 0.838 [0.825-0.849], +0.031; LVEF ≤ 45% con ECG più recente: 0.896, +0.043; QTc alto SHD 0.821 (+0.029).
- Regole sugli errori: 11 su 32 reggono; per SHD, QRS stretto (≤ 99) con età > 49 (AUROC 0.75-0.77 contro 0.82-0.84 fuori); per LVEF ≤ 45% è un effetto di prevalenza, non di discriminazione. In totale 137 confronti (15 cluster, 90 strati, 32 regole), Holm applicato.

### Decisione
**Nessun cluster né sottogruppo credibile in cui la CNN sia migliore**: la CNN è moderata-buona ovunque (strato peggiore ≥ 0.77 per SHD, ≥ 0.83 per LVEF ≤ 45%), con un calo di 0.03-0.05 per QRS stretto, PR corto e QTc basso. Gli strati sono diagnostici e correlati tra loro: **non** si usano per scegliere candidati e le differenze non sono causali. Nessun candidato nuovo.
""")

# ------------------------------------------------------------------ cap. 14 · Q5
md(r"""
## 14 · Calibrazione, punti operativi e cosa guarda la rete (fase Q5)
**Obiettivo / perché**: un AUROC alto dice che la rete *ordina* bene i pazienti, non che le probabilità siano *giuste*. Si misura (a) la **calibrazione** (ECE = scarto medio tra probabilità prevista e frequenza osservata per 10 fasce; Brier = errore quadratico medio) e se la **ricalibrazione isotonica** (addestrata fuori dal fold) la migliora; (b) i **punti operativi** (soglie per sensibilità 90%, specificità 90%, massimo di Youden) con PPV/NPV, validi solo con la prevalenza del campione (pazienti che hanno fatto un'eco: bias di selezione); (c) **cosa guarda la rete**: *occlusione* (si azzera una derivazione, poi una finestra di 0.5 s alla volta, e si misura il calo di AUROC) sul modello del fold 0 (5000 righe; 300 con `QUICK`).
""")
code(r'''
from sklearn.isotonic import IsotonicRegression
if OOF1 is None:
    print("RUN_HEAVY=False: nessuna predizione OOF disponibile; risultati attesi nel testo sotto.")
else:
    NB5 = 50 if QUICK else 200; okc5 = CLEAN & ~np.isnan(ENS[:, 0]); d5 = DEV[okc5].reset_index(drop=True); E5 = ENS[okc5]; g5 = d5.patient_key.values; fo5 = d5.fold.values
    print("prevalenza SHD (clean): train %.3f contro validation %.3f -> la calibrazione assoluta dipende dalla popolazione" % (d5[d5.split == "train"][T].mean(), d5[d5.split == "val"][T].mean()))
    for nm in (T, "lvef_lte_45_flag"):
        j = QT.index(nm); y = d5[nm].values.astype(int); p = E5[:, j]; pi = np.zeros_like(p)
        for f in range(N_FOLDS):                                        # isotonica fuori dal fold
            m = fo5 == f; pi[m] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[~m], y[~m]).predict(p[m])
        er, ei = boot_stat_cluster(y, p, g5, ece, NB5), boot_stat_cluster(y, pi, g5, ece, NB5)
        lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); sl = LogisticRegression(C=1e6).fit(lg[:, None], y).coef_[0, 0]
        print(f"{nm[:14]:14s} ECE grezza {fmt3(er)} -> isotonica {fmt3(ei)} | Brier {brier(y, p):.3f} -> {brier(y, pi):.3f} | pendenza {sl:.2f} | AUROC grezza {safe_auc(y, p):.3f}, isotonica {safe_auc(y, pi):.3f}")
        if nm == T: CHECKS.append(("Q5 ECE SHD grezza", 0.031, er[0]))
        pos, neg = y == 1, y == 0; thr_y = None; grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 197)))
        def ss(t):
            pr = p >= t; tp, fp, fn, tn = (pr & pos).sum(), (pr & neg).sum(), (~pr & pos).sum(), (~pr & neg).sum()
            return dict(sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1), ppv=tp / max(tp + fp, 1), npv=tn / max(tn + fn, 1))
        thr_y = grid[int(np.argmax([ss(t)["sens"] + ss(t)["spec"] - 1 for t in grid]))]
        ops = []
        for lab, t in (("sens 90%", np.quantile(p[pos], 0.10)), ("spec 90%", np.quantile(p[neg], 0.90)), ("Youden", thr_y)):
            r = {"punto": lab, "soglia": round(float(t), 3)}
            for k in ("sens", "spec", "ppv", "npv"):
                v = boot_stat_cluster(y, p, g5, lambda yy, pp, k=k, t=t: (lambda pr: {"sens": (pr & (yy == 1)).sum() / max((yy == 1).sum(), 1), "spec": (~pr & (yy == 0)).sum() / max((yy == 0).sum(), 1),
                                    "ppv": (pr & (yy == 1)).sum() / max(pr.sum(), 1), "npv": (~pr & (yy == 0)).sum() / max((~pr).sum(), 1)}[k])(pp >= t), n_boot=NB5)
                r[k] = fmt3(v)
            ops.append(r)
        print("Punti operativi (PPV/NPV validi solo con prevalenza %.2f):" % y.mean()); show(pd.DataFrame(ops))
''')
code(r'''
if MODELS1 is not None:
    NROWS = 300 if QUICK else 5000; NB5 = 50 if QUICK else 200; F0 = 0
    net0, mu0, sd0 = MODELS1[F0]; rd = WaveReader(DEV, list(range(12))); rg = np.random.default_rng(42)
    cand = np.where((DEV.fold.values == F0) & CLEAN)[0]; ix = np.sort(rg.choice(cand, min(NROWS, len(cand)), replace=False)); X0 = rd.get(ix); g0 = DEV.patient_key.values[ix]
    tgs = {"shd": 0, "lvef45": QT.index("lvef_lte_45_flag")}; Y0 = {k: DEV[QT[j]].values[ix].astype(int) for k, j in tgs.items()}
    def pred(Xm):
        outs = []; net0.eval()
        with torch.no_grad():
            for a in range(0, len(Xm), 256):
                x = torch.from_numpy(Xm[a:a + 256]).permute(0, 2, 1).contiguous().to(DEVICE)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                    o = net0(x)
                outs.append(torch.sigmoid(o.float()[:, :NB_BIN]).cpu().numpy())
        return np.vstack(outs)
    base = pred(X0); occ = []
    def occlude(label, Xo):
        po = pred(Xo); r = {"occlusione": label}
        for k, j in tgs.items(): v = auc_diff_ci_cluster(Y0[k], po[:, j], base[:, j], g0, NB5); r["Δ " + k] = f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"
        occ.append(r)
    for c in range(12): Xo = X0.copy(); Xo[:, :, c] = 0.0; occlude(f"canale {c}", Xo)
    for w in range(10): Xo = X0.copy(); Xo[:, w * 250:(w + 1) * 250, :] = 0.0; occlude(f"{w * 0.5:.1f}-{(w + 1) * 0.5:.1f} s", Xo)
    show(pd.DataFrame(occ)); del X0
''')
md(r"""
### Risultato che abbiamo ottenuto (fonte: `reports/Q5_calibration-interpret.md`, handoff Q5)
- **Calibrazione**: ECE SHD 0.031 [0.026-0.036] grezza → **0.003** isotonica (Brier 0.175 → 0.174); ECE LVEF ≤ 45% 0.034 → 0.002 (Brier 0.115 → 0.113); VD 0.016, stenosi aortica 0.004; pendenza 0.76-0.90 (leggermente troppo sicura). La ricalibrazione è utile solo per il *valore assoluto* delle probabilità (AUROC invariata). **Shift di prevalenza**: SHD 0.575 in train contro 0.484 in validation; la rete sovrastima nel validation (0.518 prevista contro 0.484 osservata).
- **Punti operativi SHD**: sens 90% → specificità 0.446 [0.437-0.455], PPV 0.683 (soglia 0.296); spec 90% → sens 0.542 [0.532-0.554], PPV 0.878 (0.739); Youden 0.682/0.799 (0.586). **LVEF ≤ 45%**: sens 90% → spec 0.573 [0.563-0.580], NPV 0.944 (0.073); spec 90% → sens 0.669 [0.655-0.684], PPV 0.695 (0.372); Youden 0.755/0.835 (0.228). PPV/NPV valgono con prevalenza 0.57 (SHD) e 0.25 (LVEF).
- **Cosa guarda la rete**: solo un canale toracico (canale 6) fa calare l'AUROC SHD in modo apprezzabile (−0.026 [−0.035, −0.018] nel fold 0; −0.013 [−0.022, −0.003] nel fold 1); tutte le altre derivazioni < 0.01 (informazione ridondante). **Nessuna finestra** da 0.5 s pesa più di 0.004 (SHD) / 0.006 (LVEF): la rete usa tutto il tracciato in modo diffuso. Coerenza tra fold 0 e 1: solo il messaggio principale; l'ordine fine non è coerente (Spearman 0.20 / 0.26 per le derivazioni, p ≥ 0.31).

### Decisione
La rete è discretamente calibrata ma sovrastima dove la prevalenza è minore: **se si mostrano probabilità, ricalibrare (isotonica)** e dichiarare che PPV/NPV dipendono dalla prevalenza. Interpretabilità: «nessuna finestra temporale o derivazione sufficiente da sola a spiegare la predizione, a parte un canale toracico»; **nessuna affermazione causale**. Nessun candidato nuovo.
""")

# ------------------------------------------------------------------ cap. 15 · Q6-Q8
md(r"""
## 15 · Candidati, revisione e secondo test (fasi Q6, Q7, Q8) — *testo e controlli, nessun test caricato*
**Obiettivo / perché**: scegliere con regole scritte **prima** di guardare il test i modelli da valutare una seconda volta, farli rivedere da un revisore scettico e poi valutarli, una sola volta, sul test. Il test **non** viene aperto da questo notebook: la tabella sotto è testo copiato da `reports/final_test2_table.md`.

**Q6 · candidati congelati** (`results/Q6/preregistration.md`, 2026-11-06, impronte sha256 in `agentic/FROZEN2.json`): C1 SHD (12 derivazioni), C2 LVEF ≤ 45% (12), C3 disfunzione VD (12), C4 stenosi aortica (12), C5 SHD con 3 derivazioni. Massimo 6 candidati; addestrati sul pool di sviluppo intero (3 seed). **Criteri di conferma** (fissati prima): *confermato* = AUROC test con IC inferiore ≥ 0.75 **e** calo dalla CV ≤ 0.05; *buona* = AUROC ≥ 0.80 con IC inferiore ≥ 0.75.
**Q7 · revisione**: verdetto **PASS**; 5/5 candidati ammessi (30/30 impronte integre); 216 AUROC riprodotte con sklearn (differenza massima 0.000); 0 pazienti condivisi tra addestramento e verifica in 56 file OOF; test chiuso (nessun `allow_test` negli script Q); affermazioni 14 supportate / 5 deboli / 0 non supportate; 725 configurazioni nel registro. Rischi residui: popolazione del criterio non scritta (si usa *clean*), riferimento CV di C5 a seed singolo, seconda lettura del test informata dalla prima, C4 non batte il tabulare in CV.
**Q8 · secondo test** (una sola esecuzione, 5 candidati, approvata da un umano; **è la seconda lettura del test**, la prima è P7):

| Candidato | Target | Test (tutte le righe) | Test, righe con LVEF noto (*clean*) | Confronto (tabulare) | Differenza vs confronto | AUROC CV dev |
|---|---|---|---|---|---|---|
| C1_shd_all | SHD | 0.837 [0.827-0.846] (n=5442, pos=2318) | 0.824 [0.813-0.836] (n=4827, pos=2318) | 0.762 [0.750-0.774] | +0.075 [+0.063, +0.086] | 0.815 |
| C2_lvef45_all | LVEF ≤ 45% | 0.905 [0.893-0.915] (n=5442, pos=962) | 0.897 [0.886-0.908] (n=4827, pos=962) | 0.793 [0.776-0.809] | +0.112 [+0.099, +0.127] | 0.869 |
| C3_rv_all | disfunzione VD | 0.899 [0.885-0.914] (n=5442, pos=419) | 0.889 [0.873-0.904] (n=4827, pos=419) | n/d | n/d | 0.844 |
| C4_as_all | stenosi aortica | 0.841 [0.817-0.863] (n=5442, pos=286) | 0.832 [0.811-0.857] (n=4827, pos=286) | n/d | n/d | 0.811 |
| C5_shd_I_II_V1 | SHD, 3 derivazioni | 0.830 [0.820-0.840] (n=5442, pos=2318) | 0.818 [0.808-0.829] (n=4827, pos=2318) | 0.762 [0.750-0.774] | +0.068 [+0.057, +0.081] | 0.802 |

ECE/Brier sul test (tutte le righe, grezze): C1 0.056/0.163, C2 0.025/0.081, C3 0.017/0.051, C4 0.008/0.044, C5 0.060/0.167. Confronto con il primo test (P7): CNN SHD 0.830, tabulare LVEF ≤ 45% 0.793.
""")
code(r'''
# Ricalcolo della regola di conferma (criteri scritti in results/Q6/preregistration.md) sui numeri della tabella (testo copiato dal report)
TEST2 = pd.DataFrame([  # candidato, AUROC test clean, IC inf, IC sup, AUROC CV, n_pos
    ("C1_shd_all", 0.824, 0.813, 0.836, 0.815, 2318), ("C2_lvef45_all", 0.897, 0.886, 0.908, 0.869, 962), ("C3_rv_all", 0.889, 0.873, 0.904, 0.844, 419),
    ("C4_as_all", 0.832, 0.811, 0.857, 0.811, 286), ("C5_shd_I_II_V1", 0.818, 0.808, 0.829, 0.802, 2318)], columns=["candidato", "auc", "lo", "hi", "auc_cv", "n_pos"])
TEST2["calo_cv_test"] = TEST2.auc_cv - TEST2.auc                       # negativo = test sopra la CV (non è evidenza di miglioramento)
TEST2["confermato"] = (TEST2.lo >= 0.75) & (TEST2.calo_cv_test <= 0.05)
TEST2["classe"] = [classe(r.auc, r.lo, r.n_pos) for r in TEST2.itertuples()]
show(TEST2.round(3)); print("confermati:", TEST2.candidato[TEST2.confermato].tolist(), "| non confermati:", TEST2.candidato[~TEST2.confermato].tolist())
CHECKS.append(("Q8 C1 calo CV-test", -0.009, float(TEST2.calo_cv_test[0])))
''')
md(r"""
### Risultato che abbiamo ottenuto (fonte: `reports/Q6_candidates.md`, `Q7_review2.md`, `Q8_final-test2.md`, `final_test2_table.md`)
- **Tutti e 5 i candidati sono «confermati» e di classe «buona»** (test, righe clean, IC95% per paziente): SHD 0.824 [0.813-0.836]; LVEF ≤ 45% 0.897 [0.886-0.908]; VD 0.889 [0.873-0.904]; stenosi aortica 0.832 [0.811-0.857]; SHD con 3 derivazioni 0.818 [0.808-0.829]. Calo CV→test: C1 −0.009, C2 −0.028, C3 −0.045, C4 −0.021, C5 −0.016 (**negativo = test sopra la CV**: non va letto come miglioramento).
- CNN contro tabulare: SHD +0.075 [+0.063, +0.086]; LVEF ≤ 45% +0.112 [+0.099, +0.127]; per VD e stenosi non c'è un comparatore al test (in CV il tabulare batte la CNN per la stenosi).

### Decisione e cautele
Le conclusioni di fattibilità del capitolo 16 si basano su questi numeri. Cautele da dichiarare sempre: **seconda lettura del test** (la prima è P7: il test non è più «vergine»; due letture consumate, non se ne fanno altre); popolazione del criterio *clean* non scritta nella preregistrazione (indicata dal revisore); ECE sul test peggiore che in CV (0.056 contro 0.031 per SHD); SHD ≠ scompenso cardiaco; un solo centro.
""")

# ------------------------------------------------------------------ cap. 16 · sintesi
md(r"""
## 16 · Sintesi, fattibilità e limiti (fonte: `reports/Q9_synthesis2.md`, `reports/REPORT_COMPLESSIVO.md`)
**Verdetto di fattibilità: IN PARTE.**
| Obiettivo | Esito | Fiducia |
|---|---|---|
| SHD composito dalle forme d'onda | **fattibile**: CNN test 0.824 [0.813-0.836] (clean) contro tabulare 0.762 [0.750-0.774] | alta (CV, test, revisione) con le cautele sotto |
| LVEF ≤ 45% | **fattibile**: 0.897 [0.886-0.908] | alta |
| Disfunzione sistolica VD; stenosi aortica | **fattibili**: 0.889 [0.873-0.904]; 0.832 [0.811-0.857] | media-alta (pochi positivi per la stenosi, 286) |
| SHD con 3 derivazioni | **in parte**: 0.818 [0.808-0.829]; perdita piccola ma reale (−0.006 in CV) | media |
| Altre condizioni (spessore parietale, rigurgiti, PASP, versamento) | **moderate** in CV (0.71-0.79); rigurgito aortico **debole** (0.692) | media (non testate al test) |
| Sottogruppi / cluster in cui la predizione è migliore | **non trovati**: 0 credibili (1950 regole + 4 cluster in P; 15 cluster in Q4) | alta (negativo robusto) |
| Il QRS modifica la relazione con la LVEF | **non supportato** sopra la CNN (ΔAUROC +0.0001 [−0.0005, +0.0006]) | media (la CNN vede già il QRS) |
| LVEF numerica precisa | **no**: MAE 7.48, R² 0.482 | alta |

**Limiti**: un solo centro, solo pazienti con eco entro un anno (bias di selezione), righe senza LVEF anomale (9%, motivo non chiarito), calibrazione dipendente dalla prevalenza, nessuna validazione esterna, **725 configurazioni** provate (selezione sul validation ottimistica), test letto **due volte** (P7, Q8). **SHD non è scompenso cardiaco.**
""")
code(r'''
# Confronto dei numeri ricalcolati in questo notebook con i report (solo quelli calcolati; con QUICK=True le differenze sono attese)
if CHECKS:
    cmp_ = pd.DataFrame(CHECKS, columns=["numero", "atteso (report)", "ottenuto qui"]); cmp_["differenza"] = cmp_["ottenuto qui"] - cmp_["atteso (report)"]; show(cmp_.round(4))
print("QUICK =", QUICK, "| RUN_HEAVY =", RUN_HEAVY, "| tempo totale del notebook: %.0f s" % (time.time() - T0_NB))
''')
md(r"""
## 17 · Come riprodurre, tempi e cosa NON è replicabile qui
**Come riprodurre**: impostare `DATA_DIR` (cartella con il CSV dei metadati e i `.npy` delle forme d'onda) o la variabile `ECHONEXT_DIR`; installare `requirements.txt`; eseguire tutte le celle. Prima con `QUICK=True` (verifica del codice), poi `QUICK=False`; con `RUN_HEAVY=True` anche CNN, permutazioni e occlusione.

**Tempi**: vedere `LEGGIMI.md` (misurati in questa verifica; GPU RTX 5080). Stime dei lavori completi: Q1 ≈ 30 min (15 addestramenti da 12 epoche a ~9.5 s/epoca), Q2 ≈ 70 min.

**Non replicabile in questo notebook**: il secondo test (Q8) e il primo (P7), che richiedono gli script di valutazione finale e i candidati congelati (qui solo testo copiato dai report); la ricerca di 1 950 regole di sottogruppo (P5); l'addestramento finale (final-fit) e la congelazione con impronte sha256 (Q6); la revisione indipendente (Q7) e i controlli del registro delle configurazioni; le varianti di iperparametri di Q1. Con `QUICK=True` i numeri sono solo di plausibilità.

**Glossario**: *AUROC* probabilità di ordinare correttamente un malato e un sano; *IC95%* intervallo plausibile per il valore vero; *CV* cross-validation; *OOF* predizione fuori dal fold; *embedding* rappresentazione interna della rete; *ECE* errore di calibrazione; *PPV/NPV* valore predittivo positivo/negativo; *Holm* correzione per confronti multipli; *SHD* cardiopatia strutturale ≥ moderata (≠ scompenso cardiaco).
""")

# ------------------------------------------------------------------ scrittura
for t, s in cells:
    if t == "code":
        ast.parse(s)
def mk(t, s):
    lines = s.split("\n"); src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    c = {"cell_type": t, "metadata": {}, "source": src}
    if t == "code": c.update({"execution_count": None, "outputs": []})
    return c
nb = {"cells": [mk(t, s) for t, s in cells], "metadata": p9.get("metadata", {}), "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]): c["id"] = f"c{i:03d}"
(OUTD / "replica_completa.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("scritto", OUTD / "replica_completa.ipynb", "-", len(cells), "celle,", sum(t == "code" for t, _ in cells), "di codice")
