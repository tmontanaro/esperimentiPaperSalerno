"""P9: genera reports/P9_notebook/replica_analisi.ipynb (senza output). Rieseguibile: python3 experiments/P9/build_notebook.py
Usa nbformat se installato; altrimenti costruisce lo stesso JSON (nbformat v4.5) con un fallback programmatico.
I tempi nel testo vengono letti da results/P9/run_check.json se esiste ("non misurato" altrimenti)."""
import json, sys, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/P9_notebook/replica_analisi.ipynb"
OUT.parent.mkdir(parents=True, exist_ok=True)

try:
    import nbformat
    from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
    HAVE_NBF = True
except ImportError:
    HAVE_NBF = False
    def new_markdown_cell(s): return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": s}
    def new_code_cell(s): return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {}, "execution_count": None, "outputs": [], "source": s}
    def new_notebook(cells):
        return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                              "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}

# ------------------------------------------------------------------ tempi
t_quick = t_full = t_heavy = "non misurato"
rc = ROOT / "results/P9/run_check.json"
if rc.exists():
    j = json.loads(rc.read_text())
    if j.get("quick_total_sec"): t_quick = f"circa {j['quick_total_sec'] / 60:.0f} min (misurato in una prova, RUN_HEAVY=False)"
    if j.get("full_heavy", {}).get("sec"): t_heavy = f"circa {j['full_heavy']['sec'] / 60:.1f} min con GPU RTX 5080 (QUICK=False; CNN 12 epoche ~68 s, permutazioni ~25 s); su CPU non misurato"
    if j.get("full_total_sec"): t_full = f"circa {j['full_total_sec'] / 60:.0f} min (misurato in una prova, RUN_HEAVY=False)"

cells = []
def md(s): cells.append(new_markdown_cell(s.strip("\n")))
def code(s): cells.append(new_code_cell(s.strip("\n")))

# ================================================================== intro
md(f"""
# Replica passo-passo dell'analisi EchoNext

**Che cosa è**: un notebook autonomo con cui rifare, capitolo per capitolo, le analisi della pipeline (audit dati → baseline → condizioni → LVEF e QRS → forme d'onda → cluster → controlli del revisore).
**Fonte dei numeri «attesi»**: i report in `reports/` (P0…P8) e `reports/P8_synthesis.md`; ogni numero è accompagnato dal report di origine.

**Tempi (stima onesta)**
| Modalità | Tempo |
|---|---|
| `QUICK=True`, `RUN_HEAVY=False` | {t_quick} |
| `QUICK=False`, `RUN_HEAVY=False` | {t_full} |
| `RUN_HEAVY=True` (CNN + permutazioni dei cluster) | {t_heavy} |

**Promemoria importante**: *SHD* (cardiopatia strutturale di grado almeno moderato) **non coincide con scompenso cardiaco**: è un'etichetta ricavata dall'ecocardiogramma.
Il **test set resta chiuso**: questo notebook non lo carica (le righe di test vengono scartate subito dopo la lettura) e non esegue la valutazione finale.
""")

md("""
## Come leggere questo notebook
Ogni capitolo ha quattro parti: **Obiettivo / perché**, **codice**, **Risultato che abbiamo ottenuto** (numeri dai report) e **Decisione** (la scelta fatta e, dove possibile, una cella che ricalcola la regola di decisione).

Termini che incontrerete:
- **AUROC**: probabilità che il modello dia un punteggio più alto a un paziente malato che a uno sano scelto a caso. 0.5 = caso, 1 = perfetto.
- **IC95%** (intervallo di confidenza): intervallo plausibile per il valore vero; qui calcolato con il *bootstrap* (si ricampiona più volte il campione e si guarda la dispersione).
- **Train / validation / test**: dati su cui il modello impara / su cui si confrontano e si scelgono i modelli / dati finali, aperti una sola volta. Qui si usano solo train e validation.
- **CV raggruppata**: validazione incrociata (si addestra su parti diverse dei dati a turno) in cui tutti gli ECG dello stesso paziente stanno nella stessa parte, per non «barare» riconoscendo il paziente.
- **Cluster**: gruppi di pazienti simili trovati automaticamente (k-means) senza guardare l'etichetta.
- **Leakage**: informazione dell'esito che finisce per sbaglio tra gli input. Qui gli input sono solo le 7 feature ECG e le forme d'onda.

**Con `QUICK=True` i numeri differiranno** da quelli attesi (campione di addestramento più piccolo, meno modelli, meno ricampionamenti): servono a verificare che il codice giri, non a replicare i valori.
""")

md("## 0 · Preparazione: parametri e funzioni di supporto")
code(r'''
# ----------------------------- PARAMETRI (modificare qui) -----------------------------
DATA_DIR = ""        # <-- COMPILARE: cartella che contiene il CSV dei metadati e i file .npy delle forme d'onda
QUICK = True         # True = sottoinsieme di train + meno modelli/ricampionamenti (pochi minuti); False = analisi completa
SEED = 42
RUN_HEAVY = False    # True = addestra la CNN (cap. 5) e fa le permutazioni dei cluster (cap. 6); richiede torch (meglio con GPU)
# ---------------------------------------------------------------------------------------
import os, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, r2_score
warnings.filterwarnings("ignore")
DATA_DIR = DATA_DIR or os.environ.get("ECHONEXT_DIR", "")
NBOOT = 100 if QUICK else 300          # ricampionamenti bootstrap per gli IC95%
T = "shd_moderate_or_greater_flag"     # obiettivo principale
ECG_FEATURES = ["sex", "ventricular_rate", "atrial_rate", "pr_interval", "qrs_duration", "qt_corrected", "age_at_ecg"]
FLAG_COLS = [T, "lvef_lte_45_flag", "lvwt_gte_13_flag", "aortic_stenosis_moderate_or_greater_flag",
             "aortic_regurgitation_moderate_or_greater_flag", "mitral_regurgitation_moderate_or_greater_flag",
             "tricuspid_regurgitation_moderate_or_greater_flag", "pulmonary_regurgitation_moderate_or_greater_flag",
             "rv_systolic_dysfunction_moderate_or_greater_flag", "pericardial_effusion_moderate_large_flag",
             "pasp_gte_45_flag", "tr_max_gte_32_flag"]
# soglie di decisione del PLAYBOOK (agentic/pipeline_state.json -> config.thresholds)
THR = dict(tabular_signal_min_gain=0.02, promising_target_auroc_lo=0.65, credible_subgroup_margin=0.02,
           good_auroc=0.8, min_n_val=100, min_pos_val=25)

def load_data(data_dir):
    """Legge il CSV dei metadati e SCARTA subito le righe di test (e 'no_split'). Ritorna train, val, cartella dati."""
    root = Path(data_dir).expanduser() if data_dir else None
    if root is None or not root.exists():
        raise FileNotFoundError("Compila DATA_DIR con la cartella dei dati EchoNext.")
    c = [p for p in root.rglob("*") if p.is_file() and "metadata" in p.name.lower() and p.name.lower().endswith((".csv", ".csv.gz"))]
    if not c:
        raise FileNotFoundError(f"Nessun CSV 'metadata' in {root}")
    c.sort(key=lambda p: ("100k" not in p.name.lower(), len(str(p))))
    meta = pd.read_csv(c[0])
    meta["sex"] = meta["sex"].astype(str).str.lower().map({"male": 1, "female": 0, "m": 1, "f": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})
    meta["row_in_split"] = meta.groupby("split").cumcount()      # l'ordine del CSV = ordine dei file .npy
    meta = meta[meta.split.isin(["train", "val"])].copy()         # <-- il test set viene scartato qui
    return meta[meta.split == "train"].copy(), meta[meta.split == "val"].copy(), c[0].parent

def safe_auc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")

def boot_ci(y, p, fn=safe_auc, n=NBOOT, seed=0):
    rng = np.random.default_rng(seed); y, p = np.asarray(y), np.asarray(p); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); x = fn(y[i], p[i])
        if not np.isnan(x): v.append(x)
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)

def metrics_binary(y, p, n_boot=None):
    y = np.asarray(y); lo, hi = boot_ci(y, p, n=n_boot or NBOOT)
    return {"n": int(len(y)), "n_pos": int(y.sum()), "auroc": safe_auc(y, p), "auroc_lo": lo, "auroc_hi": hi,
            "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else np.nan}

def fmt(m):  # "0.746 [0.733-0.759]"
    return f"{m['auroc']:.3f} [{m['auroc_lo']:.3f}-{m['auroc_hi']:.3f}]"

def get_xy(df, target, features=ECG_FEATURES):
    d = df.dropna(subset=[target]); return d[list(features)].astype(float), d[target].values.astype(int), d

def make_models(quick=QUICK):
    imp = lambda: SimpleImputer(strategy="median", add_indicator=True)
    m = {"logreg": make_pipeline(imp(), StandardScaler(), LogisticRegression(max_iter=2000)),
         "hist_gb": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=250, random_state=SEED)}
    if not quick:
        m["random_forest"] = make_pipeline(imp(), RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=SEED))
        m["mlp_dense"] = make_pipeline(imp(), StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, max_iter=200, random_state=SEED))
    return m

def new_hgb():
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=250, random_state=SEED)

def oof_proba(model, X, y, groups, n_splits=5):
    """Predizioni 'fuori dal fold' con CV raggruppata per paziente (GroupKFold)."""
    return cross_val_predict(model, X, y, groups=groups, cv=GroupKFold(n_splits), method="predict_proba")[:, 1]

def add_groups(df):
    d = df.copy()
    d["qrs_bin"] = pd.cut(d.qrs_duration, [0, 100, 120, 10000], labels=["<100", "100-119", ">=120"], right=False).astype(object)
    return d

def show(df, max_rows=60):
    """Stampa solo tabelle aggregate (rifiuta identificativi di paziente e tabelle lunghe)."""
    if isinstance(df, pd.Series): df = df.to_frame()
    if "patient_key" in df.columns or "row_in_split" in df.columns: raise ValueError("show(): solo aggregati")
    if len(df) > max_rows: raise ValueError("show(): troppe righe, aggrega prima di stampare")
    print(df.round(4).to_string())

RESULTS = {}   # numeri calcolati qui, per il confronto con i report
T0_NB = time.time()
train, val, WAVE_DIR = load_data(DATA_DIR)
if QUICK:   # sottoinsieme di PAZIENTI (non di righe) per non spezzare i gruppi
    pk = np.random.default_rng(SEED).choice(train.patient_key.unique(), 6000, replace=False)
    train = train[train.patient_key.isin(pk)].copy()
print(f"QUICK={QUICK} | train: {len(train)} ECG, {train.patient_key.nunique()} pazienti | validation: {len(val)} ECG")
''')

# ================================================================== P0
md("""
## 1 · Audit dei dati (fase P0)
**Obiettivo**: verificare che i dati siano utilizzabili: split corretti, nessun paziente in due split, valori mancanti, prevalenza delle etichette.
**Perché questo passo**: se lo stesso paziente comparisse in train e validation, i risultati sarebbero gonfiati; e i valori mancanti vanno gestiti nei modelli.
""")
code(r'''
ov = len(set(train.patient_key) & set(val.patient_key))
print("pazienti presenti sia in train sia in validation:", ov)
print("ECG per paziente, massimo -> train:", int(train.groupby("patient_key").size().max()), "| val:", int(val.groupby("patient_key").size().max()))
print(f"prevalenza target SHD: train {train[T].mean():.3f}, validation {val[T].mean():.3f}")
miss = pd.DataFrame({"mancanti_%_train": train[ECG_FEATURES + ["lvef_value"]].isna().mean() * 100}).round(1)
show(miss)
prev = pd.DataFrame({"prev_train": train[FLAG_COLS].mean(), "prev_val": val[FLAG_COLS].mean(), "n_pos_val": val[FLAG_COLS].sum()})
show(prev)
print("forme d'onda presenti:", all((WAVE_DIR / f"EchoNext_{s}_waveforms.npy").exists() for s in ("train", "val")))
RESULTS["prev_train"] = float(train[T].mean()); RESULTS["prev_val"] = float(val[T].mean())
RESULTS["n_train"] = len(train); RESULTS["n_val"] = len(val)
''')
md("""
### Risultato che abbiamo ottenuto (fonte: `reports/P0_data-audit.md`, handoff P0)
- Integrità OK: 0 pazienti in comune tra train/val/test; max 1 ECG per paziente in val e test, max 146 in train.
- Train: 72 475 ECG (26 218 pazienti); validation: 4 626 ECG. Prevalenza SHD: 0.524 (train), 0.430 (validation): c'è uno **spostamento di prevalenza**.
- Mancanti (train): pr_interval 10.5%, atrial_rate 0.6%, lvef_value 8.9%.
- Con `QUICK=True` il train è un sottoinsieme di 6 000 pazienti: conteggi e prevalenze differiranno, la validation è invariata (4 626 ECG, 0.430).
""")
md("""
### Decisione
Si usa la **CV raggruppata per `patient_key`** (un paziente ha fino a 146 ECG in train) e si **imputano** `pr_interval` e `atrial_rate` dentro i modelli. Le righe con `lvef_value` mancante vanno escluse nelle analisi sulla LVEF (cap. 4).
""")

# ================================================================== P1
md("""
## 2 · Baseline (fase P1)
**Obiettivo**: capire se le 7 feature ECG tabellari (frequenze, intervalli, QRS, QTc, età, sesso) contengono un segnale oltre a **età + sesso** da soli, e fissare un modello globale di riferimento.
**Perché questo passo**: è il termine di paragone per tutto il resto; se l'ECG tabellare non aggiungesse nulla ad età e sesso, non avrebbe senso cercare sottogruppi.
""")
code(r'''
rows, fitted = [], {}
for sname, feats in {"eta_sesso": ["age_at_ecg", "sex"], "tutte_ecg": ECG_FEATURES}.items():
    Xtr, ytr, _ = get_xy(train, T, feats); Xva, yva, _ = get_xy(val, T, feats)
    for name, m in make_models().items():
        m.fit(Xtr, ytr); p = m.predict_proba(Xva)[:, 1]
        rows.append({"features": sname, "model": name, **metrics_binary(yva, p)}); fitted[(sname, name)] = m
P1TAB = pd.DataFrame(rows).sort_values(["features", "auroc"], ascending=[True, False]).reset_index(drop=True)
show(P1TAB)
best_all = P1TAB[P1TAB.features == "tutte_ecg"].sort_values("auroc", ascending=False).iloc[0]
best_ag = P1TAB[P1TAB.features == "eta_sesso"].sort_values("auroc", ascending=False).iloc[0]
GLOBAL = fitted[("tutte_ecg", "hist_gb")]                 # modello globale di riferimento (hist_gb, 7 feature)
Xva_all, yva_all, VAL_T = get_xy(val, T)
p_tab_val = GLOBAL.predict_proba(Xva_all)[:, 1]
m_glob = metrics_binary(yva_all, p_tab_val)
RESULTS["p1_hgb_all_auroc"] = m_glob["auroc"]; RESULTS["p1_best_agesex_auroc"] = float(best_ag.auroc)

# coerenza con la CV raggruppata sul train (hist_gb, 7 feature)
Xtr, ytr, dtr = get_xy(train, T)
oof = oof_proba(new_hgb(), Xtr, ytr, dtr.patient_key.values, n_splits=3 if QUICK else 5)
cv = metrics_binary(ytr, oof)
print("\nhist_gb, CV raggruppata train:", fmt(cv), "| validation:", fmt(m_glob), "| CV - val = %.3f" % (cv["auroc"] - m_glob["auroc"]))
RESULTS["p1_cv_auroc"] = cv["auroc"]

# importanza delle feature: quanto cala l'AUROC (su validation) mescolando una feature
from sklearn.inspection import permutation_importance
pi = permutation_importance(GLOBAL, Xva_all, yva_all, scoring="roc_auc", n_repeats=3 if QUICK else 10, random_state=SEED)
show(pd.DataFrame({"feature": Xva_all.columns, "calo_AUROC": pi.importances_mean}).sort_values("calo_AUROC", ascending=False))
''')
md("""
### Risultato che abbiamo ottenuto (fonte: `reports/P1_baseline.md`, handoff P1; validation, n=4 626, 1 990 positivi)
- Solo età+sesso: AUROC **0.659 [0.642-0.675]** (miglior modello: mlp_dense).
- 7 feature ECG, `hist_gb`: AUROC **0.746 [0.733-0.760]**, AUPRC 0.699 (AUPRC = precisione media, utile quando le classi sono sbilanciate).
- CV raggruppata sul train, hist_gb: 0.728 [0.724-0.731]; CV − val = −0.019: nessuna incoerenza.
- Importanza (calo di AUROC): qt_corrected 0.063, age 0.039, qrs_duration 0.031, pr_interval 0.025.
- Con `QUICK=True`: train più piccolo, solo logreg e hist_gb, CV a 3 fold: valori un po' più bassi (differenze attese di alcuni punti centesimali).
""")
md("""
### Decisione
Regola del PLAYBOOK: c'è **segnale tabulare** se (AUROC con tutte le feature − AUROC età+sesso) ≥ 0.02 **e** l'estremo inferiore dell'IC > 0.55. Nella pipeline il guadagno era 0.0877 → `tabular_signal = True`, quindi la fase P2 ha eseguito l'analisi completa (inclusa la ricerca di sottogruppi). La cella sotto ricalcola la regola.
""")
code(r'''
gain = best_all.auroc - best_ag.auroc
signal = bool(gain >= THR["tabular_signal_min_gain"] and best_all.auroc_lo > 0.55)
print(f"guadagno sopra età+sesso = {gain:.4f} (soglia {THR['tabular_signal_min_gain']}); IC inferiore = {best_all.auroc_lo:.3f} (soglia 0.55)")
print("-> tabular_signal =", signal, "| ramo:", "analisi completa dei sottogruppi in P2" if signal else "P2 ridotta (niente regole di sottogruppo)")
print("(attesa nella pipeline: guadagno 0.0877, tabular_signal=True)")
RESULTS["p1_gain"] = float(gain)
''')

# ================================================================== P2
md("""
## 3 · Condizioni singole (fase P2)
**Obiettivo**: per ciascuna delle 12 etichette (composito SHD e singole condizioni) vedere quanto è predicibile dalle feature ECG, e scegliere gli obiettivi «promettenti».
**Perché questo passo**: alcune condizioni (es. stenosi aortica) lasciano tracce ECG più chiare di altre; i passi successivi (cluster) si concentrano sulle migliori.
Un obiettivo è **promettente** se ha ≥ 25 positivi in validation e l'estremo inferiore dell'IC95% dell'AUROC è ≥ 0.65.
""")
code(r'''
rows = []
for tg in FLAG_COLS:
    Xtr, ytr, _ = get_xy(train, tg); Xva, yva, _ = get_xy(val, tg)
    if yva.sum() < 5: continue
    p = new_hgb().fit(Xtr, ytr).predict_proba(Xva)[:, 1]
    m = metrics_binary(yva, p)
    short = tg.replace("_moderate_or_greater_flag", "").replace("_moderate_large_flag", "").replace("_flag", "")
    rows.append({"target": short, "col": tg, **m, "promettente": bool(m["n_pos"] >= THR["min_pos_val"] and m["auroc_lo"] >= THR["promising_target_auroc_lo"])})
P2TAB = pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)
show(P2TAB.drop(columns="col"))
RESULTS["p2_aortic_stenosis_auroc"] = float(P2TAB.loc[P2TAB.target == "aortic_stenosis", "auroc"].iloc[0])
''')
md("""
### Risultato che abbiamo ottenuto (fonte: `reports/P2_conditions.md`, handoff P2; validation, hist_gb su 7 feature ECG)
- Promettenti: stenosi aortica **0.844 [0.819-0.868]**, rigurgito mitralico 0.770 [0.741-0.793], LVEF ≤ 45% 0.764 [0.748-0.782], disfunzione sistolica del VD 0.764 [0.737-0.787], rigurgito tricuspidale 0.756 [0.730-0.783], più pasp ≥ 45, tr_max ≥ 32, lvwt ≥ 13.
- Non promettenti: versamento pericardico 0.678 (IC inf. 0.604), rigurgito aortico 0.674 (IC inf. 0.592), rigurgito polmonare (21 positivi in validation, 0.613).
- **Sottogruppi credibili: nessuno** (1 950 regole provate sul train, 15 verificate in validation; nessuna con margine ≥ 0.02 e IC inferiore sopra il globale). Questa ricerca di regole **non è riprodotta in questo notebook** (script `experiments/P2/conditions.py`).
- Modello locale vs globale (composito, 19 strati): nessuna differenza positiva significativa.
- Con `QUICK=True` le AUROC saranno leggermente inferiori e gli IC più larghi.
""")
md("""
### Decisione
Per i cluster (cap. 6) si usano il composito più i **2 obiettivi promettenti con AUROC più alta** (nella pipeline: stenosi aortica e rigurgito mitralico). La cella ricalcola questa scelta dalla tabella sopra. Nessun sottogruppo credibile è stato portato avanti.
""")
code(r'''
prom = P2TAB[(P2TAB.promettente) & (P2TAB.target != "shd")].sort_values("auroc", ascending=False)
top2 = list(prom.target[:2])
print("obiettivi promettenti (escluso il composito):", list(prom.target))
print("-> cluster_targets ricalcolati: composito +", top2)
print("(attesi nella pipeline: composito + aortic_stenosis + mitral_regurgitation)")
CLUSTER_TARGETS = [T] + list(prom.col[:2])
''')

# ================================================================== P3
md("""
## 4 · LVEF e durata del QRS (fase P3)
**Obiettivo**: (a) predire la frazione di eiezione (LVEF, % di sangue espulso a ogni battito) in modo continuo e come soglia «LVEF ≤ 45%»; (b) verificare l'ipotesi dei medici che la **durata del QRS** (tempo di attivazione dei ventricoli, ms) modifichi la relazione tra feature ECG e LVEF.
**Perché questo passo**: è uno degli obiettivi dichiarati. Qui replichiamo (a) e il test (i) di (b): modelli separati per fascia di QRS contro il modello globale.
I test (ii) interazione con CV raggruppata e (iii) coppie di pazienti simili **non sono riprodotti** qui (`experiments/P3/lvef_qrs.py`).
Le righe senza `lvef_value` (≈ 9%) sono escluse dalla regressione.
""")
code(r'''
tr_l = add_groups(train).dropna(subset=["lvef_value"]).reset_index(drop=True)
va_l = add_groups(val).dropna(subset=["lvef_value"]).reset_index(drop=True)
print("righe con LVEF: train", len(tr_l), "| validation", len(va_l))
Xtl, ytl, Xvl, yvl = tr_l[ECG_FEATURES].astype(float), tr_l.lvef_value.values, va_l[ECG_FEATURES].astype(float), va_l.lvef_value.values
def new_hgr(): return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=250, random_state=SEED)
reg = new_hgr().fit(Xtl, ytl); pr = reg.predict(Xvl); pbase = np.full(len(yvl), ytl.mean())
def boot_reg(y, p, fn, n=NBOOT, seed=0):
    rng = np.random.default_rng(seed); v = [fn(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
mae, r2 = mean_absolute_error(yvl, pr), r2_score(yvl, pr)
print(f"LVEF regressione (hist_gb): MAE = {mae:.2f} {boot_reg(yvl, pr, mean_absolute_error)}, R2 = {r2:.3f} {boot_reg(yvl, pr, r2_score)}")
print(f"baseline (media del train): MAE = {mean_absolute_error(yvl, pbase):.2f}, R2 = {r2_score(yvl, pbase):.3f}")
RESULTS["p3_mae"] = float(mae); RESULTS["p3_r2"] = float(r2)

# classificazione LVEF <= 45% (tutte le righe con etichetta, come nella pipeline)
Xc, yc, _ = get_xy(train, "lvef_lte_45_flag"); Xcv, ycv, _ = get_xy(val, "lvef_lte_45_flag")
pc = new_hgb().fit(Xc, yc).predict_proba(Xcv)[:, 1]; mc = metrics_binary(ycv, pc)
print("LVEF <= 45%, AUROC validation:", fmt(mc), "| positivi:", mc["n_pos"])
RESULTS["p3_lte45_auroc"] = mc["auroc"]

# test (i): modello per fascia di QRS vs modello globale, R2 sugli stessi pazienti di validation
rows = []
for g in ["<100", "100-119", ">=120"]:
    k_tr, k_va = (tr_l.qrs_bin == g).values, (va_l.qrs_bin == g).values
    if k_va.sum() < 50 or k_tr.sum() < 100: continue
    loc = new_hgr().fit(Xtl[k_tr], ytl[k_tr]).predict(Xvl[k_va]); y_g, p_glob = yvl[k_va], pr[k_va]
    rng = np.random.default_rng(1); d = []
    for _ in range(NBOOT):
        i = rng.integers(0, len(y_g), len(y_g)); d.append(r2_score(y_g[i], loc[i]) - r2_score(y_g[i], p_glob[i]))
    rows.append({"QRS": g, "n_val": int(k_va.sum()), "R2_locale": r2_score(y_g, loc), "R2_globale": r2_score(y_g, p_glob),
                 "delta_R2": r2_score(y_g, loc) - r2_score(y_g, p_glob), "delta_lo": float(np.percentile(d, 2.5)), "delta_hi": float(np.percentile(d, 97.5))})
P3QRS = pd.DataFrame(rows); show(P3QRS)
''')
md("""
### Risultato che abbiamo ottenuto (fonte: `reports/P3_lvef-qrs.md` e `R1-P3_review-lvef-qrs.md`; validation)
- Regressione LVEF, hist_gb (n = 4 112): **MAE 9.54 [9.31-9.79]**, **R² 0.163 [0.136-0.191]**; baseline MAE 11.19, R² −0.02 (R² = quota di variabilità spiegata).
- LVEF ≤ 45%: AUROC **0.764 [0.748-0.782]** (866 positivi); nota: scelto guardando il validation, quindi un po' ottimistico.
- (i) QRS, ΔR² locale−globale: +0.007 [0.001, 0.013] (< 100), +0.004 [−0.014, 0.023] (100-119), +0.009 [−0.007, 0.024] (≥ 120): nessun beneficio rilevante.
- (ii) interazione sul train: significativa nel test statistico (p < 1e-4) ma, in CV raggruppata, R² 0.1400 → 0.1413: trascurabile. (iii) coppie: QRS più largo di 2-8 ms nella LVEF bassa; con IC che includono 0 in validation per 3 insiemi su 4.
- **Verdetto: ipotesi QRS inconclusiva** (è un'associazione, non una modifica d'effetto).
- Con `QUICK=True` l'R² sarà più basso e gli ΔR² più rumorosi.
""")
md("""
### Decisione
Il PLAYBOOK apre un follow-up (P3b, modelli per fascia QRS) solo se la modifica è «supportata» (≥ 2-3 test su 3 concordi). Qui il test (i) mostra ΔR² ≤ 0.01: **nessun P3b**. La cella ricalcola il ramo per il solo test (i).
""")
code(r'''
big = (P3QRS.delta_R2 >= THR["credible_subgroup_margin"]) & (P3QRS.delta_lo > 0)
print("fasce con ΔR² ≥ 0.02 e IC inferiore > 0:", int(big.sum()), "su", len(P3QRS), "| ΔR² massimo =", round(P3QRS.delta_R2.max(), 3))
print("-> ramo:", "indizio di modifica: andrebbero guardati anche i test (ii) e (iii)" if big.any() else "nessun beneficio del modello per fascia: niente follow-up P3b (QRS inconclusiva)")
''')

# ================================================================== P4
md("""
## 5 · Forme d'onda (fase P4)
**Obiettivo**: verificare se una rete neurale convoluzionale (CNN, un modello che legge direttamente i 12 tracciati ECG da 10 secondi) predice meglio delle 7 feature tabellari.
**Perché questo passo**: le 7 feature riassumono l'ECG in pochi numeri; la forma d'onda contiene molta più informazione.
La CNN è una ResNet 1D, multi-etichetta (12 etichette insieme, ignorando i valori mancanti), addestrata su train per 12 epoche con **l'ultima epoca usata** (nessuna selezione sul validation), **un solo seed**. Input: solo forme d'onda (nessuna feature tabellare).
Cella pesante: gira solo con `RUN_HEAVY=True` (con `QUICK=True`: sottoinsieme di train e 3 epoche, i numeri saranno molto più bassi).
""")
code(r'''
CNN = None
if not RUN_HEAVY:
    print("RUN_HEAVY=False: la CNN non viene addestrata. Risultati attesi (reports/P4_waveform.md): AUROC validation SHD 0.823 [0.811-0.834]")
else:
    import torch, torch.nn as nn, torch.nn.functional as F
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("dispositivo:", dev)
    class Block(nn.Module):
        def __init__(s, cin, cout, stride):
            super().__init__()
            s.c1 = nn.Conv1d(cin, cout, 7, stride, 3, bias=False); s.b1 = nn.BatchNorm1d(cout)
            s.c2 = nn.Conv1d(cout, cout, 7, 1, 3, bias=False); s.b2 = nn.BatchNorm1d(cout)
            s.sk = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout)) if (stride != 1 or cin != cout) else nn.Identity()
        def forward(s, x):
            h = F.relu(s.b1(s.c1(x))); h = s.b2(s.c2(h)); return F.relu(h + s.sk(x))
    class ECGNet(nn.Module):
        def __init__(s, n_out):
            super().__init__()
            s.stem = nn.Sequential(nn.Conv1d(12, 32, 15, 2, 7, bias=False), nn.BatchNorm1d(32), nn.ReLU())
            s.body = nn.Sequential(Block(32, 32, 2), Block(32, 64, 2), Block(64, 128, 2), Block(128, 128, 2))
            s.drop = nn.Dropout(0.3); s.head = nn.Linear(256, n_out)
        def forward(s, x):
            h = s.body(s.stem(x)); return s.head(s.drop(torch.cat([h.mean(-1), h.amax(-1)], 1)))
    TARGETS = [T] + [c for c in FLAG_COLS if c != T]
    EPOCHS, BATCH, LR = (3 if QUICK else 12), 128, 2e-3
    Wtr = np.load(WAVE_DIR / "EchoNext_train_waveforms.npy", mmap_mode="r"); Wva = np.load(WAVE_DIR / "EchoNext_val_waveforms.npy", mmap_mode="r")
    Ytr, rtr = train[TARGETS].values.astype(np.float32), train.row_in_split.values
    def get_batch(W, rows):  # lettura da disco in ordine crescente di riga
        return torch.from_numpy(np.ascontiguousarray(np.asarray(W[rows, 0], dtype=np.float32).transpose(0, 2, 1)))
    net = ECGNet(len(TARGETS)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-2)
    steps = EPOCHS * (len(train) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)
    rng = np.random.default_rng(SEED)
    for ep in range(EPOCHS):
        net.train(); perm = rng.permutation(len(train)); t0 = time.time(); tl = 0
        for s in range(len(train) // BATCH):
            b = perm[s * BATCH:(s + 1) * BATCH]; b = b[np.argsort(rtr[b])]
            x = get_batch(Wtr, rtr[b]).to(dev); y = torch.from_numpy(Ytr[b]).to(dev)
            opt.zero_grad()
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=dev == "cuda"):
                lo = net(x).float()
            m = ~torch.isnan(y); loss = (F.binary_cross_entropy_with_logits(lo, torch.nan_to_num(y), reduction="none") * m).sum() / m.sum().clamp(min=1)
            loss.backward(); opt.step(); sched.step(); tl += loss.item()
        print(f"epoca {ep + 1}/{EPOCHS}: loss {tl / (len(train) // BATCH):.4f} ({time.time() - t0:.0f}s)")
    net.eval(); P = []
    with torch.no_grad():
        for s in range(0, len(val), 256):
            rows = val.row_in_split.values[s:s + 256]
            P.append(torch.sigmoid(net(get_batch(Wva, rows).to(dev)).float()).cpu().numpy())
    P = np.vstack(P); ok = ~np.isnan(val[T].values)
    CNN = metrics_binary(val[T].values[ok], P[ok, 0]); print("CNN, SHD, validation:", fmt(CNN))
    # differenza CNN - tabulare sugli stessi pazienti (bootstrap appaiato)
    yv_, pc_, pt_ = val[T].values[ok].astype(int), P[ok, 0], p_tab_val
    rg = np.random.default_rng(0); dd = []
    for _ in range(NBOOT):
        i = rg.integers(0, len(yv_), len(yv_)); dd.append(safe_auc(yv_[i], pc_[i]) - safe_auc(yv_[i], pt_[i]))
    CNN["gain"], CNN["gain_lo"], CNN["gain_hi"] = CNN["auroc"] - m_glob["auroc"], float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))
    print("guadagno CNN - tabulare: %.3f [%.3f, %.3f]" % (CNN["gain"], CNN["gain_lo"], CNN["gain_hi"]))
    RESULTS["p4_cnn_auroc"] = CNN["auroc"]
''')
md("""
### Risultato che abbiamo ottenuto (fonte: `reports/P4_waveform.md`, `R1-P4_review-waveform.md`; validation, GPU RTX 5080)
- CNN, SHD: AUROC **0.823 [0.811-0.834]**, AUPRC 0.804; tabulare hist_gb 0.746 [0.733-0.759]; guadagno **+0.077 [0.063, 0.091]**.
- Per obiettivo (CNN): LVEF ≤ 45% 0.889 [0.876-0.902], disfunzione sistolica VD 0.885 [0.868-0.904], rigurgito tricuspidale 0.843, mitralico 0.839, stenosi aortica 0.831 (pari al tabulare, 0.844).
- Per fascia di QRS la CNN batte il tabulare: < 100 0.791, 100-119 0.844, ≥ 120 0.845 (differenze +0.087, +0.121, +0.093).
- Equità: la CNN è sopra il tabulare in 16/16 strati (etnia, setting, sesso, età); AUROC più bassa nei procedurali (0.748 [0.677-0.818], n = 177) e ≥ 75 anni (0.788).
- Cautele: un seed, una sola configurazione; picco monitorato 0.831 all'epoca 8, ma si usa l'ultima (0.823).
- Con `QUICK=True` (se acceso) i numeri sono molto inferiori; con `RUN_HEAVY=False` non c'è nulla da confrontare.
""")
md("""
### Decisione
Regola del PLAYBOOK: se (AUROC CNN − AUROC tabulare) ≥ 0.05 si imposta `waveform_dominant = True` (le forme d'onda dominano). Nella pipeline il guadagno era +0.077 → `waveform_dominant = True`, e il candidato CNN è stato portato alla revisione e al test. La cella ricalcola la regola, usando il tuo valore se hai addestrato la CNN, altrimenti quello del report.
""")
code(r'''
if CNN is not None:
    cnn_auc, fonte = CNN["auroc"], "ricalcolato in questo notebook"
else:
    cnn_auc, fonte = 0.823, "dal report P4 (CNN non riaddestrata)"
gain_w = cnn_auc - m_glob["auroc"] if CNN is not None else 0.823 - 0.746
print(f"AUROC CNN = {cnn_auc:.3f} ({fonte}); guadagno vs tabulare = {gain_w:.3f}")
print("-> waveform_dominant =", bool(gain_w >= 0.05), "| ramo:", "la CNN è il modello principale" if gain_w >= 0.05 else "restano le feature tabellari")
''')

# ================================================================== P5
md("""
## 6 · Cluster di pazienti (fase P5)
**Obiettivo**: capire se esistono **gruppi di pazienti** (cluster) in cui un modello addestrato solo su quel gruppo predice *meglio* del modello globale.
**Perché questo passo**: un modello globale potrebbe funzionare male in certi tipi di pazienti e bene in altri; un cluster è «credibile» solo se, in validation, ha ≥ 100 pazienti (≥ 25 positivi e ≥ 25 negativi), l'AUROC locale supera quella globale di ≥ 0.02 **e** l'estremo inferiore dell'IC supera l'AUROC globale.
**Semplificazione**: nella pipeline le feature di clustering sono state scelte sul train con una selezione all'indietro (78 configurazioni, solo train) e k-means sparso; qui si usano **direttamente le feature e k scelti** (sesso, atrial_rate, qrs_duration; k = 4). La selezione è in `experiments/P5/select_cluster.py`.
""")
code(r'''
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
CL_FEATS, K = ["sex", "atrial_rate", "qrs_duration"], 4
pool = train.sample(frac=1, random_state=SEED).drop_duplicates("patient_key")        # 1 ECG per paziente
med = pool[CL_FEATS].median(); mu = pool[CL_FEATS].fillna(med).mean(); sd = pool[CL_FEATS].fillna(med).std()
prep = lambda df: ((df[CL_FEATS].fillna(med) - mu) / sd).clip(-4, 4).values
Z = prep(pool)
km = KMeans(K, n_init=10, random_state=SEED).fit(Z)
centers = km.cluster_centers_[np.argsort(-np.bincount(km.labels_, minlength=K))]    # cluster 0 = il più grande
assign = lambda df: ((prep(df)[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
tr_cl, va_cl = assign(train), assign(val)
sil_i = np.random.default_rng(SEED).choice(len(Z), min(4000, len(Z)), replace=False)
sil = silhouette_score(Z[sil_i], km.labels_[sil_i])
perm = np.random.default_rng(SEED).permutation(len(Z)); a, b = perm[:len(Z) // 2], perm[len(Z) // 2:]
ari = adjusted_rand_score(KMeans(K, n_init=3, random_state=1).fit(Z[a]).predict(Z), KMeans(K, n_init=3, random_state=2).fit(Z[b]).predict(Z))
print(f"k = {K}, silhouette = {sil:.3f} (separazione dei gruppi), stabilità ARI tra due metà = {ari:.3f} (1 = stessi gruppi)")
print("dimensione dei cluster in validation:", np.bincount(va_cl, minlength=K).tolist())
RESULTS["p5_silhouette"] = float(sil); RESULTS["p5_ari"] = float(ari)

# modello locale (addestrato solo sul cluster) vs globale (intero train), valutati in validation
rows = []
for tg in CLUSTER_TARGETS:
    Xt, yt, dt = get_xy(train, tg); Xv, yv, dv = get_xy(val, tg)
    ctr, cva = tr_cl[train.index.get_indexer(dt.index)], va_cl[val.index.get_indexer(dv.index)]
    gm = metrics_binary(yv, new_hgb().fit(Xt, yt).predict_proba(Xv)[:, 1])
    for c in range(K):
        kt, kv = ctr == c, cva == c
        if kv.sum() < 20 or len(np.unique(yt[kt])) < 2: continue
        m = metrics_binary(yv[kv], new_hgb().fit(Xt[kt], yt[kt]).predict_proba(Xv[kv])[:, 1])
        ok = bool(m["n"] >= THR["min_n_val"] and m["n_pos"] >= THR["min_pos_val"] and (m["n"] - m["n_pos"]) >= THR["min_pos_val"]
                  and m["auroc"] - gm["auroc"] >= THR["credible_subgroup_margin"] and m["auroc_lo"] > gm["auroc"])
        rows.append({"target": tg.replace("_moderate_or_greater_flag", "").replace("_flag", ""), "cluster": c, "n_val": m["n"], "n_pos": m["n_pos"],
                     "AUROC_locale": m["auroc"], "lo": m["auroc_lo"], "hi": m["auroc_hi"], "AUROC_globale_val": gm["auroc"], "credibile": ok})
P5TAB = pd.DataFrame(rows); show(P5TAB)
ct = pd.crosstab(va_cl, val[T]); print("chi-quadro cluster x SHD (validation): %.1f, p = %.2g" % (stats.chi2_contingency(ct)[0], stats.chi2_contingency(ct)[1]))
''')
code(r'''
# Rumore di fondo (permutazioni): si mescolano le etichette e si guarda l'AUROC locale massima che si ottiene per caso.
# Cella pesante; non ripete la selezione delle feature di clustering (la pipeline sì).
if not RUN_HEAVY:
    print("RUN_HEAVY=False. Atteso (reports/P5_clusters.md): p empirico 0.032 sul massimo AUROC (30 permutazioni); 0.52 / 0.77 / 0.94 sul guadagno locale-globale (SHD / stenosi / mitrale).")
else:
    n_perm = 5 if QUICK else 30; rng = np.random.default_rng(SEED); obs = P5TAB.AUROC_locale.max(); mx = []
    for _ in range(n_perm):
        best = 0
        for tg in CLUSTER_TARGETS:
            Xt, yt, dt = get_xy(train, tg); Xv, yv, dv = get_xy(val, tg)
            yt_p, yv_p = rng.permutation(yt), rng.permutation(yv)
            ctr, cva = tr_cl[train.index.get_indexer(dt.index)], va_cl[val.index.get_indexer(dv.index)]
            for c in range(K):
                kt, kv = ctr == c, cva == c
                if kv.sum() >= THR["min_n_val"]:
                    best = max(best, safe_auc(yv_p[kv], new_hgb().fit(Xt[kt], yt_p[kt]).predict_proba(Xv[kv])[:, 1]))
        mx.append(best)
    print(f"AUROC locale massima osservata {obs:.3f}; massimo sotto etichette mescolate {np.max(mx):.3f} (mediana {np.median(mx):.3f}); p empirico = {(1 + np.sum(np.array(mx) >= obs)) / (1 + n_perm):.3f}")
''')
md("""
### Risultato che abbiamo ottenuto (fonte: `reports/P5_clusters.md`, `R1-P5_review-clusters.md`; validation)
- k = 4, silhouette 0.569, stabilità ARI 0.999 (domina il sesso, variabile binaria; 0.976 con tutte le feature). Feature: sex, atrial_rate, qrs_duration.
- **Cluster credibili: nessuno.** Nessun IC inferiore locale supera l'AUROC globale (SHD 0.746, stenosi aortica 0.844, rigurgito mitralico 0.767). Esempio: stenosi aortica, cluster 2: locale 0.812 [0.761-0.868] < globale 0.844.
- 40 confronti appaiati locale-globale: IC95% > 0 in 2 (stenosi aortica, cluster 2: +0.021 [+0.003, +0.038] logreg e +0.018 [+0.002, +0.034] Lasso), < 0 in 1, con 0 in 37; i due positivi **non sono confermati** dalla permutazione (p = 0.77).
- I cluster differiscono per **prevalenza** di SHD (chi-quadro 281.7, p = 9e-61), non per **predicibilità**.
- 168 configurazioni totali per P5 (78 solo train, 47 prove, 43 su validation).
- Con `QUICK=True`: ARI e silhouette simili (il clustering è semplice), AUROC locali più rumorose.
""")
md("""
### Decisione
Nessun cluster supera la regola di credibilità → **nessun candidato da congelare da P5**, il gate P5b (stabilità) non serve. Il risultato negativo viene riportato ai partner: i cluster non migliorano la predizione rispetto al modello globale. La cella seguente ricalcola il ramo.
""")
code(r'''
n_cred = int(P5TAB.credibile.sum())
print("cluster credibili ricalcolati:", n_cred, "su", len(P5TAB), "confronti")
print("-> ramo:", "candidati da portare in revisione (P5b se ARI < 0.6 o p tra 0.05 e 0.10)" if n_cred else "nessun candidato: si riporta il risultato negativo")
RESULTS["p5_n_credible"] = n_cred
''')

# ================================================================== P6
md("""
## 7 · Controlli del revisore (fase P6, round R1)
**Obiettivo**: far controllare i risultati da un «revisore scettico»: leakage, riproducibilità, equità per sottogruppi, e anomalie dei dati.
**Perché questo passo**: i numeri migliori sono spesso quelli sbagliati. Verdetto della pipeline: **PASS** (41 affermazioni confermate, 5 deboli, 0 errate; 0 discrepanze numeriche oltre 0.01 di AUROC; nessun leakage di ingresso).
Qui si rifà il controllo più importante emerso nel round 2: le righe **senza `lvef_value`** hanno tutte le 12 etichette = 0 (artefatto dell'etichettatura, non leakage di input), e le AUROC sulle sole righe con LVEF misurata sono più basse.
Gli altri controlli (equità CNN su 16 strati, riproduzione con un altro seed, `candidate.predict`) usano i modelli salvati e non sono riprodotti qui.
""")
code(r'''
nol = train[train.lvef_value.isna()]
print("righe train senza lvef_value:", len(nol), "| massimo, tra le 12 etichette, della prevalenza in queste righe:", float(nol[FLAG_COLS].mean().max()))
has = ~val.lvef_value.isna().values
Xv7, yv7, dv7 = get_xy(val, T)
has7 = ~dv7.lvef_value.isna().values
p7 = GLOBAL.predict_proba(Xv7)[:, 1]
m_all, m_sub = metrics_binary(yv7, p7), metrics_binary(yv7[has7], p7[has7])
print("hist_gb SHD, validation: tutte le righe", fmt(m_all), "| solo righe con LVEF misurata", fmt(m_sub), f"(n={m_sub['n']})")
print("differenza:", round(m_sub["auroc"] - m_all["auroc"], 3))
RESULTS["p6_hgb_with_lvef_auroc"] = m_sub["auroc"]
''')
md("""
### Risultato che abbiamo ottenuto (fonte: `reports/P6_review.md`, handoff P6; validation)
- Righe senza `lvef_value`: train 6 457, validation 514; **tutte e 12 le etichette = 0** (prevalenza SHD 0).
- AUROC sulle sole righe con LVEF misurata: CNN **0.812 [0.798-0.824]** (−0.011), hist_gb **0.732 [0.716-0.746]** (−0.015), LVEF ≤ 45% 0.749 [0.731-0.769].
- Equità CNN: 16/16 strati sopra hist_gb; minimi: procedurale 0.748, ≥ 75 anni 0.788.
- Multiplicità: 433 configurazioni su validation (511 in totale, 1 950 regole e 78 clusterizzazioni solo su train): i migliori valori su validation sono leggermente ottimistici.
- Con `QUICK=True` la hist_gb avrà valori leggermente diversi da 0.732.
""")
md("""
### Decisione
Il verdetto PASS ha ammesso due candidati (CNN `cnn_multilabel` per SHD e `lvef_lte45_global`) e nessun sottogruppo/cluster. Ha **pre-specificato** che nel test finale si riportasse anche l'AUROC sulle sole righe con LVEF misurata, e che i partner chiariscano il significato delle righe senza LVEF prima di diffondere i numeri.
""")

# ================================================================== P7/P8
table = (ROOT / "reports/final_test_table.md")
table_txt = table.read_text().strip() if table.exists() else "(tabella non trovata: vedere reports/P7_final-test.md)"
md(f"""
## 8 · Test finale, sintesi e limiti (fasi P7-P8)
> **Avviso**: il test set si valuta **una sola volta** (fase P7, già eseguita da `tools/final_eval.py` con i soli candidati congelati). **Non va riaperto per scegliere o confrontare modelli**: altrimenti il test diventa un secondo validation e i numeri perdono valore. Questo notebook non lo carica e non lo valuta.

**Tabella del test finale** (copiata da `reports/final_test_table.md`, AUROC con IC95%):

{table_txt}

Lettura (fonte: `reports/P7_final-test.md`, `reports/P8_synthesis.md`):
- SHD composito, CNN: **0.830 [0.820-0.841]** sul test (validation 0.823, calo −0.007 → nessuna evidenza di ottimismo da selezione); modello tabulare sugli stessi pazienti 0.762 [0.750-0.774]. Sulle sole righe con LVEF nota: 0.817 [0.807-0.829]. Classe di accuratezza: *buona*.
- LVEF ≤ 45% (tabulare globale): **0.793 [0.776-0.809]** (validation 0.764); sulle righe con LVEF nota 0.780. Classe: *moderata*. Il «comparatore» SHD (0.775) non è pertinente per questo obiettivo.
- **0 sottogruppi e 0 cluster credibili** (1 950 regole, 40 confronti di cluster, p = 0.77); modifica del QRS **inconclusiva**.

### Limiti
- **SHD non è scompenso cardiaco**: è un'etichetta ecocardiografica (cardiopatia strutturale ≥ moderata).
- Centro unico, solo pazienti che hanno fatto un ecocardiogramma: la generalizzazione ad altri contesti non è provata.
- Un solo seed e una sola configurazione per la CNN; calibrazione delle probabilità non valutata.
- Le righe senza LVEF sono negative per costruzione: chiarire con i partner prima di diffondere i numeri.
- Questo notebook **semplifica** la pipeline: non riproduce la ricerca di 1 950 regole di sottogruppo, i test (ii)-(iii) sul QRS, la selezione delle feature di clustering, il round di revisione sui modelli salvati, né la valutazione sul test.
- Sintesi completa, glossario e prossimi passi: `reports/P8_synthesis.md` (e `reports/FINAL_REPORT.md`).
""")

code(r'''
print("Tempo totale del notebook: %.0f s" % (time.time() - T0_NB))
''')

nb = new_notebook(cells=cells) if HAVE_NBF else new_notebook(cells)
if HAVE_NBF:
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbformat.write(nb, str(OUT))
else:
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("scritto", OUT, "| celle:", len(cells), "| nbformat:", HAVE_NBF)
