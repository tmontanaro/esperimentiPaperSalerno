# PLAYBOOK: fasi, definizioni e regole di decisione

Lo modifica l'umano, non gli agenti. Le soglie numeriche vivono in `config.thresholds` di `agentic/pipeline_state.json`
(modificabili con `python3 tools/state.py set-config thresholds.good_auroc 0.85`). I valori sotto sono quelli iniziali.

## Definizioni operative (valgono per tutti)
- **Split**: train per apprendere e scegliere in CV raggruppata; validation per confrontare e verificare; test solo in P7.
- **Segnale tabulare** (`TABULAR_SIGNAL: yes`): AUROC_val(tutte le feature ECG, miglior modello) − AUROC_val(solo età+sesso, miglior modello) ≥ `tabular_signal_min_gain` (0.02)
  **e** limite inferiore IC95% dell'AUROC con tutte le feature > 0.55.
- **Target promettente**: `n_pos_val ≥ min_pos_val` (25) e AUROC_val con IC95% inferiore ≥ `promising_target_auroc_lo` (0.65).
- **Sottogruppo o cluster credibile** (tutte le condizioni):
  1. definito usando solo il train (o la CV sul train); il validation serve solo a verificarlo;
  2. in validation: `n ≥ min_n_val` (100), positivi ≥ `min_pos_val` (25) e negativi ≥ 25;
  3. AUROC_val nel sottogruppo − AUROC_val globale sull'intero validation ≥ `credible_subgroup_margin` (0.02), con **limite inferiore dell'IC95% > AUROC_val globale**;
  4. confronto equo "modello locale vs globale sugli stessi pazienti" riportato;
  5. per i cluster: p empirico del test con etichette mescolate ≤ 0.05 e stabilità ARI ≥ 0.6.
- **Buona accuratezza** (per il messaggio ai partner): AUROC_test ≥ `good_auroc` (0.80) con limite inferiore IC95% ≥ `good_auroc` − 0.05. Sotto: "moderata" (0.70-0.80) o "debole" (<0.70).
- **Multiplicità**: il numero di configurazioni provate (`results/registry.csv`) va sempre riportato accanto a ogni "scoperta".

## Candidati (contratto per la valutazione finale)
Ogni modello/sottogruppo/cluster che vuoi proporre al test va salvato con `lib.save_candidate(...)` in `results/<ID>/candidates/<nome>/` con:
- `candidate.json` (nome, target, tipo, descrizione, metriche di validation),
- `candidate.py` che definisce `predict(df, lib) -> DataFrame` con colonne `score` (probabilità o punteggio) e `in_scope` (bool: il paziente appartiene al sottogruppo/cluster; `True` per un modello globale) con lo stesso indice di `df`,
- i file del modello (joblib) accanto a `candidate.py`, caricati con percorso relativo a `__file__`.
`df` contiene le colonne ECG e `qrs_bin`/`age_bin`; per le forme d'onda usa `lib.load_data()`-style con `row_in_split` (vedi esempio di P4).
Esempio minimo (modello globale):
```python
import joblib
from pathlib import Path
import pandas as pd
_M = joblib.load(Path(__file__).parent / "model.joblib")
def predict(df, lib):
    X = df[lib.ECG_FEATURES].astype(float)
    return pd.DataFrame({"score": _M.predict_proba(X)[:, 1], "in_scope": True}, index=df.index)
```
Un candidato di sottogruppo restituisce `in_scope` calcolato dalla regola (es. `df.qrs_duration >= 120`) e `score` da un modello addestrato **solo sul train**.
Un candidato che non rispetta il contratto non può essere valutato.

---

## Fasi

### P0 · data-audit (data-auditor)
Trova i file (la cartella è in `config.data_dir`), verifica integrità: nessun paziente condiviso tra gli split (`Data.integrity`), un ECG per paziente in val/test, forme d'onda presenti con shape `(N,1,2500,12)`,
valori mancanti, prevalenza delle 12 etichette in train e validation, distribuzione di QRS ed età, controllo che le colonne corrispondano allo schema atteso. Verifica se c'è una GPU (`torch.cuda.is_available()`).
Output: `results/P0/data_profile.json`. Handoff: `INTEGRITA: OK|PROBLEMI`, `N_TRAIN`, `N_VAL`, `PREV_TARGET_TRAIN`, `WAVEFORMS: presenti|mancanti`, `GPU: sì|no`.
STATUS FALLITO se c'è sovrapposizione di pazienti tra split o mancano i metadati.

### P1 · baseline (baseline-modeler)
Sul target composito: due insiemi di feature (solo età+sesso; tutte le 7 feature ECG) × 5 modelli (`lib.model_zoo`), valutati su validation con IC95%; CV raggruppata sul train come controllo di coerenza (se CV ≫ validation c'è un problema);
permutation importance. Salva il miglior modello globale come candidato `results/P1/candidates/global_best` (serve come confronto in P7).
Handoff: `TABULAR_SIGNAL: yes|no`, `AUROC_VAL_ETA_SESSO`, `AUROC_VAL_TUTTE` (con IC), `MIGLIOR_MODELLO`, `CANDIDATO_GLOBALE: results/P1/candidates/global_best`.

### P2 · conditions (condition-analyst)
Riferimento: `legacy_notebooks/01_SHD_per_condizione_e_sottogruppi.ipynb`.
(a) AUROC su validation per il composito e per ciascuna delle 11 condizioni (3 modelli, IC95%), saltando quelle con pochi positivi;
(b) stratificazione per QRS, sesso, età, setting, etnia: modello globale vs modello locale sugli stessi pazienti (`race_ethnicity` e `location_setting` SOLO per stratificare e controllare equità, mai come input);
(c) scoperta di sottogruppi con regole semplici (singole e coppie AND) usando predizioni out-of-fold del train; poi verifica su validation. Registra il **numero di regole provate**.
Se `config.tabular_signal` è false: analisi ridotta (solo (a) e (b), e documentare che (c) non è stata eseguita per assenza di segnale).
Salva come candidati solo i sottogruppi credibili. Handoff: `TARGET_PROMETTENTI: [..]`, `SOTTOGRUPPI_CREDIBILI: [nome: regola, n_val, AUROC [IC]]`, `N_REGOLE_PROVATE`.

### P3 · lvef-qrs (lvef-qrs-analyst)
Riferimento: `legacy_notebooks/02_LVEF_e_ruolo_del_QRS.ipynb`.
(a) Regressione di `lvef_value` (4 modelli, MAE/R² contro baseline "media del train") e classificazione `lvef_lte_45_flag`;
(b) ipotesi dei medici "il QRS modifica la relazione feature-LVEF" testata in **tre modi indipendenti**: modelli per fascia di QRS vs globale; **test formale di interazione**
(modello additivo vs modello con termini `qrs × feature` sul train, test del rapporto di verosimiglianza o confronto in CV raggruppata); coppie di pazienti simili con LVEF diverso;
(c) il parametro "a parità di ..." usato dai colleghi è SCONOSCIUTO: ripeti l'analisi delle coppie con ALMENO 3 insiemi `MATCH_ON` diversi
(es. [età, sesso, frequenza], [età, sesso], [età, sesso, frequenza, PR, QTc]) e riporta la sensibilità del risultato; registra ciascun insieme con `log-exp`.
Handoff: `QRS_MODIFICA: supportata|non_supportata|inconclusiva` + le 3 evidenze con numeri, `MATCH_ON_TESTATI`, `LVEF_REG: MAE=..,R2=..`, `LVEF_LTE45_AUROC: ..`.
"Supportata" richiede che almeno 2 dei 3 test su 3 concordino con IC/p dichiarati; altrimenti "inconclusiva" o "non_supportata".

### P4 · waveform (waveform-modeler)
Riferimento: `legacy_notebooks/04_Modelli_su_forme_d_onda.ipynb`. CNN 1D multi-etichetta (composito + condizioni) con loss che ignora le etichette mancanti.
Prima smoke test con `debug_n=2000` e 1-2 epoche; poi addestramento completo SOLO se c'è una GPU (`GPU: sì` in P0). Senza GPU: STATUS PARZIALE, lascia lo script pronto e spiega come lanciarlo altrove.
Valuta su validation per condizione e per fascia di QRS, confronta con la baseline tabulare (P1) sulle stesse righe; salva predizioni di validation (non dati paziente-livello nei report) e il candidato `results/P4/candidates/cnn_multilabel`
(il candidato carica pesi e definizione del modello da file accanto a `candidate.py` e legge le forme d'onda tramite `lib.load_data(allow_test=...)`-compatibile: usa `data.waveforms(split, df.row_in_split)`).
Handoff: `CNN_AUROC_VAL_COMPOSITO`, `GAIN_VS_TABULARE`, `GPU_USATA`, `TARGET_MIGLIORI: [...]`, `EPOCHE`.

### P5 · clusters (cluster-analyst)
Riferimento: `legacy_notebooks/03_Clustering_con_selezione_feature.ipynb`.
(1) Scelta delle feature per il clustering: eliminazione all'indietro (silhouette + stabilità ARI) e sparse k-means; (2) clustering finale; (3) i cluster differiscono per esito? (chi-quadro sul target);
(4) per ciascun target in `config.cluster_targets`: modello globale vs locali sugli stessi pazienti di validation, con selezione all'indietro delle feature predittive **dentro la CV del cluster sul train**;
(5) soglia di rumore con etichette mescolate (≥ 30 permutazioni, selezione inclusa). Se `config.clusters_mode == "reduced"`: solo (1)-(3) e un'unica prova (4) sul composito.
Facoltativo se P4 è completata: clustering sugli embedding della CNN (si possono usare gli embedding di train per DEFINIRE i cluster; la valutazione resta su validation).
Salva come candidati solo i cluster credibili. Handoff: `FEATURE_PER_CLUSTER`, `K`, `STABILITA_ARI`, `CLUSTER_CREDIBILI: [...]`, `P_EMPIRICO`, `N_CONFIGURAZIONI`.

### P6 · review (skeptic-reviewer)
Revisione indipendente (non fidarti delle narrazioni: ricalcola dai file). Vedi il file dell'agente per la checklist.
Handoff: `VERDETTO: PASS|REVISE`, `CANDIDATI_AMMESSI: [cartelle]`, `REVISIONI_RICHIESTE: [fase: azione]`, `AFFERMAZIONI: n_supportate/n_deboli/n_non_supportate`.

### P7 · final-test (final-evaluator)
Esegue UNA volta `python3 tools/final_eval.py` sui candidati di `agentic/FROZEN.json` e interpreta: AUROC con IC nel sottogruppo, confronto col modello globale sugli stessi pazienti, calo rispetto alla validation (un calo > 0.05 è segnale di ottimismo da selezione).
Richiede approvazione umana (`python3 tools/state.py approve P7`). Handoff: tabella test, `CALO_VAL_TEST` per candidato, `CLASSE_ACCURATEZZA: buona|moderata|debole` per candidato.

### P8 · synthesis (synthesizer)
Report finale `reports/P8_synthesis.md` (+ `reports/FINAL_REPORT.md` copia) spiegato per i partner. Struttura in `.claude/agents/synthesizer.md`.

---

## Gate: cosa fa l'orchestratore dopo ogni fase
L'orchestratore esegue SOLO azioni con `tools/state.py` e annota ogni decisione con `state.py note`. Non fa analisi.

### Gate P0
Se STATUS ≠ OK o `INTEGRITA: PROBLEMI` → `note --kind issue` e lascia che il driver si fermi. Se `GPU: no` → `note --kind assumption "P4 girerà in smoke test; addestramento completo da fare su macchina con GPU"`.

### Gate P1
- `TABULAR_SIGNAL: yes` → `set-config tabular_signal true`.
- `TABULAR_SIGNAL: no` → `set-config tabular_signal false`, `set-config clusters_mode '"reduced"'`, `set-priority P4 10` (le forme d'onda diventano la via principale) e `note` con la motivazione.

### Gate P2
- `set-config cluster_targets` = fino a 3 target: il composito + i `TARGET_PROMETTENTI` di P2 (in ordine di AUROC). Se nessuno è promettente: solo il composito.
- Se P2 ha `SOTTOGRUPPI_CREDIBILI` ma `N_REGOLE_PROVATE` > 500 e nessuna replica indipendente → `note --kind issue` (il revisore ne terrà conto).

### Gate P3
- `QRS_MODIFICA: supportata` → aggiungi UN follow-up: `add-phase --id P3b --name qrs-stratified --agent condition-analyst --depends-on P2,P3 --blocks P6 --gate --goal "Per i TARGET_PROMETTENTI di P2 e per lvef_lte_45_flag, modelli separati per fascia di QRS (<120, >=120), confronto con il globale sugli stessi pazienti; salva i candidati credibili"`.
- Altrimenti solo `note`.

### Gate P4
- `GAIN_VS_TABULARE ≥ 0.05` → `set-config waveform_dominant true` e `note`.
- Se STATUS PARZIALE per mancanza di GPU → `note --kind issue "CNN non addestrata a pieno: esito P4 provvisorio"`; non bloccare le altre fasi.

### Gate P5
- Se esiste un cluster credibile con `STABILITA_ARI < 0.6` o `P_EMPIRICO` tra 0.05 e 0.10: UN follow-up `add-phase --id P5b --name cluster-stability --agent cluster-analyst --depends-on P5 --blocks P6 --gate --goal "Ripetere la selezione e il clustering con 5 semi/ricampionamenti diversi e riportare la stabilità dei cluster candidati"`.
- Altrimenti solo `note`.

### Gate P6 (il più importante)
Leggi `handoff P6` e `reports/P6_review.md`.
- `VERDETTO: REVISE` e `review_rounds < max_review_rounds` → per ogni voce di `REVISIONI_RICHIESTE` crea una fase di revisione con `add-phase --id R<n>-<fase> --revises <fase> --agent <agente della fase> --blocks P6 --reopen`
  (e `--depends-on` alle dipendenze della fase originale; `--reopen` riapre P6, che attenderà la revisione). Poi `python3 tools/state.py bump-review` (rifiuta oltre `max_review_rounds`). Le fasi di revisione non consumano il tetto dei follow-up.
- `VERDETTO: PASS` oppure round esauriti → scrivi `agentic/FROZEN.json` con `{"comparator_dir": "<CANDIDATO_GLOBALE di P1>", "candidates": [{"name","dir","target"}...]}` contenente SOLO i `CANDIDATI_AMMESSI` dal revisore, poi `python3 tools/state.py freeze`.
- Se `CANDIDATI_AMMESSI` è vuoto → scrivi FROZEN.json con `candidates: []`, `python3 tools/state.py skip P7 --reason "nessun candidato credibile: risultato negativo"`. P8 racconterà il risultato negativo.
- Prima del test l'umano deve approvare P7 (il driver si ferma da solo).

### Regole generali dei gate
- Massimo `max_followups` fasi aggiuntive in totale (`state.py` rifiuta oltre). Preferisci non aggiungere fasi se l'evidenza è ambigua: scrivi `note --kind issue`.
- Non cambiare mai soglie né definizioni. Non modificare i report. Non toccare il test.
- Se un report è `PARZIALE`, decidi se è sufficiente per proseguire o se serve un follow-up, e scrivi il motivo.
