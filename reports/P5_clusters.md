# Report P5 · clusters
- Agente: cluster-analyst
- Data: 2026-11-06 15:00
- Stato: PARZIALE

## Obiettivo
Trovare gruppi (cluster) omogenei di pazienti usando poche feature ECG, e verificare se dentro ciascun cluster un modello "locale" predice meglio di un modello "globale" sugli stessi pazienti di validation. Il clustering non conosce l'esito: un cluster ben separato non implica predicibilità, quindi il test che conta è locale vs globale. Ricordiamo che SHD (cardiopatia strutturale) NON coincide con scompenso cardiaco.

## Input letti
- `agentic/PLAYBOOK.md` (sezione P5, definizione di cluster credibile), `agentic/REPORT_TEMPLATE.md`, `legacy_notebooks/03_Clustering_con_selezione_feature.ipynb` (ricetta).
- Target usati (`config.cluster_targets`): `shd_moderate_or_greater_flag`, `aortic_stenosis_moderate_or_greater_flag`, `mitral_regurgitation_moderate_or_greater_flag`.
- Risultati in `results/P5/` (vedi Artefatti). Non sono stati riletti gli Handoff delle fasi precedenti in questa correzione del report.

## Cosa è stato fatto
1. `experiments/P5/select_cluster.py`: pool di clustering = un ECG per paziente del train (n = 26218), standardizzato e con clip a ±4. Selezione delle feature con eliminazione all'indietro (punteggio = media di silhouette e stabilità ARI su due metà; minimo 3 feature) e sparse k-means, per k ∈ {3,4,5}. Solo train, nessun uso del validation.
2. Assegnazione di train e validation ai cluster; profilo e prevalenza dei target per cluster; chi-quadro tra cluster.
3. `experiments/P5/local_vs_global.py`: per ogni target, modello globale (hist_gb) contro modelli locali (logreg, Lasso, hist_gb) e locale con selezione all'indietro delle feature dentro la CV raggruppata (`patient_key`) del solo train del cluster; valutazione sugli stessi pazienti di validation con IC95%. Cluster saltati con meno di 60 positivi in train o 20 in validation (cluster 3 per stenosi aortica e rigurgito mitralico).
4. Soglia di rumore: 30 permutazioni delle etichette per target, ripetendo la procedura (selezione inclusa).
5. Smoke test con `debug_n=2000` (output in `results/P5/debug/`), poi esecuzione completa (`results/P5/run_full.log`).
6. Non eseguito: clustering sugli embedding della CNN di P4 (passo facoltativo).

## Risultati
**Scelta feature e k** (train, 1 ECG/paziente, n = 26218; 78 configurazioni):

| k | feature scelte | silhouette | ARI (stabilità) | punteggio |
|---|---|---|---|---|
| 3 | sex, atrial_rate, qrs_duration | 0.536 | 0.992 | 0.764 |
| 4 | sex, atrial_rate, qrs_duration | 0.569 | 0.999 | 0.784 |
| 5 | sex, atrial_rate, qrs_duration | 0.569 | 0.999 | 0.784 |

Scelto k = 4. Con tutte le feature, stesso k: silhouette 0.205, ARI 0.976. Attenzione: la silhouette alta con 3 feature è in gran parte un effetto della dimensionalità ridotta e del sesso binario (cluster 0 = solo donne/uomini, definiti dal sesso); la stabilità ARI è alta in entrambi i casi.

**Cluster e prevalenza SHD moderata+** (train su ECG del cluster; validation):

| cluster | descrizione | n val | prev. SHD train | prev. SHD val |
|---|---|---|---|---|
| 0 | sesso 0, QRS ~84 ms | 2104 | 0.432 | 0.336 |
| 1 | sesso 1, QRS ~91 ms | 1824 | 0.528 | 0.437 |
| 2 | QRS largo (~140 ms), età ~70 | 563 | 0.735 | 0.705 |
| 3 | frequenza atriale ~288 (fibrillazione/flutter), età ~71 | 135 | 0.779 | 0.667 |

Chi-quadro (3 gradi di libertà) tra cluster: SHD validation χ² = 281.7, p = 9e-61; stenosi aortica validation χ² = 75.9, p = 2e-16; rigurgito mitralico validation χ² = 101.1, p = 9e-22. I cluster differiscono per prevalenza (in gran parte perché riflettono età e QRS).

**Locale vs globale, validation** (AUROC [IC95%]; "globale" = hist_gb globale sugli stessi pazienti; Δ = migliore locale − globale stessi pazienti):

| target | AUROC globale su tutta la validation (n pos) | cluster | n val / pos | AUROC globale stessi pazienti | miglior locale | Δ |
|---|---|---|---|---|---|---|
| SHD | 0.746 [0.733-0.759] (1990) | 0 | 2104/706 | 0.722 [0.699-0.744] | hist_gb 0.723 [0.700-0.744] | +0.001 |
| SHD | | 1 | 1824/797 | 0.702 [0.679-0.723] | logreg 0.697 [0.673-0.718] | -0.005 |
| SHD | | 2 | 563/397 | 0.750 [0.709-0.790] | hist_gb 0.740 [0.701-0.783] | -0.009 |
| SHD | | 3 | 135/90 | 0.651 [0.566-0.741] | Lasso 0.685 [0.604-0.772] | +0.034 (IC include 0: [-0.046, +0.114]) |
| Stenosi aortica | 0.844 [0.819-0.868] (252) | 0 | 2104/91 | 0.835 [0.798-0.874] | logreg 0.834 | -0.001 |
| Stenosi aortica | | 1 | 1824/76 | 0.852 [0.809-0.889] | Lasso 0.857 [0.819-0.889] | +0.005 |
| Stenosi aortica | | 2 | 563/73 | 0.792 [0.739-0.850] | logreg 0.812 [0.761-0.868] | +0.021 (IC [+0.003, +0.038]) |
| Rigurgito mitralico | 0.767 [0.735-0.789] (282) | 0 | 2104/117 | 0.769 [0.728-0.817] | local_sel 0.768 | -0.001 |
| Rigurgito mitralico | | 1 | 1824/66 | 0.748 [0.691-0.798] | hist_gb 0.741 | -0.007 |
| Rigurgito mitralico | | 2 | 563/82 | 0.678 [0.608-0.740] | hist_gb 0.662 | -0.015 |

Il modello locale con selezione all'indietro (`local_sel_logreg`) non batte mai il globale in modo significativo (Δ da -0.032 a +0.021, tutti gli IC includono 0, tranne nessuno in positivo).

**Soglia di rumore** (30 permutazioni, validation; p minimo ottenibile = 1/31 = 0.032):

| target | AUROC massimo osservato | massimo sotto etichette mescolate (media / p95) | p empirico sul massimo AUROC | p empirico sul massimo guadagno locale-globale |
|---|---|---|---|---|
| SHD | 0.740 | 0.550 / 0.629 | 0.032 | 0.52 |
| Stenosi aortica | 0.857 | 0.556 / 0.592 | 0.032 | 0.77 |
| Rigurgito mitralico | 0.768 | 0.541 / 0.587 | 0.032 | 0.94 |

Lettura: il massimo AUROC supera il rumore (le feature ECG contengono segnale), ma questo è già vero per il modello globale. Il guadagno dei modelli locali rispetto al globale è compatibile col caso (p = 0.52-0.94): i cluster non aggiungono predicibilità.

**Verifica di credibilità (PLAYBOOK)**: un cluster è credibile se AUROC cluster − AUROC globale su tutta la validation ≥ 0.02 con limite inferiore IC95% > AUROC globale. Nessun cluster lo soddisfa: l'AUROC massimo per target nei cluster (SHD 0.750, lo 0.709 < 0.746; stenosi aortica 0.857, lo 0.819 < 0.844; rigurgito mitralico 0.769, lo 0.728 < 0.767 con Δ < 0.02) non supera la soglia. Il cluster 2 per la stenosi aortica (+0.021 contro il globale sugli stessi pazienti) è inferiore al globale complessivo (0.812 < 0.844).

## Decisioni e motivazioni
- Scelto k = 4 (punteggio più alto, k = 5 equivalente ma con un cluster in più senza guadagno), feature sex/atrial_rate/qrs_duration (ARI ≥ 0.99).
- Nessun candidato salvato: nessun cluster credibile. È un risultato negativo valido: i cluster separano gruppi con prevalenza diversa (età, QRS, ritmo) ma non gruppi in cui la predizione migliora.
- Scartati: clustering sugli embedding CNN (passo facoltativo, non eseguito per mancanza di tempo/priorità).
- Il confronto "globale vs locale" usa gli stessi pazienti, come richiesto: confrontare l'AUROC di un sottogruppo con quello globale su pazienti diversi sarebbe fuorviante (la prevalenza cambia).

## Configurazioni provate
- Selezione feature (solo train, senza validation): 78 configurazioni (backward k=3,4,5: 69 insiemi + sparse k-means 9), registrate in `results/registry.csv` (E0014, senza `--uses-val`).
- Smoke test `debug_n=2000`: 47 configurazioni, registrate (E0015, `--uses-val`).
- Esecuzione completa locale vs globale in validation: 3 modelli globali + 10 cluster × 4 modelli locali = 43 configurazioni che usano il validation. **NON risultano registrate in `results/registry.csv`** (vedi Problemi aperti).
- Totale: 78 + 47 + 43 = 168 configurazioni provate (125 registrate; 90 hanno usato il validation, di cui 43 non registrate). Il validation non è stato usato per scegliere k né le feature del clustering, ma i confronti locale/globale sono sul validation: nessuna selezione è stata fatta su di esso.

## Artefatti
- Script: `experiments/P5/select_cluster.py`, `experiments/P5/local_vs_global.py`.
- `results/P5/selection_summary.json`, `selection_log.csv`, `best_per_k.csv`, `cluster_model.joblib`, `train_cluster.npy`, `val_cluster.npy`.
- `results/P5/cluster_profile.csv`, `chi2.csv`, `local_vs_global.csv`, `local_selected_features.csv`, `permutation.csv`, `perm_*.json`, `result_*.joblib`, `global_*.joblib`, `run_full.log`.
- Smoke test: `results/P5/debug/`. Nessun candidato (`results/P5/candidates/` non creata).

## Problemi aperti
- L'esecuzione completa di `local_vs_global.py` (43 configurazioni sul validation) non è stata registrata con `log-exp`. Va aggiunta: `python3 tools/state.py log-exp --phase P5 --desc "locale vs globale, run completa (3 target, 10 cluster)" --uses-val --n-configs 43`. In questa correzione ho potuto modificare solo il report, quindi non l'ho eseguita.
- Passo facoltativo (cluster sugli embedding CNN di P4) non eseguito.
- Il p empirico minimo con 30 permutazioni è 0.032: non permette di distinguere p più piccoli. Il test sul massimo AUROC è poco informativo (il globale supera già il rumore); quello sul guadagno locale-globale è il test rilevante.
- Con 3 feature il clustering è dominato dal sesso: i cluster 0 e 1 differiscono solo per sesso e ripartiscono la popolazione in modo non clinicamente specifico.
- Cluster 3 (n val = 135): IC larghi, conclusioni deboli.

## Handoff
STATUS: PARZIALE
RISULTATI_CHIAVE:
- FEATURE_PER_CLUSTER: sex, atrial_rate, qrs_duration (clustering); le feature predittive locali selezionate sono in `results/P5/local_selected_features.csv`.
- K: 4 (silhouette 0.569).
- STABILITA_ARI: 0.999 (due metà del pool di train; con tutte le feature 0.976).
- CLUSTER_CREDIBILI: [] (nessuno: nessun lo-IC locale supera l'AUROC globale di validation: SHD 0.746, stenosi aortica 0.844, rigurgito mitralico 0.767).
- P_EMPIRICO: 0.032 sul massimo AUROC (30 permutazioni, tutti e 3 i target); 0.52 / 0.77 / 0.94 sul guadagno locale-globale (SHD / stenosi / rigurgito).
- N_CONFIGURAZIONI: 168 (78 selezione senza validation + 47 smoke + 43 run completa in validation, di cui 43 da registrare).
- Chi-quadro tra cluster su SHD in validation: 281.7, p = 9e-61 (i cluster differiscono per prevalenza, non per predicibilità).
RACCOMANDAZIONE:
- Nessun candidato da congelare da P5: riportare ai partner il risultato negativo (i cluster non migliorano la predizione rispetto al modello globale).
- Registrare le 43 configurazioni mancanti in `registry.csv` prima della revisione P6. Il gate P5b non è necessario (nessun cluster credibile).
