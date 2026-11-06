# Report P6 · review
- Agente: skeptic-reviewer
- Data: 2026-11-06 15:45
- Stato: PARZIALE

## Obiettivo
Revisione indipendente e scettica delle fasi P2-P5 prima dell'apertura (unica) del test set: cercare leakage (uso di informazioni che il modello non avrebbe nella realtà), ricalcolare i numeri chiave dai file, valutare quanto il validation è stato "consumato" da scelte multiple, verificare che le etichette "credibile"/"supportata" rispettino il PLAYBOOK e decidere quali candidati possono andare al test. Promemoria: SHD (cardiopatia strutturale moderata o più) NON coincide con lo scompenso cardiaco.

## Input letti
- Handoff/report: `reports/P2_conditions.md`, `reports/P3_lvef-qrs.md`, `reports/P4_waveform.md`, `reports/P5_clusters.md`; stato in `agentic/pipeline_state.json`; `agentic/DECISIONS.md`; `results/registry.csv`; `agentic/PLAYBOOK.md`.
- Script: `experiments/P2/conditions.py`, `experiments/P3/lvef_qrs.py`, `experiments/P4/train.py`, `experiments/P4/model_def.py`, `experiments/P5/select_cluster.py`, `experiments/P5/local_vs_global.py`; `lib/echonext_lib.py` (righe 26-41: `ECG_FEATURES`, `FLAG_COLS`, `ECHO_ONLY_COLS`); `tools/final_eval.py` (righe 47-75).
- File di risultato aggregati (nessun file di dati paziente-livello aperto): `results/P2/c_rules_val.csv`; `results/P3/a_regression.csv`, `a_classification.csv`, `b1_stratification.csv`, `b2_interaction.csv`, `b3_pairs.csv`; `results/P4/val_per_target.csv`, `val_by_qrs.csv`, `history.json`; `results/P5/local_vs_global.csv`, `permutation.csv`, `chi2.csv`, `selection_summary.json`.
- Candidati: `results/P1/candidates/global_best/candidate.py` (comparatore), `results/P3/candidates/lvef_lte45_global/{candidate.py,candidate.json}`, `results/P4/candidates/cnn_multilabel/{candidate.py,candidate.json,config.json}`.

## Cosa è stato fatto
Limite importante di questa sessione: l'esecuzione di comandi (Bash) è stata negata dall'ambiente. Quindi NON ho potuto eseguire `tools/state.py brief/handoff`, `tools/validate_report.py`, né rieseguire script o `candidate.predict` su un campione. Tutte le verifiche sono state fatte leggendo il codice e ricalcolando/confrontando i numeri dei report con i file CSV/JSON aggregati in `results/`. I report sono stati letti per intero (la loro sezione Handoff inclusa).
1. Leakage: lettura riga per riga degli script di P2-P5 alla ricerca di colonne di `ECHO_ONLY_COLS` tra gli input, di CV non raggruppata, di accessi al test.
2. Riproduzione: per ogni fase, 3-6 numeri chiave confrontati con i file di `results/` (tolleranza 0.01 AUROC).
3. Multiplicità: somma delle configurazioni in `results/registry.csv` per fase e confronto con quanto dichiarato.
4. Definizioni: controllo dei criteri del PLAYBOOK su ogni etichetta "credibile"/"promettente"/"supportata".
5. Classificazione delle 27 righe RISULTATI_CHIAVE degli Handoff P2-P5.
6. Contratto dei candidati: controllo statico di `candidate.py` rispetto al PLAYBOOK e a come `tools/final_eval.py` li chiama.

## Risultati
**1. Leakage: nessun leakage trovato (verifica statica del codice).**

| Fase | Input dei modelli | CV raggruppata per paziente | Test | Esito |
|---|---|---|---|---|
| P2 | `lib.get_xy` → 7 `ECG_FEATURES`; `location_setting`/`race_ethnicity` solo in `strata_vars` (conditions.py r.58) per stratificare; regole su sole feature ECG (r.85-89) | OOF con `lib.oof_proba(..., patient_key)` (r.104) | mai | OK |
| P3 | `F = lib.ECG_FEATURES` (r.27, r.35); `lvef_value` solo come esito; interazioni `qrs×feature` su F (r.105-116) | `GroupKFold(5)` su `patient_key` (r.143); Wald con errori raggruppati per paziente (r.117-125) | mai | OK |
| P4 | solo forme d'onda (`--use-tab` non usato; `assert a.use_tab is False`, train.py r.175); le altre 11 etichette eco sono **uscite** ausiliarie (multi-etichetta), non input: lecito | split predefinito train/val (pazienti disgiunti per P0) | candidato legge il test solo con `ECHONEXT_ALLOW_TEST=1`, impostato da `final_eval.py` r.47 | OK |
| P5 | clustering su F (select_cluster.py r.16-22), modelli locali su F (local_vs_global.py r.71) | `GroupKFold(3)` su `patient_key` nella selezione dentro il cluster (r.53) | mai | OK |

Uso del validation per l'addestramento: nessun modello è addestrato su righe di validation. Usi "morbidi" del validation (non leakage, ma ottimismo da selezione): P2 sceglie i 2 target focali e il "miglior modello" per target guardando l'AUROC di val (conditions.py r.41-46); P3 sceglie `best_cls` sul val (lvef_qrs.py r.79, 5 modelli entro 0.008 di AUROC); P4 monitora l'AUROC di val per epoca ma usa l'epoca finale fissata a priori (12); P5 decide l'eleggibilità dei cluster anche in base al numero di positivi in val (r.78, soglia 20), effetto trascurabile.

**2. Riproduzione dai file (tutte entro tolleranza, 0 discrepanze di AUROC).**

| Fase | Affermazione nel report | Valore nel file | File |
|---|---|---|---|
| P2 | composito globale val 0.746 [0.733-0.759] | 0.7464 [0.7331-0.7595] | P5/local_vs_global.csv r.2, P2/c_rules_val.csv |
| P2 | regola 2 composito: 0.779 [0.731-0.832], n=425, margine +0.033, non credibile | 0.7793 [0.7315-0.8315], n=425, 157 pos, margine 0.0329; IC inf 0.731 < 0.746 | P2/c_rules_val.csv r.3 |
| P2 | stenosi aortica regola 1: 0.950, 8 positivi | 0.9496, 8 pos | P2/c_rules_val.csv r.7 |
| P2 | 650 regole/target | 38 atomi + C(38,2) − 91 coppie sulla stessa feature = 38 + 612 = 650 (×3 = 1950) | ricalcolo da conditions.py r.83-97 |
| P3 | LVEF<=45 AUROC 0.764 [0.748-0.782] | 0.7643 [0.7478-0.7815] | P3/a_classification.csv r.2 |
| P3 | LVEF MAE 9.54 [9.31-9.79], R² 0.163 | 9.538 [9.308-9.794], 0.1630 | P3/a_regression.csv r.5 |
| P3 | Wald per paziente p<1e-4 / 0.0069; CV ΔMSE +0.28 [-0.03,0.56]; R² 0.1400→0.1413 | p=3.6e-8 / 0.00686; 0.276 [-0.028, 0.557]; 0.14005→0.14128 | P3/b2_interaction.csv |
| P3 | ΔR² <100: +0.007 [0.001,0.013] | 0.00668 [0.00069, 0.01330] | P3/b1_stratification.csv |
| P4 | CNN composito 0.823 [0.811-0.834]; gain +0.077 [0.063,0.091] | 0.8232 [0.8110-0.8336]; 0.0768 [0.0625, 0.0909] | P4/val_per_target.csv r.2 |
| P4 | epoca 12 = 0.823, picco 0.831 all'epoca 8 | 0.8232 (ep.12), 0.8307 (ep.8) | P4/history.json |
| P4 | QRS>=120: CNN 0.845 vs tab 0.752 | 0.8452 [0.814-0.883] vs 0.7517 [0.710-0.798] | P4/val_by_qrs.csv |
| P5 | SHD cluster 3: Lasso 0.685, Δ +0.034 [-0.046,+0.114] | 0.6854, +0.0341 [-0.0457, 0.1141] | P5/local_vs_global.csv r.20 |
| P5 | stenosi aortica cluster 2: logreg 0.812, Δ +0.021 [+0.003,+0.038] | 0.8124, +0.0206 [0.0030, 0.0377] | P5/local_vs_global.csv r.35 |
| P5 | p empirico guadagno 0.52/0.77/0.94; p max-AUROC 0.032 | 0.516/0.774/0.935; 0.0323 (30 perm.) | P5/permutation.csv |
| P5 | χ² SHD val 281.7, p 9e-61; K=4, silhouette 0.569, ARI 0.999 | 281.66, 9.3e-61; 0.5692, 0.9992 | P5/chi2.csv, selection_summary.json |

Discrepanze NON numeriche trovate:
- P3, coppie: il report dice "tenuto il 20% di coppie più vicine", ma per MATCH_ON A e B quasi tutte le distanze sono 0 (appaiamento esatto su età intera e sesso) e il filtro al quantile 20% non elimina nulla: B train ha `n_pairs` = 23637 = tutti i pazienti del pool (b3_pairs.csv r.4). Inoltre le coppie sono "vicino più prossimo" orientate, quindi coppie reciproche compaiono due volte: insieme al bootstrap per coppia (non per paziente) gli IC sono troppo stretti (es. train A: differenza 6 [6-6] ms, IC degenere).
- P3, `candidate.json` di `lvef_lte45_global`: la descrizione dice "addestrato sul train (solo righe con LVEF)", ma il codice addestra su tutte le righe di train con `lvef_lte_45_flag` disponibile (lvef_qrs.py r.74 usa `tr_all`; in val 4626 righe con flag vs 4112 con LVEF). Il modello è corretto, la descrizione no.
- P5: nel testo "Il modello locale con selezione all'indietro non batte mai il globale in modo significativo" è vero solo per `local_sel_logreg`; nello stesso cluster 2 della stenosi aortica due modelli locali hanno IC appaiato interamente sopra 0 (logreg +0.021 [0.003, 0.038]; lasso +0.018 [0.002, 0.034], local_vs_global.csv r.35-36). Su 40 confronti locale-globale e con p empirico del massimo guadagno 0.77 è compatibile col caso, ma va detto esplicitamente, non nascosto in tabella.

**3. Selezione sul validation e multiplicità.**

| Fase | Configurazioni registrate con uso del val | Senza val | Dichiarate nel report | Note |
|---|---|---|---|---|
| P1 | 10 | - | 10 | smoke test non registrato (DECISIONS 14:21:37) |
| P2 | 308 (E0002 148 + E0003 160) | 1950 regole su train/OOF | 308 + 1950 | 15 regole verificate su val |
| P3 | 17 (E0004-E0011) | test di interazione su train | 17 | coerente |
| P4 | 2 (E0012-E0013) | - | 2 | una sola configurazione, epoca fissata |
| P5 | 47 (E0015) | 78 (E0014) | 47 + 43 + 78 = 168 | **43 configurazioni su val NON registrate** |
| Totale | 384 registrate (427 reali) | 2028 | | |

- Sottogruppi (P2): regole definite solo su OOF del train, poi verificate su val: procedura corretta (criterio 1 del PLAYBOOK rispettato). Gate: 1950 regole > 500 senza replica indipendente; non produce danni perché nessuna regola è stata dichiarata credibile.
- Cluster (P5): definiti solo su train (feature e k scelti senza target né val). Il p empirico include la selezione all'indietro delle feature predittive e la scelta del massimo tra cluster e modelli (la statistica è il massimo su tutti i cluster e i 4 modelli, local_vs_global.py r.89-96, ricalcolata in ogni permutazione r.134-142): corretto. Non include la selezione delle feature del clustering, ma questa non usa l'esito, quindi è legittimo tenerla fissa. Limite: 30 permutazioni → risoluzione minima 0.032.
- IC su pazienti indipendenti: in val 1 ECG per paziente (P0), quindi il bootstrap per riga in val è per paziente: corretto per P2, P3 (classificazione, regressione, stratificazione), P4, P5. Eccezione: le coppie di P3 (paziente presente in più coppie), vedi sopra.

**4. Definizioni.** Nessuna etichetta "credibile" è stata assegnata in P2 (0 su 15 regole) e P5 (0 su 10 cluster×target). Verifica dei casi limite: composito regola 2 (margine +0.033 ma IC inferiore 0.731 < 0.746 → criterio 3 fallito); stenosi aortica regole 1-5 (8-18 positivi < 25 → criterio 2 fallito); stenosi aortica cluster 2 (AUROC 0.812 < globale 0.844 → criterio 3 fallito; p empirico 0.77 → criterio 5 fallito). Le assenze di credibilità sono quindi corrette. "Target promettente": verificato per stenosi aortica (IC inf 0.819 ≥ 0.65, 252 pos) e LVEF<=45 (0.748, 866 pos).
Stabilità ARI 0.999 di P5: formalmente ≥ 0.6, ma è quasi automatica perché il sesso (binario) domina 2 dei 4 cluster; non è un'evidenza di struttura clinica.

**5. Affermazioni vs dati (27 righe RISULTATI_CHIAVE).**

| Fase | Riga | Giudizio | Motivo |
|---|---|---|---|
| P2 | TARGET_PROMETTENTI | SUPPORTATA | AS e LVEF<=45 ricalcolati; scelta del miglior modello su val = lieve ottimismo dichiarato |
| P2 | Non promettenti | SUPPORTATA | IC inf < 0.65 o < 25 positivi |
| P2 | SOTTOGRUPPI_CREDIBILI: [] | SUPPORTATA | c_rules_val.csv: nessuna riga soddisfa n/pos/margine/IC |
| P2 | N_REGOLE_PROVATE 1950 | SUPPORTATA | ricalcolato 650×3 |
| P2 | locale vs globale per strati | SUPPORTATA | coerente con lo script; b_strata.csv non riletto riga per riga |
| P2 | composito 0.746 [0.733-0.759] | SUPPORTATA | ricalcolato |
| P3 | QRS_MODIFICA: inconclusiva | SUPPORTATA | vedi sotto |
| P3 | (i) stratificazione | SUPPORTATA | b1_stratification.csv |
| P3 | (ii) interazione | SUPPORTATA | b2_interaction.csv |
| P3 | (iii) coppie, IC sopra 50% | DEBOLE | IC per coppia non per paziente, coppie reciproche duplicate, filtro 20% non applicato in A/B; misura associazione marginale, non modifica d'effetto |
| P3 | MATCH_ON_TESTATI | SUPPORTATA | 4 insiemi in b3_pairs.csv |
| P3 | LVEF_REG | SUPPORTATA | a_regression.csv |
| P3 | LVEF_LTE45_AUROC | SUPPORTATA | a_classification.csv |
| P4 | CNN_AUROC_VAL_COMPOSITO | SUPPORTATA | val_per_target.csv |
| P4 | GAIN_VS_TABULARE | SUPPORTATA | val_per_target.csv |
| P4 | GPU_USATA | SUPPORTATA | ~4-6 s/epoca in history.json, compatibile con GPU |
| P4 | TARGET_MIGLIORI | SUPPORTATA | val_per_target.csv |
| P4 | EPOCHE 12, picco ep. 8 | SUPPORTATA | history.json |
| P4 | CNN > tabulare per QRS | SUPPORTATA | val_by_qrs.csv, IC non sovrapposti in tutte e 3 le fasce |
| P4 | CANDIDATO | SUPPORTATA | file presenti; predict non eseguito da me (vedi Problemi aperti) |
| P5 | FEATURE_PER_CLUSTER | SUPPORTATA | selection_summary.json |
| P5 | K=4 | SUPPORTATA | selection_summary.json |
| P5 | STABILITA_ARI 0.999 | DEBOLE | numero corretto ma quasi automatico per il sesso binario |
| P5 | CLUSTER_CREDIBILI: [] | SUPPORTATA | local_vs_global.csv |
| P5 | P_EMPIRICO | SUPPORTATA | permutation.csv |
| P5 | N_CONFIGURAZIONI 168 | DEBOLE | il registro ne contiene 125; 43 non registrate |
| P5 | χ² 281.7 | SUPPORTATA | chi2.csv |

Totale: 24 supportate, 3 deboli, 0 non supportate.

`QRS_MODIFICA`: i tre test NON concordano. (i) stratificazione: nessun beneficio (ΔMAE con IC che include 0 in tutte le fasce; ΔR² ≤ 0.009). (ii) interazione: statisticamente non nulla (Wald raggruppato p=3.6e-8 con QRS continuo) ma praticamente irrilevante (R² in CV 0.1401→0.1413, ΔMSE con IC che include 0). (iii) coppie: misura l'associazione marginale QRS-LVEF, non la modifica d'effetto (lo ammette il report), e si riduce a 2 ms [-2, 4] nella replica val con MATCH_ON C. Applicando la regola "≥2 test su 3 concordi" in modo letterale si potrebbero contare (ii) e (iii) come "concordi"; ma (iii) non testa l'ipotesi e (ii) non ha rilevanza pratica, quindi "inconclusiva" è un'etichetta difendibile, anzi generosa: l'evidenza di una modifica d'effetto clinicamente rilevante con 7 feature tabulari è sostanzialmente nulla.

**6. Senso clinico e limiti.**
- SHD ≠ scompenso: ricordato in tutti i report P2-P5. Composito: 1990/4626 positivi in val (43%); un AUROC del composito mescola condizioni con segnale ECG molto diverso (AS 0.84 tabulare vs pericardial effusion 0.68).
- Centro unico, solo pazienti con un ecocardiogramma (bias di selezione: popolazione già con sospetto clinico, prevalenza 43-52% lontana da quella di uno screening): nessuna fase lo discute esplicitamente come limite di generalizzazione; va detto in P8.
- Shift train/val (prevalenza 0.524 vs 0.430, setting diverso): calibrazione non valutata da nessuna fase.
- Equità: per il modello tabulare (P2, composito) l'AUROC varia per etnia 0.698 [0.611-0.785] (asian, n=134) - 0.779 (unknown) e per setting 0.676 [0.595-0.742] (procedural, n=177) - 0.765 (emergency); per età la stenosi aortica scende a 0.634 [0.542-0.734] a 65-74 anni. **Per il candidato principale (CNN) non esiste alcuna analisi per etnia, setting, sesso o età**: P4 stratifica solo per QRS. Non è possibile dire se la CNN degrada in un gruppo.

**7. Contratto dei candidati (verifica statica; esecuzione non possibile).**

| Candidato | Contratto | Indice/colonne | Note |
|---|---|---|---|
| `results/P4/candidates/cnn_multilabel` | carica `weights.pt`, `model_def.py`, `config.json` relativi a `__file__`; restituisce `score` = sigmoid dell'uscita 0 (`config.json`: indice 0 = `shd_moderate_or_greater_flag`, coerente con `candidate.json`), `in_scope=True` | `index=df.index` | richiede le colonne `split` e `row_in_split` in `df`: `final_eval.py` passa il `df` di test completo (r.50, r.61-63), quindi presenti; il test si apre solo con `ECHONEXT_ALLOW_TEST=1` |
| `results/P3/candidates/lvef_lte45_global` | identico all'esempio del PLAYBOOK | `index=df.index` | descrizione in candidate.json imprecisa (vedi sopra) |
| `results/P1/candidates/global_best` (comparatore) | identico all'esempio | `index=df.index` | è un modello del **composito**: in `final_eval.py` (r.70-74) viene usato come comparatore anche per il candidato LVEF<=45, dove il confronto non ha senso |

## Decisioni e motivazioni
- VERDETTO REVISE e non PASS: non ci sono problemi di leakage e due candidati sono ammissibili, ma ci sono lacune correggibili a basso costo che riguardano proprio il candidato principale (equità della CNN mai valutata; esecuzione di `predict` non verificata in modo indipendente) e la contabilità del validation (43 configurazioni non registrate, regola 4 di CLAUDE.md). Nessuna revisione richiede nuovi addestramenti né nuove scelte sul validation.
- Candidati ammessi (provvisoriamente, in attesa delle revisioni): `results/P4/candidates/cnn_multilabel` (una sola configurazione, epoca fissata a priori, AUROC val 0.823 [0.811-0.834]: la stima di validation è poco ottimistica) e `results/P3/candidates/lvef_lte45_global` (modello globale tabulare, scelto tra 5 modelli sul val con differenze ≤ 0.008: ottimismo trascurabile; accuratezza solo moderata).
- Nessun sottogruppo né cluster è ammesso: nessuno soddisfa i criteri, e la verifica lo conferma.
- Non ho modificato report, dati o registro di altre fasi; non ho toccato il test.

## Configurazioni provate
0 configurazioni nuove: questa revisione non ha addestrato modelli né valutato nulla sul validation (solo lettura di codice e di file aggregati). Nessuna registrazione `log-exp` necessaria.

## Artefatti
- `reports/P6_review.md` (questo file). Nessun file in `results/P6/` (non è stato possibile eseguire codice).

## Problemi aperti
- L'ambiente ha negato l'esecuzione di comandi: non ho potuto eseguire `python3 tools/state.py brief`/`handoff`, `python3 tools/validate_report.py reports/P6_review.md`, né rieseguire script o `candidate.predict` su un campione di validation (punto 7 della checklist verificato solo staticamente; punto 2 verificato contro i file, non rieseguendo). Va lanciato il validatore su questo report.
- Comparatore unico in `FROZEN.json` (`comparator_dir` = modello del composito di P1): per il candidato LVEF<=45 il campo `comparator_in_scope` di `final_eval.py` confronterà un modello SHD con etichette LVEF; P7 deve ignorarlo o segnalarlo.
- La testa LVEF<=45 della CNN (val 0.889 [0.876-0.902], gain +0.125 sul tabulare) è molto migliore del candidato tabulare LVEF ma non è un candidato (il candidato CNN espone solo l'uscita 0). Promuoverla ora sarebbe una scelta fatta guardando il validation tra 12 teste (lieve ottimismo, da dichiarare): decisione lasciata all'orchestratore/umano.
- Una sola esecuzione della CNN (un seed): variabilità tra addestramenti ignota.
- Calibrazione e bias di selezione (solo pazienti con eco, centro unico) non valutati da nessuna fase.

## Handoff
STATUS: PARZIALE
RISULTATI_CHIAVE:
- VERDETTO: REVISE (nessun leakage trovato in P2-P5; numeri chiave riprodotti dai file entro 0.001 di AUROC, 0 discrepanze numeriche)
- CANDIDATI_AMMESSI: [results/P4/candidates/cnn_multilabel, results/P3/candidates/lvef_lte45_global] (CNN composito val 0.823 [0.811-0.834]; LVEF<=45 tabulare 0.764 [0.748-0.782]); 0 sottogruppi, 0 cluster
- REVISIONI_RICHIESTE: [P4: stratificare il candidato CNN su validation per race_ethnicity, location_setting, sex, age_bin (AUROC con IC, n, positivi, CNN vs hist_gb tabulare sugli stessi pazienti con IC appaiato) usando val_pred_emb.npz senza riaddestrare, e documentare un'esecuzione di candidate.predict su 500 righe di val (colonne, indice uguale a df, max |score - predizione salvata|); P5: registrare con log-exp le 43 configurazioni su validation mancanti, correggere Configurazioni provate/N_CONFIGURAZIONI e dichiarare esplicitamente che in stenosi aortica cluster 2 logreg +0.021 [0.003,0.038] e lasso +0.018 [0.002,0.034] sono gli unici IC appaiati >0 su 40 confronti, non confermati dalla permutazione (p=0.77), e che ARI 0.999 è dominato dal sesso binario; P3: rifare il bootstrap delle coppie per paziente eliminando le coppie reciproche duplicate, correggere la descrizione del filtro 20% (in MATCH_ON B n_pairs=23637 = tutti i pazienti) e la descrizione di candidate.json di lvef_lte45_global (addestrato su tutte le righe con flag, non solo con LVEF)]
- AFFERMAZIONI: 24/3/0 (deboli: P3 coppie, P5 ARI, P5 N_CONFIGURAZIONI)
- QRS_MODIFICA: i tre test non concordano ((i) ΔR² ≤0.009; (ii) Wald p=3.6e-8 ma R² CV 0.1401→0.1413; (iii) associazione marginale, 2 [-2,4] ms in val con MATCH_ON C): "inconclusiva" difendibile, di fatto nessuna modifica clinicamente rilevante
- Multiplicità: 384 configurazioni su val registrate (427 reali) + 1950 regole e 78 configurazioni di clustering solo su train
- Equità: tabulare composito per etnia 0.698-0.779, per setting 0.676-0.765 (IC ampi); per la CNN nessuna analisi di equità disponibile
RACCOMANDAZIONE:
- Orchestratore: creare le 3 fasi di revisione R1-P4, R1-P5, R1-P3 (in quest'ordine di importanza) e riaprire P6; nessuna richiede riaddestramento.
- P7: per lvef_lte45_global il comparatore di FROZEN.json (modello del composito) non è pertinente; leggere il confronto solo per cnn_multilabel.
- P8: ribadire SHD ≠ scompenso, centro unico, solo pazienti con eco, calibrazione non valutata.
