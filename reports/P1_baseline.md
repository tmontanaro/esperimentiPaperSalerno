# Report P1 · baseline
- Agente: baseline-modeler
- Data: 2026-11-06 12:10
- Stato: OK

## Obiettivo
Stabilire quanto segnale c'è nelle sole misure ECG tabulari (età, sesso, frequenze, intervalli PR, QRS, QTc) per predire il target composito `shd_moderate_or_greater_flag` (cardiopatia strutturale almeno moderata; NON coincide con scompenso cardiaco). Da qui dipendono le decisioni delle fasi successive.

## Input letti
- `python3 tools/state.py brief` e handoff P0 (contiene solo un «risultato simulato», senza INTEGRITA/GPU: nessun valore ereditato).
- `agentic/PLAYBOOK.md` sezione P1, soglie da `pipeline_state.json` (`tabular_signal_min_gain=0.02`, IC inferiore > 0.55).
- Dati in `/tmp/echonext_synth` (train 2127 righe / 1207 pazienti; validation 252 righe, 159 positivi).

## Cosa è stato fatto
1. `experiments/P1/baseline.py --debug` (smoke test `debug_n=2000`, output in `results/P1/debug/`), poi esecuzione completa.
2. Due insiemi di feature (età+sesso; 7 feature ECG) × 5 modelli (`lib.model_zoo`), addestrati sul train e valutati sul validation (AUROC con IC95% bootstrap, AUPRC).
3. Controllo di coerenza: AUROC out-of-fold con `GroupKFold` (5 fold, per `patient_key`) sul train, hist_gb. Indagine estesa a tutti i modelli: `experiments/P1/cv_check.py`.
4. Permutation importance (AUROC, 20 ripetizioni) su validation per il miglior modello con tutte le feature.
5. Candidato globale salvato. Test set non toccato.

## Risultati
Validation (n=252, 159 positivi, prevalenza 0.631). AUROC = capacità di ordinare i positivi sopra i negativi (0.5 = caso).

| Feature | Modello | AUROC val | IC95% | AUPRC |
|---|---|---|---|---|
| età+sesso | mlp_dense | 0.588 | 0.515–0.664 | 0.708 |
| età+sesso | hist_gb | 0.582 | 0.512–0.649 | 0.694 |
| età+sesso | logreg | 0.573 | 0.504–0.647 | 0.701 |
| età+sesso | lasso_logreg | 0.572 | 0.504–0.645 | 0.700 |
| età+sesso | random_forest | 0.571 | 0.516–0.640 | 0.711 |
| tutte ECG | lasso_logreg | 0.574 | 0.497–0.649 | 0.700 |
| tutte ECG | mlp_dense | 0.572 | 0.502–0.654 | 0.709 |
| tutte ECG | logreg | 0.570 | 0.497–0.644 | 0.700 |
| tutte ECG | random_forest | 0.564 | 0.493–0.641 | 0.678 |
| tutte ECG | hist_gb | 0.492 | 0.424–0.568 | 0.644 |

Riferimento AUPRC di un classificatore casuale = prevalenza = 0.631.

**TABULAR_SIGNAL** (definizione PLAYBOOK): miglior modello tutte-ECG (lasso_logreg) 0.574 − miglior età+sesso (mlp_dense) 0.588 = **−0.014** (soglia ≥ 0.02: NON soddisfatta); IC95% inferiore 0.497 (soglia > 0.55: NON soddisfatta). → **no**.

**Coerenza CV vs validation** (hist_gb, tutte ECG): CV su train 0.600 [0.579–0.624], validation 0.492 [0.424–0.568]: differenza **0.108 > 0.05**. Indagine (`results/P1/cv_vs_val.csv`), differenza CV−val per tutte le combinazioni:
- età+sesso: da −0.023 a +0.013 (coerenti);
- tutte ECG: logreg +0.042, lasso +0.035, random_forest +0.050, mlp +0.021, hist_gb +0.108.
Solo hist_gb con tutte le feature supera nettamente 0.05; il suo IC di validation non contiene il valore CV. Gli altri modelli sono entro o al limite della soglia. Con 252 righe di validation gli IC sono larghi (~±0.07). Non ho evidenze sufficienti per attribuire la discrepanza di hist_gb a una causa specifica; è un punto aperto. I modelli CV (0.59–0.61) restano comunque bassi.

**Permutation importance** (validation, lasso_logreg tutte ECG, calo medio di AUROC): età 0.034 (sd 0.030), atrial_rate 0.021 (sd 0.008), sex 0.008, pr_interval 0.001, ventricular_rate 0.000, qt_corrected 0.000, qrs_duration −0.0004 (sd 0.028). Salvata in `results/P1/permutation_importance.csv`. Nota: gli intervalli ±sd sono ampi rispetto agli effetti, tranne atrial_rate.

## Decisioni e motivazioni
- Miglior modello globale per il candidato: lasso_logreg con tutte le feature ECG (AUROC val più alto in quell'insieme). La scelta è fatta guardando il validation → stima ottimistica; la differenza tra i primi modelli è comunque entro il rumore.
- Si è seguita alla lettera la definizione di TABULAR_SIGNAL (miglior modello per ciascun insieme); non si sono cambiate soglie.
- Esito negativo riportato come tale: nelle misure ECG tabulari non si osserva segnale oltre età+sesso, e l'età+sesso stesso è debole (≈0.57–0.59).

## Configurazioni provate
10 configurazioni (2 insiemi × 5 modelli), tutte valutate sul validation e registrate con `log-exp` (E0002, `--uses-val --n-configs 10`). Il registro mostra P1 = 15 perché conteneva già 5 configurazioni di una esecuzione precedente (simulata); il numero reale di questa fase è 10. Gli output del controllo CV (train soltanto) e della permutation importance non aggiungono configurazioni.

## Artefatti
- `experiments/P1/baseline.py`, `experiments/P1/cv_check.py`
- `results/P1/val_results.csv`, `summary.json`, `permutation_importance.csv`, `cv_vs_val.csv`, `results/P1/debug/` (smoke test)
- Candidato: `results/P1/candidates/global_best/` (candidate.py, candidate.json, model.joblib)

## Problemi aperti
- Handoff P0 simulato: integrità e GPU non verificati da questa fase.
- Validation piccolo (252 righe): IC larghi, differenze < 0.05 tra modelli non distinguibili.
- Discrepanza CV−val di hist_gb (0.108) non spiegata.
- Contatore del registro (15) incoerente con le 10 configurazioni reali per via di una riga precedente.
- Il dataset sembra di dimensioni ridotte rispetto al dataset completo (train 2127 righe); non valutato perché.

## Handoff
STATUS: OK
RISULTATI_CHIAVE:
- TABULAR_SIGNAL: no (gain −0.014 < 0.02; IC inf. 0.497 < 0.55)
- AUROC_VAL_ETA_SESSO: 0.588 [0.515–0.664] (mlp_dense), n=252, 159 pos
- AUROC_VAL_TUTTE: 0.574 [0.497–0.649] (lasso_logreg); AUPRC 0.700 vs prevalenza 0.631
- MIGLIOR_MODELLO: lasso_logreg (tutte le feature ECG)
- CV train hist_gb 0.600 [0.579–0.624] vs val 0.492: diff 0.108 >0.05, indagata (solo hist_gb; altri ≤0.05)
- Permutation importance: età 0.034, atrial_rate 0.021, resto ≈0
- CANDIDATO_GLOBALE: results/P1/candidates/global_best
- Configurazioni: 10 (usano il validation)
RACCOMANDAZIONE:
- Applicare gate P1 «no»: tabular_signal=false, clusters_mode reduced, priorità a P4 (forme d'onda).
- Usare cautela con ogni sottogruppo: validation piccolo e baseline vicina al caso.
