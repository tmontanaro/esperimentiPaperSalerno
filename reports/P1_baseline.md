# Report P1 · baseline
- Agente: baseline-modeler
- Data: 2026-11-06 14:30
- Stato: OK

## Obiettivo
Stabilire quanto segnale c'è nelle sole misure ECG tabulari (7 feature) per predire il target composito `shd_moderate_or_greater_flag` (cardiopatia strutturale di grado almeno moderato; NON coincide con scompenso cardiaco), rispetto al solo uso di età e sesso. Da qui dipendono le scelte delle fasi successive.

## Input letti
- Handoff P0: integrità OK, N_TRAIN 72475 ECG (26218 pazienti), N_VAL 4626, prevalenza target train 0.524 vs validation 0.430, mancanti pr_interval 10.5% e atrial_rate 0.6% (gestiti dall'imputazione mediana con indicatore nei modelli di `lib.model_zoo`).
- `agentic/PLAYBOOK.md` (sezione P1), `lib/echonext_lib.py`.

## Cosa è stato fatto
Script: `experiments/P1/baseline.py`.
1. Smoke test con `debug_n=2000` (output in `results/P1/debug/`): tutto funzionante.
2. Run completo: due insiemi di feature (età+sesso; 7 feature ECG) x 5 modelli (logreg, lasso_logreg, random_forest, hist_gb, mlp_dense), addestrati sul train e valutati su validation con AUROC (area sotto la curva ROC: 0.5 = caso, 1 = perfetto), IC95% bootstrap e AUPRC (precisione-richiamo).
3. Controllo di coerenza: AUROC out-of-fold con GroupKFold (5 fold) per `patient_key` sul train, modello hist_gb, tutte le feature.
4. Permutation importance (calo di AUROC mescolando una feature) su validation per il miglior modello (hist_gb, 10 ripetizioni).
5. Salvato il candidato globale (hist_gb addestrato sul train, 7 feature).

## Risultati
Validation (n=4626, positivi 1990, prevalenza 0.430):

| Feature | Modello | AUROC | IC95% | AUPRC |
|---|---|---|---|---|
| età+sesso | mlp_dense | 0.6587 | 0.6415-0.6750 | 0.5825 |
| età+sesso | lasso_logreg | 0.6550 | 0.6388-0.6695 | 0.5692 |
| età+sesso | logreg | 0.6548 | 0.6387-0.6692 | 0.5686 |
| età+sesso | hist_gb | 0.6543 | 0.6371-0.6712 | 0.5772 |
| età+sesso | random_forest | 0.6503 | 0.6339-0.6666 | 0.5744 |
| 7 ECG | hist_gb | 0.7464 | 0.7331-0.7595 | 0.6988 |
| 7 ECG | mlp_dense | 0.7446 | 0.7305-0.7589 | 0.6940 |
| 7 ECG | random_forest | 0.7410 | 0.7270-0.7540 | 0.6915 |
| 7 ECG | lasso_logreg | 0.7380 | 0.7246-0.7533 | 0.6824 |
| 7 ECG | logreg | 0.7380 | 0.7246-0.7533 | 0.6824 |

TABULAR_SIGNAL (definizione PLAYBOOK): AUROC_val migliore con tutte le feature 0.7464 (hist_gb) meno AUROC_val migliore con età+sesso 0.6587 (mlp_dense) = guadagno 0.0877, soglia 0.02: soddisfatta. Limite inferiore IC95% 0.7331 > 0.55: soddisfatta. Quindi `TABULAR_SIGNAL: yes`. Nota: il miglior modello è scelto guardando il validation, quindi i valori sono leggermente ottimistici; le differenze tra i 5 modelli a parità di feature sono piccole (IC sovrapposti), mentre quella tra i due insiemi di feature no.

Controllo di coerenza (train, CV raggruppata, hist_gb, n=72475, positivi 37958): AUROC CV 0.7278 (IC95% 0.7241-0.7314) contro 0.7464 in validation. Differenza CV - validation = -0.019: non c'è alcun gonfiamento da CV (soglia di allarme 0.05); la validation è anzi leggermente più alta. Non è stata indagata la causa della differenza: non c'è evidenza per attribuirla a qualcosa (si nota solo che la prevalenza differisce, 0.524 train vs 0.430 validation, e che la validation ha un ECG per paziente).

Permutation importance (validation, hist_gb, calo medio di AUROC +- dev. std su 10 ripetizioni):

| Feature | Calo AUROC |
|---|---|
| qt_corrected | 0.0628 +- 0.0041 |
| age_at_ecg | 0.0389 +- 0.0040 |
| qrs_duration | 0.0309 +- 0.0031 |
| pr_interval | 0.0251 +- 0.0026 |
| sex | 0.0094 +- 0.0015 |
| ventricular_rate | 0.0052 +- 0.0010 |
| atrial_rate | 0.0033 +- 0.0012 |

## Decisioni e motivazioni
- Miglior modello globale: hist_gb su 7 feature (AUROC_val più alto), usato come candidato e come comparatore per P7. Alternative scartate: gli altri 4 modelli, statisticamente indistinguibili in validation.
- Il segnale è moderato (AUROC 0.75, sotto la soglia "buona" 0.80 del PLAYBOOK, che comunque si applica al test).

## Configurazioni provate
10 configurazioni (2 insiemi di feature x 5 modelli), tutte valutate sul validation; registrate con `log-exp` (E0001, n-configs 10, uses-val). Lo smoke test (debug_n=2000) non è stato registrato perché è un sottoinsieme dello stesso run senza scelte basate su di esso. CV e permutation importance non aggiungono configurazioni di selezione.

## Artefatti
- `experiments/P1/baseline.py`
- `results/P1/val_models.csv`, `results/P1/perm_importance.csv`, `results/P1/summary.json`, `results/P1/debug/`
- `results/P1/candidates/global_best/` (candidate.json, candidate.py, model.joblib)

## Problemi aperti
- Il test set non è stato toccato.
- La permutation importance indica la dipendenza del modello dalle feature, non un rapporto causale.
- Il candidato è stato addestrato solo sul train (non sul train+validation).
- Il miglior modello è stato scelto sul validation (ottimismo da selezione, piccolo dato che 5 modelli sono quasi pari).

## Handoff
STATUS: OK
RISULTATI_CHIAVE:
- TABULAR_SIGNAL: yes (guadagno 0.0877 >= 0.02; IC inferiore 0.7331 > 0.55)
- AUROC_VAL_ETA_SESSO: 0.6587 [0.6415-0.6750] (mlp_dense)
- AUROC_VAL_TUTTE: 0.7464 [0.7331-0.7595], AUPRC 0.6988 (hist_gb)
- MIGLIOR_MODELLO: hist_gb (7 feature ECG)
- CV raggruppata train hist_gb: AUROC 0.7278 [0.7241-0.7314]; CV - val = -0.019 (nessuna incoerenza)
- Importanza: qt_corrected 0.063, age 0.039, qrs_duration 0.031, pr_interval 0.025
- Configurazioni provate in P1: 10 (validation)
- CANDIDATO_GLOBALE: results/P1/candidates/global_best
RACCOMANDAZIONE:
- Gate P1: segnale tabulare presente (tabular_signal true); P2 può eseguire l'analisi completa incluse le regole di sottogruppo.
- Riferimento globale per i sottogruppi: AUROC_val 0.7464 [0.7331-0.7595]; ricordare il possibile shift di prevalenza train/validation.
