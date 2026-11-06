# Report P1 · baseline
- Agente: baseline-modeler
- Data: 2026-11-06 12:20
- Stato: OK

## Obiettivo
Stabilire quanto segnale c'è nelle sole misure ECG tabulari (7 feature) per il target composito `shd_moderate_or_greater_flag`, rispetto a un riferimento con solo età e sesso. Da qui dipendono le decisioni delle fasi successive. Nota: SHD (cardiopatia strutturale) NON coincide con scompenso cardiaco.

## Input letti
- Handoff P0: N_TRAIN 72475, N_VAL 4626, prevalenza target train 0.5237 vs val 0.4302 (shift), forme d'onda mancanti, pr_interval mancante 10.5% (gestito con imputazione mediana + indicatore nei modelli di `lib.model_zoo`).
- `agentic/PLAYBOOK.md` (sezione P1), `legacy_notebooks/00_Setup_EDA_Baseline.ipynb` come ricetta (non riletto: usate le funzioni di `lib`).

## Cosa è stato fatto
Script: `experiments/P1/baseline.py`.
1. Smoke test con `debug_n=2000` (`python3 experiments/P1/baseline.py debug`, output in `results/P1/debug/`), poi esecuzione completa.
2. Due insiemi di feature (età+sesso; 7 feature ECG) × 5 modelli (logreg, lasso_logreg, random_forest, hist_gb, mlp_dense), addestrati sul train, valutati su validation (AUROC con IC95% bootstrap, AUPRC).
3. Controllo di coerenza: `oof_proba` con GroupKFold(5) per `patient_key` sul train, hist_gb, 7 feature.
4. Permutation importance (10 ripetizioni, scoring AUROC) su validation per il miglior modello.
5. TABULAR_SIGNAL con la definizione del PLAYBOOK.
6. Salvato il candidato globale.

## Risultati
Validation (n=4626, positivi=1990, prevalenza 0.430):

| Feature | Modello | AUROC | IC95% | AUPRC |
|---|---|---|---|---|
| età+sesso | mlp_dense | 0.659 | 0.642–0.675 | 0.582 |
| età+sesso | lasso_logreg | 0.655 | 0.639–0.670 | 0.569 |
| età+sesso | logreg | 0.655 | 0.639–0.669 | 0.569 |
| età+sesso | hist_gb | 0.654 | 0.637–0.671 | 0.577 |
| età+sesso | random_forest | 0.650 | 0.634–0.667 | 0.574 |
| 7 ECG | hist_gb | 0.746 | 0.733–0.760 | 0.699 |
| 7 ECG | mlp_dense | 0.745 | 0.731–0.759 | 0.694 |
| 7 ECG | random_forest | 0.741 | 0.727–0.754 | 0.692 |
| 7 ECG | lasso_logreg | 0.738 | 0.725–0.753 | 0.682 |
| 7 ECG | logreg | 0.738 | 0.725–0.753 | 0.682 |

Coerenza CV/validation (hist_gb, 7 feature): AUROC CV raggruppata sul train = 0.728; validation = 0.746. Differenza CV − val = −0.018 (non > 0.05): nessun segnale di sovra-adattamento o leakage. La CV è anche più bassa della validation; non ho indagato il motivo (non c'è evidenza per spiegarlo; si noti lo shift di prevalenza train/val segnalato da P0).

Permutation importance (hist_gb, validation, calo di AUROC): qt_corrected 0.063 (sd 0.004), age_at_ecg 0.039, qrs_duration 0.031, pr_interval 0.025, sex 0.009, ventricular_rate 0.005, atrial_rate 0.003.

TABULAR_SIGNAL: guadagno = 0.7464 − 0.6587 = 0.0877 ≥ 0.02 e limite inferiore IC95% 0.733 > 0.55 → **yes**.
Valore assoluto: AUROC 0.746, ossia "moderata" (0.70–0.80) in validation; non è una stima finale (test chiuso).

## Decisioni e motivazioni
- Miglior modello per ciascun insieme scelto per AUROC su validation, come richiesto dal PLAYBOOK: la stima è quindi leggermente ottimistica (selezione tra 5 modelli; le differenze tra i primi tre modelli ECG sono ≤ 0.005, dentro gli IC).
- Candidato globale = hist_gb addestrato sul train con le 7 feature ECG.
- Alternative: nessuna ottimizzazione di iperparametri (fuori dalla fase).

## Configurazioni provate
10 configurazioni (5 modelli × 2 insiemi), tutte con uso del validation; registrate con `log-exp --phase P1 --n-configs 10`. La CV sul train (1 modello) non usa il validation.

## Artefatti
- `experiments/P1/baseline.py`
- `results/P1/models_eta_sesso.csv`, `results/P1/models_tutte.csv`, `results/P1/permutation_importance.csv`, `results/P1/summary.json`
- `results/P1/debug/` (smoke test)
- Candidato: `results/P1/candidates/global_best/`

## Problemi aperti
- Shift di prevalenza train (0.524) vs validation (0.430) non investigato qui.
- Il validation è stato usato per scegliere il miglior modello: AUROC lievemente ottimistico.
- Forme d'onda mancanti (P0): non affrontate in questa fase.

## Handoff
STATUS: OK
RISULTATI_CHIAVE:
- TABULAR_SIGNAL: yes (guadagno 0.0877 ≥ 0.02; IC inf 0.733 > 0.55)
- AUROC_VAL_ETA_SESSO: 0.659 [0.642–0.675] (mlp_dense)
- AUROC_VAL_TUTTE: 0.746 [0.733–0.760], AUPRC 0.699 (hist_gb)
- MIGLIOR_MODELLO: hist_gb (7 feature ECG)
- CV train (GroupKFold, hist_gb) 0.728 vs val 0.746: coerenti (diff −0.018)
- Feature più importanti: qt_corrected, age_at_ecg, qrs_duration, pr_interval
- CANDIDATO_GLOBALE: results/P1/candidates/global_best
- Configurazioni provate con validation: 10
RACCOMANDAZIONE:
- Procedere con P2/P3/P5 in modalità normale (`tabular_signal true`).
- Confrontare i sottogruppi con AUROC globale val 0.746 [0.733–0.760].
