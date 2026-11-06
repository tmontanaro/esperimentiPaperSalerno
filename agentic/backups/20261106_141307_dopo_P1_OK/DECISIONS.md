# Registro decisioni (append-only)

| Quando | Tipo | Fase | Testo |
|---|---|---|---|
| 2026-11-06 14:10:04 | config | - | pipeline inizializzata, data_dir=/tmp/echonext_synth |
| 2026-11-06 14:11:36 | decision | P0 | Gate P0: STATUS OK, INTEGRITA OK, GPU sì (train 2127, val 252). Nessuna modifica a config/fasi; si procede con P1. |
| 2026-11-06 14:11:36 | issue | P0 | Dati: 380 righe no_split non classificate; data_dir=/tmp/echonext_synth, possibili dati sintetici (da verificare). Condizioni con <10 positivi in val (pulmonary_regurgitation 4, pericardial_effusion 4; lvef_lte_45 21/198) non valutabili: trattare con cautela in P2/P3. |
| 2026-11-06 14:12:37 | decision | P1 | TABULAR_SIGNAL=no: AUROC val tutte 0.574 (IC 0.497-0.649) vs eta+sesso 0.588; gain -0.013, IC inf <0.55. Candidato global_best = lasso_logreg (scelto su val, ottimistico). |
