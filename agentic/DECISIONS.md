# Registro decisioni (append-only)

| Quando | Tipo | Fase | Testo |
|---|---|---|---|
| 2026-11-06 11:46:02 | config | - | pipeline inizializzata, data_dir=/tmp/echonext_synth |
| 2026-11-06 11:46:11 | decision | P0 | gate P0 simulato: nessuna modifica |
| 2026-11-06 11:46:12 | decision | P1 | gate P1 simulato: nessuna modifica |
| 2026-11-06 11:46:12 | decision | P2 | gate P2 simulato: nessuna modifica |
| 2026-11-06 11:46:12 | decision | P3 | gate P3 simulato: nessuna modifica |
| 2026-11-06 11:46:12 | decision | P4 | gate P4 simulato: nessuna modifica |
| 2026-11-06 11:46:12 | decision | P5 | gate P5 simulato: nessuna modifica |
| 2026-11-06 11:46:12 | decision | P6 | congelati 1 candidati per la valutazione sul test |
| 2026-11-06 11:46:12 | decision | P6 | gate P6 simulato: nessuna modifica |
| 2026-11-06 11:46:28 | decision | P7 | approvazione umana per l'esecuzione |
| 2026-11-06 11:46:35 | decision | P7 | test set usato una volta su 1 candidati |
| 2026-11-06 11:47:37 | config | - | pipeline inizializzata, data_dir=/tmp/echonext_synth |
| 2026-11-06 11:48:13 | issue | P0 | fase interrotta: riparte dai file parziali in results/ e dal file .progress.md |
| 2026-11-06 11:53:45 | issue | P0 | fase interrotta: riparte dai file parziali in results/ e dal file .progress.md |
| 2026-11-06 11:58:17 | issue | P0 | Handoff P0 senza campi INTEGRITA e GPU (contiene solo risultato simulato): non posso applicare la clausola GPU:no; assunzione P4 non registrata. Attempts=3 con due interruzioni. |
| 2026-11-06 11:58:17 | decision | P0 | Gate P0: STATUS=OK, nessun INTEGRITA: PROBLEMI riportato, GPU non indicata. Nessuna modifica a config/fasi; si procede con P1. |
| 2026-11-06 11:59:26 | decision | P1 | TABULAR_SIGNAL=no: migliore tutte-ECG lasso_logreg 0.574 [0.497-0.649] vs migliore eta+sesso mlp_dense 0.588; gain -0.014 <0.02, IC inf 0.497<0.55 |
| 2026-11-06 12:00:01 | config | - | tabular_signal = false |
| 2026-11-06 12:00:01 | config | - | clusters_mode = "reduced" |
| 2026-11-06 12:00:01 | decision | P1 | Gate P1: TABULAR_SIGNAL=no (gain -0.014 <0.02; IC inf 0.497 <0.55; lasso_logreg 0.574 [0.497-0.649] vs eta+sesso 0.588). Impostati tabular_signal=false, clusters_mode=reduced, priorità P4=10 (forme d'onda via principale). Cautela sui sottogruppi: validation piccolo (n=252), baseline vicina al caso. |
| 2026-11-06 12:00:01 | issue | P1 | hist_gb: CV train 0.600 vs val 0.492 (diff 0.108>0.05), indagata dalla fase, solo questo modello; nessun follow-up aggiunto. |
