# Log delle esecuzioni (append-only)

| Quando | Fase | Agente | Esito | Durata | Costo stimato | Note |
|---|---|---|---|---|---|---|
| 2026-11-06 14:11:31 | P0 | data-auditor | OK (OK) | 1.3 min | $0.27 | reports/P0_data-audit.md |
| 2026-11-06 14:11:39 | gate P0 | orchestrator | OK | 0.1 min | $0.06 | **Decisione:** il gate P0 è superato e non ho cambiato nulla. Si procede con P1.  **Azioni:** ho registrato due note con `state.py`: una `decision` e una `issue |
| 2026-11-06 14:13:07 | P1 | baseline-modeler | OK (OK) | 1.5 min | $0.25 | reports/P1_baseline.md |
| 2026-11-06 14:13:17 | gate P1 | orchestrator | OK | 0.2 min | $0.05 | **Decisione:** Gate P1 con TABULAR_SIGNAL = no. Le feature tabulari ECG non hanno segnale sul composito.  **Azioni eseguite:** - `set-config tabular_signal fals |
