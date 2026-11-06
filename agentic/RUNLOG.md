# Log delle esecuzioni (append-only)

| Quando | Fase | Agente | Esito | Durata | Costo stimato | Note |
|---|---|---|---|---|---|---|
| 2026-11-06 12:07:47 | P0 | data-auditor | OK (PARZIALE) | 1.8 min | $0.31 | reports/P0_data-audit.md |
| 2026-11-06 12:07:58 | gate P0 | orchestrator | OK | 0.2 min | $0.03 | **Decisione:** il Gate P0 passa con riserva. Si procede con le fasi tabulari P1, P2, P3 e P5. P4 resta bloccata.  **Azioni eseguite:** - Ho registrato una `note |
| 2026-11-06 12:09:24 | P1 | baseline-modeler | OK (OK) | 1.4 min | $0.22 | reports/P1_baseline.md |
| 2026-11-06 12:09:33 | gate P1 | orchestrator | OK | 0.1 min | $0.03 | **Decisione:** P1 ha trovato segnale nelle feature tabulari (`TABULAR_SIGNAL: yes`), quindi la pipeline prosegue in modalità normale. **Azioni:** ho eseguito `s |
