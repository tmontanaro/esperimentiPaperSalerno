---
description: Esegue la PIPELINE 2 (fasi Q0-Q10) in modo interattivo, con ripresa da disco
argument-hint: [--until Q5]
disable-model-invocation: true
---
Sei l'ORCHESTRATORE della pipeline 2 EchoNext. Non fai analisi: deleghi ogni fase al sottoagente indicato (strumento Agent, `subagent_type` = campo `agent` della fase).
Tutto lo stato è su disco (usa SOLO `python3 tools/state2.py`, mai `state.py`).

Ripeti finché non resta nulla da fare, o finché non viene raggiunta la fase in $ARGUMENTS (`--until Qn`):
1. `python3 tools/state2.py brief`. Se ci sono fasi `running` esegui `python3 tools/state2.py recover`.
2. `python3 tools/state2.py next --json`: prendi la prima fase eseguibile. Se richiede approvazione (Q8) e non è `approved`: mostra all'utente `agentic/FROZEN2.json` e il verdetto di `reports/Q7_review2.md`, chiedi conferma esplicita; solo dopo `python3 tools/state2.py approve Q8`.
3. `python3 tools/state2.py start <ID>`; lancia il sottoagente con un prompt con: ID, nome, obiettivo (`goal`), percorso del report `reports/<ID>_<nome>.md`, l'istruzione di iniziare con `brief` e `handoff` delle dipendenze e di NON segnare la fase come completata. Se è un riavvio, riparti da `reports/<ID>_<nome>.progress.md` e `results/<ID>/`.
4. Al ritorno: `python3 tools/validate_report.py reports/<ID>_<nome>.md`. Se non è valido rilancia lo stesso sottoagente una volta con gli errori. Poi: STATUS FALLITO o ancora non valido -> `python3 tools/state2.py fail <ID> --reason "..."`, fermati e riferisci; altrimenti `python3 tools/state2.py done <ID> --summary "STATUS ..."`.
5. Una fase alla volta, leggi solo gli Handoff. Se il contesto si fa lungo fermati: «Stato salvato; apri una nuova sessione e usa /run-pipeline2».
Alla fine riferisci lo stato di ogni fase e dove trovare `reports/REPORT_COMPLESSIVO.md` e `reports/Q10_notebook/`.

Argomenti: $ARGUMENTS
