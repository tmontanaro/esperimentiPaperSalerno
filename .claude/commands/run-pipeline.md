---
description: Esegue la pipeline multi-agente EchoNext in modo interattivo, con ripresa da disco
argument-hint: [--until P3]
disable-model-invocation: true
---
Sei l'ORCHESTRATORE della pipeline EchoNext. Non fai analisi: deleghi ogni fase al sottoagente indicato (strumento Agent, `subagent_type` = campo `agent` della fase).
Tutto lo stato è su disco; se il contesto si esaurisce potrai ripartire con `/resume`.

Ripeti questo ciclo finché non resta nulla da fare, o finché non viene raggiunta la fase indicata in $ARGUMENTS (`--until Pn`):
1. `python3 tools/state.py brief`. Se ci sono fasi `running` (di una sessione precedente) esegui `python3 tools/state.py recover`.
2. Gate dovuti: applica la sezione «Gate <ID>» di `agentic/PLAYBOOK.md` solo con `tools/state.py`, annota con `state.py note`, poi `python3 tools/state.py gate-done <ID>`.
3. `python3 tools/state.py next --json`: prendi la prima fase eseguibile.
   - Se richiede approvazione (P7) e non è `approved`: mostra all'utente `agentic/FROZEN.json` e il verdetto di `reports/P6_review.md`, chiedi conferma esplicita; solo dopo `python3 tools/state.py approve P7`.
4. `python3 tools/state.py start <ID>`; lancia il sottoagente con un prompt che contiene: ID, nome, obiettivo (campo `goal`), percorso del report `reports/<ID>_<nome>.md`, l'istruzione di iniziare con `brief` e con `handoff` delle dipendenze,
   e di NON segnare la fase come completata. Se è un riavvio, indica di ripartire da `reports/<ID>_<nome>.progress.md` e `results/<ID>/`.
5. Al ritorno: `python3 tools/validate_report.py reports/<ID>_<nome>.md`. Se non è valido, rilancia lo stesso sottoagente una volta con gli errori. Poi:
   - STATUS FALLITO o validazione ancora fallita: `python3 tools/state.py fail <ID> --reason "..."`, fermati e riferisci all'utente.
   - altrimenti `python3 tools/state.py done <ID> --summary "STATUS ..."` e torna al punto 2 (il gate di questa fase).
6. Leggi solo gli Handoff (`state.py handoff <ID>`), non i report interi. Una fase alla volta. Se noti che il contesto si fa lungo, fermati: «Stato salvato; apri una nuova sessione e usa /resume».
Alla fine riferisci: stato di ogni fase, decisioni dei gate, e dove trovare `reports/P8_synthesis.md`.

Argomenti: $ARGUMENTS
