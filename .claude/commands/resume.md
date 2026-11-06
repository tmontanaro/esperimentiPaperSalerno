---
description: Riprende la pipeline EchoNext dopo un'interruzione (limite di token, crash, nuova sessione)
argument-hint: [--until P3]
disable-model-invocation: true
---
Sei appena ripartito e non ricordi nulla: va bene, tutto è su disco.
1. `python3 tools/state.py brief`.
2. Se ci sono fasi `running` (interrotte): `python3 tools/state.py recover`. Il lavoro parziale resta in `results/<ID>/` e in `reports/<ID>_<nome>.progress.md`: la fase riparte da lì.
3. Se il brief segnala GATE DOVUTI, applica prima quei gate come descritto in `agentic/PLAYBOOK.md` e chiudili con `python3 tools/state.py gate-done <ID>`.
4. Continua poi esattamente come `/run-pipeline`, rispettando gli argomenti: $ARGUMENTS
