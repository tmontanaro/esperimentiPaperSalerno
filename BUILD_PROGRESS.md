# Avanzamento della costruzione del kit (per ripartire se il lavoro si interrompe)

Aggiornato: 2026-11-06 09:00 (ora locale). Cartella del kit: `/home/claude/echonext-agentic/`. Backup numerati: `/home/claude/backups/`.

## Fatto e verificato
- `lib/echonext_lib.py`: caricamento dati con test set bloccato, metriche, modelli, `save_candidate`. Testato su dati sintetici.
- `tools/state.py`: stato su disco, recupero dopo crash, follow-up e revisioni, registro esperimenti, `brief`, `bump-review`, `gate-done`. Testato.
- `tools/run_pipeline.py` (driver): nuovi tentativi, limite d'uso (uscita 75), pausa per approvazione (uscita 10), gate persistenti,
  **snapshot di stato prima/dopo ogni fase** in `agentic/backups/` e **checkpoint git** se il progetto è un repository. Testato end-to-end.
- `tools/validate_report.py`, `tools/final_eval.py` (test valutabile una sola volta), `tools/make_synth.py`, `tools/fake_claude.py` (finto claude), `tools/reset.py` (azzera lo stato generato).
- `CLAUDE.md`, `.claude/settings.json`, `agentic/PLAYBOOK.md`, `agentic/REPORT_TEMPLATE.md`, `requirements.txt`, `.gitignore`.
- 10 agenti in `.claude/agents/` e 3 comandi in `.claude/commands/` (`run-pipeline`, `resume`, `status`).
- `legacy_notebooks/`: i 5 notebook già consegnati.
- Snippet shell di nuovo tentativo dopo il limite d'uso: provato con uno stub.
- Consegnata all'utente una versione intermedia dello zip (senza GUIDA.md).

## Da fare (in ordine)
1. `GUIDA.md`: guida passo passo (installazione, avvertenze su account e dati PhysioNet, prova su dati sintetici, esecuzione, ripresa, personalizzazione).
2. Ricontrollo finale: compilazione, simulazione con `tools/fake_claude.py`, `python3 tools/reset.py --yes`.
3. Pacchetto `echonext-agentic.zip` definitivo e consegna.

## Cose NON verificate (dirlo nella guida)
- Nessun agente è stato eseguito con un Claude reale: qui manca l'autenticazione. Provati cablaggio e strumenti con il finto `claude`.
- I campi dell'output JSON di `claude -p` sono letti con `.get`: confermati dalla documentazione, non provati dal vivo.
- Le soglie del PLAYBOOK sono valori iniziali da discutere.
- La politica PhysioNet (24/09/2025) cita MIMIC e derivati, non EchoNext: l'utente deve leggere il DUA di EchoNext.
- Non sono stati verificati i termini di ritenzione dei dati di Anthropic per l'account dell'utente.

## Note per chi riprende
- Se un file risulta modificato su disco, è stato toccato di proposito: non ripristinarlo.
- Dopo ogni prova: `python3 tools/reset.py --yes` per eliminare stato e risultati generati.

## Aggiornamento 09:20: scoperte dal collaudo con la CLI vera (la CLI funziona in questo ambiente)
- Confermato: `claude -p ... --agent <nome> --output-format json --permission-mode dontAsk --max-turns --max-budget-usd` è accettato; un agente inesistente dà «not found. Available agents»; il JSON contiene `total_cost_usd` e `session_id`.
- SCOPERTA: in modalità `-p`, se la cartella non è stata approvata interattivamente, la CLI **ignora le regole `allow` di `.claude/settings.json`**. Il driver ora lo rileva (uscita 3, `UNTRUSTED_MSG`).
  DA FARE nella GUIDA.md: aggiungere in §2/§5 il passo «apri `claude` una volta nella cartella e accetta la fiducia», codice di uscita 3, e correggere §9 (ora un collaudo con Claude reale è stato fatto, vedi sotto).
- IN CORSO: collaudo reale di P0 e P1 su dati sintetici con HOME finto `$S/fakehome` (fiducia impostata via `.claude.json`), tetto 2 USD per chiamata.
  Se interrotto: `cd /home/claude/echonext-agentic; python3 tools/state.py brief` mostra dove era; poi rilanciare
  `HOME=$S/fakehome python3 tools/run_pipeline.py --claude-bin $S/cc_install/node_modules/.bin/claude --until P1 --max-budget-usd 2 --model sonnet`
  (S = /tmp/claude-0/-home-claude/a5e12470-0ce0-5243-890a-079d975cad2e/scratchpad). Alla fine: `python3 tools/reset.py --yes` e rimuovere `.claude/settings.local.json`.
- Ancora da fare dopo il collaudo: correggere i prompt secondo ciò che emerge, aggiornare GUIDA.md, zip definitivo, consegna.

## Aggiornamento 10:10: collaudo reale concluso
- P0 e P1 eseguiti con la CLI vera su dati sintetici: report validi, gate OK (P1: TABULAR_SIGNAL=no, corretto per dati sintetici).
- GUIDA.md aggiornata (passo di fiducia cartella, codice di uscita 3, §9). Resta: pulizia, zip finale, consegna.
