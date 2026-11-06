# Guida: dal notebook eseguito a mano alla pipeline multi-agente (Claude Code)

Questa guida porta il workflow EchoNext (i 5 notebook) in una pipeline in cui **agenti specializzati lavorano uno dopo l'altro**, un **orchestratore** decide cosa fare in base ai risultati,
un **revisore indipendente** cerca gli errori e alla fine esce un **report spiegato**. Ogni passaggio viene scritto su disco, quindi se il processo si ferma (token finiti, crash, chiusura del computer) **riparti da dove eri, non dall'inizio**.

## Indice
0. Prima di iniziare: tre verifiche (leggile davvero)
1. Installare Claude Code
2. Preparare il progetto
3. Come funziona (in due minuti)
4. Prova a secco: collaudare la pipeline prima dei dati veri
5. Eseguire sui dati reali
6. Se si blocca: ripartire
7. Leggere i risultati e approvare il test
8. Personalizzare
9. Cosa non è stato verificato

---

## 0. Prima di iniziare: tre verifiche

**A. Serve un account che includa Claude Code.** Secondo la documentazione ufficiale serve un account Pro, Max, Team, Enterprise o Console; **il piano gratuito di claude.ai non include Claude Code**.
In alternativa si può usare un provider API di terze parti (Amazon Bedrock, Google Cloud, Microsoft Foundry). Chiedi al tuo ateneo se ha un accordo.

**B. Verifica che i dati EchoNext possano passare da un servizio cloud. Questo è il punto più importante.**
Quando un agente esegue codice sui tuoi dati, ciò che il codice stampa (e ciò che i file leggibili contengono) viene inviato ad Anthropic per essere elaborato.
- PhysioNet, nel suo avviso del 24 settembre 2025 «Use of MIMIC Data with Large Language Models and Online Services», scrive che il Data Use Agreement *vieta di condividere l'accesso ai dati con terze parti, inclusi l'invio tramite API o l'uso su piattaforme online*.
  Per i servizi cloud chiede di verificare zero data retention, nessun uso dei dati per il training e nessuna revisione umana, avverte che i requisiti dichiarati dai servizi possono non bastare e **raccomanda fortemente modelli LLM in locale**.
- Quell'avviso cita MIMIC e derivati, **non EchoNext**. EchoNext ha il suo accordo d'uso (la licenza è nella pagina del dataset): **leggilo** e chiedi conferma scritta al responsabile del progetto o dell'ufficio dati.
- Non ho verificato le condizioni di conservazione dei dati di Anthropic per il tuo tipo di account: controllale tu.
- Il kit **riduce** l'esposizione (gli agenti sono istruiti a stampare solo aggregati, non possono aprire i file dati con lo strumento di lettura, `lib.show()` rifiuta tabelle a livello di paziente), ma non la azzera: codice scritto da un agente potrebbe stampare altro.
- Se non hai la conferma, **fermati al passo 4**: la prova completa si fa su dati sintetici e non ha alcun rischio.

**C. Risorse.** Le forme d'onda pesano circa 12 GB (spazio su disco libero doppio se estrai da uno zip). Per la fase P4 (CNN) serve una GPU NVIDIA: senza, la fase fa solo una prova e dice come lanciarla altrove.
Le fasi tabulari girano su un portatile.

---

## 1. Installare Claude Code

Comandi ufficiali (verificati sulla documentazione, novembre 2026):

| Sistema | Comando |
|---|---|
| Linux, macOS, WSL | `curl -fsSL https://claude.ai/install.sh \| bash` |
| Windows PowerShell | `irm https://claude.ai/install.ps1 \| iex` |
| Homebrew | `brew install --cask claude-code` |
| npm (serve Node.js 22+) | `npm install -g @anthropic-ai/claude-code` (mai con `sudo`) |

1. Apri un **nuovo** terminale e verifica: `claude --version` (stampa un numero di versione). Diagnostica: `claude doctor`.
2. Primo avvio e accesso: nella cartella del progetto digita `claude` e segui le istruzioni nel browser. Se compare «command not found», la cartella di installazione non è nel PATH: vedi la pagina *Troubleshoot installation* della documentazione.
3. Requisiti: Ubuntu 20.04+, Debian 10+, macOS 13+, Windows 10 (1809+), almeno 4 GB di RAM. Su Linux gli aggiornamenti sono automatici (installazione nativa).
4. Documentazione: https://code.claude.com/docs

---

## 2. Preparare il progetto

```bash
unzip echonext-agentic.zip          # oppure: python3 -m zipfile -e echonext-agentic.zip .
cd echonext-agentic
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # torch: con GPU NVIDIA usa la build CUDA indicata su pytorch.org
git init && git add -A && git commit -m "kit iniziale"    # consigliato: permette di tornare indietro e il driver salva un commit dopo ogni fase
```

I dati **restano fuori** dalla cartella del progetto (es. `/home/idalab/Desktop/Teo/physionet.org/files/echonext/1.1.1/`): il kit li trova anche in sottocartelle o dentro uno zip.
Se il caricamento non trova il CSV, l'errore elenca cosa c'è nella cartella.

---

## 3. Come funziona (in due minuti)

```
   ┌────────────┐  gate   ┌────────────┐  gate   ┌────────────┐         ┌──────────┐  ok umano  ┌────────┐   ┌────────────┐
   │ P0 audit   │ ──────▶ │ P1 baseline│ ──────▶ │ P2 / P3 /  │ ──────▶ │ P6 review│ ─────────▶ │P7 test │──▶│ P8 sintesi │
   │ dati       │         │ tabulare   │         │ P4 / P5    │         │ scettico │            │ (1 volta)│  │ report     │
   └────────────┘         └────────────┘         └────────────┘         └──────────┘            └────────┘   └────────────┘
        ▲  ogni fase = un agente, una sessione nuova, un report su disco  ▲  il gate (orchestratore) legge l'Handoff e decide: salta, aggiunge, riordina
```

- **Memoria = file.** Nessun agente si "ricorda" nulla: ognuno comincia con `python3 tools/state.py brief`, che stampa stato, decisioni e gli *Handoff* delle fasi precedenti.
- **Un report per fase** (`reports/Pn_nome.md`), sempre con le stesse sezioni (Obiettivo, Input letti, Cosa è stato fatto, Risultati, Decisioni, Configurazioni provate, Artefatti, Problemi aperti, **Handoff**). Un validatore rifiuta i report incompleti.
- **Le decisioni adattive** sono nel `PLAYBOOK` (sezioni «Gate»): se le sole misure tabulari non portano segnale, le forme d'onda diventano prioritarie e il clustering si riduce; se il QRS risulta un modificatore d'effetto, parte un'analisi stratificata; ecc.
- **Protezioni imposte dal codice**: il test set non si apre fuori da P7 e solo una volta; le colonne derivate dall'eco sono rifiutate come input; i candidati si valutano con un contratto fisso.

Mappa dei file:

| Percorso | Contenuto |
|---|---|
| `CLAUDE.md` | regole di progetto lette da tutti gli agenti |
| `agentic/PLAYBOOK.md` | fasi, soglie, definizioni di «credibile», regole dei gate (**qui decidi tu**) |
| `agentic/pipeline_state.json` | stato (si modifica solo con `tools/state.py`) |
| `agentic/DECISIONS.md`, `agentic/RUNLOG.md` | registro delle decisioni e log di ogni chiamata (esito, durata, costo stimato) |
| `agentic/backups/` | copia dello stato prima e dopo ogni fase |
| `reports/` | un report per fase + `*.progress.md` (appunti per riprendere a metà fase) |
| `results/`, `results/registry.csv` | output, candidati, elenco di ogni configurazione provata |
| `.claude/agents/`, `.claude/commands/` | gli agenti e i comandi `/run-pipeline`, `/resume`, `/status` |
| `legacy_notebooks/` | i notebook di partenza (ricette per gli agenti) |
| `BUILD_PROGRESS.md` | note su come è stato costruito il kit |

---

## 4. Prova a secco: collaudare la pipeline prima dei dati veri

**4A. Simulazione senza token (2 minuti).** Un finto `claude` simula gli agenti e verifica che stato, ripresa, pause e controlli funzionino sul tuo computer.

```bash
python3 tools/make_synth.py /tmp/echonext_synth 3000        # dati sintetici con lo stesso schema (nessun valore scientifico)
python3 tools/state.py init --data-dir /tmp/echonext_synth
python3 tools/run_pipeline.py --claude-bin tools/fake_claude.py ; echo "codice di uscita: $?"
```
Il driver si ferma con codice 10 prima di P7 (serve la tua approvazione). Procedi:
```bash
python3 tools/state.py approve P7
python3 tools/run_pipeline.py --claude-bin tools/fake_claude.py
```
Per provare i guasti: `FAKE_FAIL_ONCE=P2 FAKE_LIMIT_ON=P3 FAKE_BAD_REPORT=P4 python3 tools/run_pipeline.py --claude-bin tools/fake_claude.py`
(errore, limite d'uso e report incompleto: il driver riprova, esce con 75 al limite e alla riesecuzione riparte da P3).
Poi: `python3 tools/reset.py --yes`.

**4B. Prova con Claude vero sui dati sintetici (costa pochi token).** Verifica che i permessi e i prompt funzionino davvero e non c'è alcun rischio sui dati.
```bash
python3 tools/state.py init --data-dir /tmp/echonext_synth --force
python3 tools/run_pipeline.py --until P1 --max-budget-usd 3
```
Leggi `reports/P0_data-audit.md`, `reports/P1_baseline.md`, `agentic/RUNLOG.md`. Se un agente è stato bloccato dai permessi (in modalità headless ciò che non è consentito viene negato senza chiedere), lo vedrai nei report in «Problemi aperti»: aggiusta `.claude/settings.json`.
Poi `python3 tools/reset.py --yes`.

---

## 5. Eseguire sui dati reali (solo dopo la verifica B del passo 0)

```bash
python3 tools/reset.py --yes
python3 tools/state.py init --data-dir /home/idalab/Desktop/Teo/physionet.org/files/echonext/1.1.1
```
`init` scrive anche una regola in `.claude/settings.local.json` che impedisce agli agenti di aprire i file della cartella dati con lo strumento di lettura.

**Passo obbligatorio prima di tutto: approva la cartella.** Apri `claude` una volta nella cartella del progetto (`cd echonext-agentic && claude`) e accetta la richiesta «Do you trust this folder?». Senza questo passo, in modalità non interattiva (`-p`, usata dal driver) Claude Code **ignora le regole di permesso** del progetto e gli agenti non possono lavorare: il driver se ne accorge e termina con codice 3. Poi esci (`/exit`).

### Modo A: interattivo (consigliato la prima volta)
```bash
claude
> /run-pipeline            # oppure: /run-pipeline --until P3
```
La sessione principale fa da orchestratore e lancia un sottoagente per fase; vedi tutto e puoi intervenire. Se modifichi un file in `.claude/agents/` riavvia `claude`, perché gli agenti si caricano all'avvio.

### Modo B: non interattivo (lascialo lavorare)
```bash
python3 tools/run_pipeline.py                    # una sessione nuova per fase, tutto su disco
nohup python3 tools/run_pipeline.py > run.log 2>&1 &
tail -f run.log                                  # Ctrl+C esce dal tail, non ferma la pipeline
```
Per riprovare da solo quando finisce il limite d'uso (codice di uscita 75), ogni 30 minuti:
```bash
until python3 tools/run_pipeline.py; do
  rc=$?
  [ "$rc" -eq 75 ] || break
  echo "limite raggiunto: riprovo tra 30 minuti"; sleep 1800
done
```
Codici di uscita: **0** finita · **2** fase fallita o bloccata · **3** cartella non approvata (vedi sopra) · **10** attende la tua approvazione · **75** limite d'uso.
Opzioni utili: `--until P3`, `--model sonnet`, `--max-turns`, `--max-budget-usd`, `--timeout-min`, `--no-gates`, `--dry-run`.

---

## 6. Se si blocca: ripartire

**Regola d'oro: non ricominciare mai da capo.** Lo stato è in `agentic/pipeline_state.json` e i report sono in `reports/`.

| Cosa è successo | Cosa fare |
|---|---|
| Limite di token/uso a metà fase | Il driver esce con 75 e rimette la fase in coda senza contare il tentativo. Rilancia lo stesso comando. L'agente riparte da `reports/<ID>_<nome>.progress.md` e da `results/<ID>/`. |
| Sessione interattiva piena o chiusa | Apri `claude` e digita `/resume`. Fa `brief`, recupera le fasi interrotte, chiude i gate dovuti e continua. |
| Il computer si è spento o il processo è stato ucciso | `python3 tools/run_pipeline.py` all'avvio rimette in coda le fasi rimaste «running». In interattivo: `/resume`. |
| Una fase è fallita 3 volte | Il driver si ferma con codice 2. Leggi `agentic/RUNLOG.md` e il report; correggi la causa; poi `python3 tools/state.py reset <ID> --attempts` e rilancia. |
| Vuoi rifare una fase | `python3 tools/state.py reset <ID> --attempts`, poi rilancia. Le fasi successive già concluse restano: azzerale con lo stesso comando se dipendono dalla fase rifatta. |
| Lo stato sembra rovinato | Ripristina da `agentic/backups/<data>_prima_<fase>/pipeline_state.json` o con `git log` / `git checkout <commit> -- agentic reports results`. |

Se i comandi `/resume` non sono disponibili, incolla questo prompt in una nuova sessione di Claude Code dentro la cartella del progetto:
```
Riprendi la pipeline EchoNext. Esegui `python3 tools/state.py brief`. Se ci sono fasi 'running' esegui `python3 tools/state.py recover`.
Applica i gate dovuti descritti in agentic/PLAYBOOK.md solo con tools/state.py. Poi continua la fase successiva con il sottoagente indicato nel campo `agent`.
Non rifare ciò che è già in reports/ e results/. Non aprire il test set. Fermati prima di P7 e chiedimi l'approvazione.
```

---

## 7. Leggere i risultati e approvare il test

Ordine di lettura consigliato:
1. `reports/P8_synthesis.md` (copia in `FINAL_REPORT.md`): la risposta spiegata, con numeri e limiti.
2. `reports/P6_review.md`: cosa ha trovato il revisore (leakage, numeri non riproducibili, affermazioni deboli).
3. `agentic/DECISIONS.md`: perché l'orchestratore ha aggiunto o saltato fasi. `results/registry.csv`: quante configurazioni sono state provate (più sono, più cautela).
4. I report delle singole fasi, partendo dagli *Handoff*.

**Prima del test (P7)** il driver si ferma. Guarda `agentic/FROZEN.json` (quali candidati) e `reports/P6_review.md` (verdetto). Se ti convincono: `python3 tools/state.py approve P7` e rilancia.
Il test si apre **una sola volta**: se un candidato va male lo si riporta così com'è, perché ritoccare il modello dopo aver visto il test significherebbe selezionare sul test.

**Per riportare ai partner**, usa le classi del PLAYBOOK: «buona» (AUROC test ≥ 0.80 con limite inferiore dell'IC ≥ 0.75), «moderata» (0.70-0.80), «debole». Se non ci sono candidati credibili,
il risultato è esso stesso utile: le sole misure ECG tabulari non bastano e la strada sono le forme d'onda. Ricorda sempre che SHD non coincide con scompenso cardiaco.

---

## 8. Personalizzare

- **Soglie**: `python3 tools/state.py set-config thresholds.good_auroc 0.85` (vedi l'elenco con `python3 tools/state.py status --json`). Le *definizioni* si cambiano in `agentic/PLAYBOOK.md`.
- **Modelli e costi**: in ogni `.claude/agents/*.md` la riga `model:` accetta `sonnet`, `opus`, `haiku`. Il revisore usa `opus`, gli altri `sonnet`. Un modello più piccolo costa meno ma sbaglia di più; non abbassare il revisore. Per forzarlo ovunque: `--model sonnet`.
- **Limiti**: `maxTurns` nei file degli agenti, `--max-turns` e `--max-budget-usd` nel driver.
- **Aggiungere un agente**: crea `.claude/agents/nome.md` (frontmatter `name`, `description`, `tools`, `model`), aggiungi la sua sezione al PLAYBOOK e inseriscilo con
  `python3 tools/state.py add-phase --id P9 --name nome --agent nome --goal "..." --depends-on P6 --gate`.
- **Permessi**: `.claude/settings.json` (cosa gli agenti possono scrivere e lanciare). In headless ciò che non è nell'elenco viene negato.
- **Obiettivo diverso** (es. un solo target): cambia `config.primary_target` e le sezioni del PLAYBOOK.

---

## 9. Cosa non è stato verificato

- **Collaudo con Claude reale (parziale).** Su dati sintetici sono state eseguite con la CLI vera le fasi **P0 (audit)** e **P1 (baseline)** con i relativi gate (circa 0,6 USD in tutto): gli agenti hanno prodotto report validi, usato `lib.show()`, registrato le configurazioni con `log-exp`, salvato il candidato e segnalato correttamente «nessun segnale» (i dati sintetici non ne hanno). **Le fasi P2–P8 sono state provate solo con un finto `claude`** (stato, driver, ripresa dopo errore/limite/report incompleto, pausa per approvazione, blocco del test set, valutazione finale, snapshot, checkpoint git). I loro prompt vanno quindi verificati con il passo 4B e probabilmente rifiniti dopo la prima esecuzione vera. Costi e tempi su dati reali (e CNN su GPU in P4) non sono stimati.
- L'output JSON di `claude -p` (`result`, `session_id`, `total_cost_usd`, `is_error`) è letto in modo tollerante: i costi mostrati in `RUNLOG.md` sono una stima e possono risultare `n/d`.
- Le soglie del PLAYBOOK sono valori iniziali ragionevoli, non standard: discutile con i colleghi.
- Gli agenti sono LLM: possono sbagliare o inventare. Per questo ci sono il revisore, le protezioni nel codice e il registro delle configurazioni, ma **la lettura finale spetta a una persona** (e, per il significato clinico, a un cardiologo).
- Informazioni su Claude Code (installazione, subagenti, comandi, modalità headless) e sulla politica PhysioNet: documentazione consultata il 6 novembre 2026; possono cambiare.
